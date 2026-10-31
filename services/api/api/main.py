"""Phase 7 REST API.

  GET  /health                     liveness + Neo4j reachability (no key)
  POST /ask                        GraphRAG answer: strategy A | B | C, kg_version raw | resolved   (viewer)
  GET  /analytics/top              top entities by PageRank, with Louvain community                 (viewer)
  GET  /analytics/communities      largest Louvain communities and their top members                (viewer)
  GET  /analytics/quality          RQ2 raw vs resolved graph-quality table (eval/results)           (viewer)
  GET  /graph/neighbours           small subgraph around an entity, with edge ids                   (viewer)
  GET  /provenance/{edge_id}       every mention of one edge: evidence, chunk, document, url         (viewer)
  GET  /metrics                    Prometheus text: requests, latency, errors, LLM tokens           (admin)
  POST /admin/metrics/reset        reset the in-process counters                                    (admin)

All Cypher here is fixed and parameterised, run in read transactions; LLM-written Cypher (strategy C)
goes through the Phase 5 safety filter inside the engine. Every query filters on kg_version.
"""

import csv
import os
import time
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .auth import require
from .monitoring import METRICS, log_event

KG = "^(raw|resolved)$"
RESULTS = Path(os.environ.get("EVAL_RESULTS_DIR", "/app/eval/results"))
REL_TYPES = ("DEVELOPED_BY|WORKS_AT|AUTHORED|CITES|USES|TRAINED_ON|COMPETES_WITH|PARTNERS_WITH|FUNDED_BY|"
             "ACQUIRED|LOCATED_IN|BASED_ON")

app = FastAPI(title="AI-ecosystem knowledge graph API", version="1.0")


# ------------------------------------------------------------------ dependencies (overridden in tests)
@lru_cache(maxsize=1)
def get_engine():
    from graphrag.engine import default_engine
    return default_engine()


@lru_cache(maxsize=1)
def get_driver():
    from analytics.db import get_driver as gd
    return gd()


def read(cypher: str, **params) -> list[dict]:
    def work(tx):
        return [r.data() for r in tx.run(cypher, **params)]
    with get_driver().session() as s:
        return s.execute_read(work)


# ------------------------------------------------------------------ middleware: JSON log + metrics
@app.middleware("http")
async def observe(request: Request, call_next):
    rid = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        dt = time.perf_counter() - t0
        route = request.scope.get("route")
        path = getattr(route, "path", None) or "unmatched"
        METRICS.observe(request.method, path, status, dt)
        log_event(event="request", request_id=rid, method=request.method, route=path, status=status,
                  latency_ms=round(dt * 1000, 1), role=getattr(request.state, "role", None),
                  client=request.client.host if request.client else None)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log_event(event="error", route=request.url.path, error=type(exc).__name__, detail=str(exc)[:300])
    return JSONResponse(status_code=500, content={"detail": "internal error"})


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health() -> dict:
    try:
        read("RETURN 1 AS ok")
        neo4j = "ok"
    except Exception as e:                       # report, do not crash: /health must answer
        neo4j = f"unreachable ({type(e).__name__})"
    return {"status": "ok" if neo4j == "ok" else "degraded", "neo4j": neo4j}


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    strategy: str = Field(default="B", pattern="^[ABCabc]$")
    kg_version: str = Field(default="resolved", pattern=KG)


@app.post("/ask")
def ask(req: AskRequest, role: str = Depends(require("viewer"))) -> dict:
    from extraction.llm import LLMCallFailed, QuotaExhausted
    try:
        out = get_engine().answer(req.question, req.strategy.upper(), req.kg_version)
    except QuotaExhausted:
        raise HTTPException(status_code=429, detail="daily LLM quota reached; cached questions still work")
    except LLMCallFailed as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {str(e)[:200]}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    METRICS.observe_llm(out.get("strategy", req.strategy.upper()), out)
    log_event(event="ask", strategy=out.get("strategy"), kg_version=req.kg_version, llm_calls=out.get("llm_calls"),
              cache_hits=out.get("cache_hits"), tokens_in=out.get("tokens_in"), tokens_out=out.get("tokens_out"),
              latency_s=out.get("latency_s"), found=bool(out.get("sources")))
    out.pop("context", None)                     # the full prompt context stays server-side
    return out


@app.get("/analytics/top")
def analytics_top(kg_version: str = Query("resolved", pattern=KG), limit: int = Query(15, ge=1, le=100),
                  role: str = Depends(require("viewer"))) -> list[dict]:
    p, c = f"pagerank_{kg_version}", f"community_{kg_version}"
    return read(f"""MATCH (e:Entity {{kg_version: $v}}) WHERE e.{p} IS NOT NULL
                    RETURN e.id AS id, e.name AS name, [l IN labels(e) WHERE l <> 'Entity'][0] AS label,
                           round(e.{p}, 3) AS pagerank, e.{c} AS community
                    ORDER BY pagerank DESC, name LIMIT $limit""", v=kg_version, limit=limit)


@app.get("/analytics/communities")
def analytics_communities(kg_version: str = Query("resolved", pattern=KG), limit: int = Query(10, ge=1, le=50),
                          role: str = Depends(require("viewer"))) -> list[dict]:
    p, c = f"pagerank_{kg_version}", f"community_{kg_version}"
    return read(f"""MATCH (e:Entity {{kg_version: $v}}) WHERE e.{c} IS NOT NULL
                    WITH e.{c} AS community, e ORDER BY e.{p} DESC
                    WITH community, count(e) AS size, collect(e.name)[..5] AS top_members
                    RETURN community, size, top_members ORDER BY size DESC, community LIMIT $limit""",
                v=kg_version, limit=limit)


@app.get("/analytics/quality")
def analytics_quality(role: str = Depends(require("viewer"))) -> list[dict]:
    path = RESULTS / "rq2_graph_quality.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="rq2_graph_quality.csv not found (make analytics)")
    with path.open() as f:
        return list(csv.DictReader(f))


@app.get("/graph/neighbours")
def graph_neighbours(entity: str = Query(min_length=1, max_length=200), kg_version: str = Query("resolved", pattern=KG),
                     limit: int = Query(15, ge=1, le=50), role: str = Depends(require("viewer"))) -> dict:
    hit = read("""MATCH (e:Entity {kg_version: $v}) WHERE e.id = $q OR toLower(e.name) = toLower($q)
                  RETURN e.id AS id, e.name AS name, [l IN labels(e) WHERE l <> 'Entity'][0] AS label
                  ORDER BY coalesce(e.mention_count, 0) DESC LIMIT 1""", v=kg_version, q=entity)
    if not hit:
        raise HTTPException(status_code=404, detail=f"no entity {entity!r} in the {kg_version} graph")
    edges = read(f"""MATCH (e:Entity {{id: $id, kg_version: $v}})-[r:{REL_TYPES}]-(n:Entity {{kg_version: $v}})
                     WHERE r.kg_version = $v
                     RETURN r.key AS edge_id, type(r) AS relation, startNode(r).id AS source, endNode(r).id AS target,
                            n.id AS id, n.name AS name, [l IN labels(n) WHERE l <> 'Entity'][0] AS label,
                            r.mention_count AS mentions, r.evidence_text AS evidence
                     ORDER BY mentions DESC, name LIMIT $limit""", id=hit[0]["id"], v=kg_version, limit=limit)
    return {"center": hit[0], "edges": edges}


@app.get("/provenance/{edge_id:path}")
def provenance(edge_id: str, kg_version: str = Query("resolved", pattern=KG),
               role: str = Depends(require("viewer"))) -> dict:
    parts = edge_id.split("|")
    if len(parts) != 3 or parts[1] not in REL_TYPES.split("|"):
        raise HTTPException(status_code=400, detail="edge_id must be '<source id>|<RELATION>|<target id>'")
    src, rel, tgt = parts
    rows = read(f"""MATCH (a:Entity {{id: $src, kg_version: $v}})-[r:{rel} {{kg_version: $v}}]->(b:Entity {{id: $tgt, kg_version: $v}})
                    UNWIND range(0, size(r.chunk_ids) - 1) AS i
                    MATCH (c:Chunk {{id: r.chunk_ids[i]}})-[:PART_OF]->(d:Document)
                    RETURN a.name AS source, type(r) AS relation, b.name AS target, r.evidence_texts[i] AS evidence,
                           r.confidences[i] AS confidence, r.extractor_model AS extractor_model,
                           c.id AS chunk_id, d.id AS doc_id, d.title AS document, d.url AS url
                    ORDER BY confidence DESC, chunk_id""", src=src, tgt=tgt, v=kg_version)
    if not rows:
        raise HTTPException(status_code=404, detail="edge not found in this graph version")
    return {"edge_id": edge_id, "kg_version": kg_version, "mentions": rows}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics(role: str = Depends(require("admin"))) -> str:
    return METRICS.prometheus()


@app.post("/admin/metrics/reset")
def metrics_reset(role: str = Depends(require("admin"))) -> dict:
    METRICS.reset()
    log_event(event="metrics_reset")
    return {"reset": True}
