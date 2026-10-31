"""The three strategies behind one interface: Engine.answer(question, strategy, kg_version)."""

import json
import math
import os
import re
import time
from functools import lru_cache
from pathlib import Path

from analytics import queries as AQ
from analytics.db import check_version, named_queries

from . import NOT_FOUND, PROMPT_VERSION, STRATEGIES
from .cache import LLMCache, cache_key
from .linking import keywords, link
from .prompts import (ANSWER_SCHEMA, ANSWER_SYSTEM, CYPHER_SCHEMA, CYPHER_SYSTEM, VERBALIZE_SYSTEM,
                      answer_user, cypher_user, verbalize_user)
from .safety import UnsafeCypher, check_cypher

QUERIES = named_queries((Path(__file__).parent / "queries.cypher").read_text())
TOKEN_BUDGET = 2500                       # context tokens (estimated as words / 0.75)
PASSAGE_WORDS = 140
K_CHUNKS = 5

# Question words -> relation types to rank first in the k-hop expansion.
RELATION_CUES = {
    "DEVELOPED_BY": r"develop|built|build|creat|made|make|maker|founded|founder",
    "WORKS_AT": r"work|employ|ceo|founder|researcher|scientist|staff|people|who leads|head of|cto",
    "USES": r"\buse|using|technolog|rel(y|ies) on", "TRAINED_ON": r"train", "AUTHORED": r"author|wrote|paper",
    "FUNDED_BY": r"fund|invest|back(ed|er)", "ACQUIRED": r"acqui|bought|buy", "LOCATED_IN": r"where|located|based|headquarter",
    "COMPETES_WITH": r"compet|rival", "PARTNERS_WITH": r"partner|collaborat", "BASED_ON": r"based on|derived|built on",
}
# Aggregation intent -> graph aggregate query (with provenance doc ids).
AGG_CUES = [(r"\bmost\b|\bhow many\b|\btop\b|\blargest\b|\bnumber of\b|\bcount\b", r"model", "agg_model_developers"),
            (r"\bmost\b|\bhow many\b|\btop\b|\bpopular\b|\bwidely\b", r"technolog", "agg_technologies"),
            (r"\bmost\b|\bhow many\b|\blargest\b|\bbiggest\b", r"people|employ|researcher|staff", "agg_employers")]


def short_id(doc_id: str) -> str:
    return doc_id[:8]


def est_tokens(text: str) -> int:
    return math.ceil(len(text.split()) / 0.75)


def preferred_relations(question: str) -> list[str]:
    q = question.lower()
    return [rel for rel, cue in RELATION_CUES.items() if re.search(cue, q)]


def aggregate_intents(question: str) -> list[str]:
    q = question.lower()
    return [name for cue, topic, name in AGG_CUES if re.search(cue, q) and re.search(topic, q)]


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion of several ranked id lists."""
    score = {}
    for ranking in rankings:
        for i, x in enumerate(ranking):
            score[x] = score.get(x, 0.0) + 1.0 / (k + i + 1)
    return sorted(score, key=lambda x: (-score[x], x))


def excerpt(text: str, words: int = PASSAGE_WORDS) -> str:
    w = " ".join(text.split()).split(" ")
    return " ".join(w[:words]) + (" ..." if len(w) > words else "")


def lucene_terms(words: list[str]) -> str:
    return " OR ".join(re.sub(r'([+\-&|!(){}\[\]^"~*?:\\/])', r"\\\1", w) for w in words)


class Engine:
    def __init__(self, driver, llm, settings, cache: LLMCache, embed_fn):
        self.driver, self.llm, self.s, self.cache, self.embed = driver, llm, settings, cache, embed_fn

    # ------------------------------------------------------------------ db helpers
    def run(self, name: str, kg_version: str = None, **params) -> list[dict]:
        if kg_version is not None:
            params["kg_version"] = kg_version
        with self.driver.session() as s:
            return [r.data() for r in s.run(QUERIES[name], parameters=params)]

    def docs(self, doc_ids) -> dict:
        ids = sorted({d for d in doc_ids if d})
        return {r["doc_id"]: r for r in self.run("documents", ids=ids)} if ids else {}

    # ------------------------------------------------------------------ LLM with cache
    def llm_json(self, step: str, strategy: str, question: str, kg_version: str, system: str, user: str,
                 schema: dict, stats: dict) -> dict:
        key = cache_key(strategy, question, kg_version, PROMPT_VERSION, step)
        hit = self.cache.get(key)
        if hit is None:
            raw, usage = self.llm.chat_json(system=system, user=user, schema=schema, schema_name=f"graphrag_{step}",
                                            purpose=f"graphrag-{STRATEGIES[strategy]}-{step}",
                                            ref=f"{kg_version}:{question[:60]}", model=self.s.answer_model)
            hit = {"raw": raw, "usage": usage, "model": self.s.answer_model, "step": step}
            self.cache.put(key, hit)
            stats["llm_live_s"] += usage["latency_s"]
        else:
            stats["cache_hits"] += 1
        stats["llm_calls"] += 1
        stats["tokens_in"] += hit["usage"]["input_tokens"]
        stats["tokens_out"] += hit["usage"]["output_tokens"] + hit["usage"].get("thinking_tokens", 0)
        stats["llm_latency_s"] += hit["usage"]["latency_s"]
        return json.loads(hit["raw"])

    # ------------------------------------------------------------------ retrieval
    def vector_chunks(self, question: str, k: int = K_CHUNKS) -> list[str]:
        return [r["chunk_id"] for r in self.run("vector_chunks", k=k, embedding=self.embed(question))]

    def hybrid_chunks(self, question: str, entity_ids: list[str], kg_version: str, k: int = K_CHUNKS) -> list[str]:
        rankings = [self.vector_chunks(question, 2 * k)]
        kw = keywords(question)
        if kw:
            rankings.append([r["chunk_id"] for r in self.run("fulltext_chunks", query=lucene_terms(kw), k=2 * k)])
        if entity_ids:
            mention = [r["chunk_id"] for r in self.run("mention_chunks", kg_version, ids=entity_ids, k=2 * k)]
            rankings += [mention, mention]          # graph evidence counts double
        return rrf(rankings)[:k]

    def passages(self, chunk_ids: list[str]) -> list[dict]:
        rows = {r["chunk_id"]: r for r in self.run("chunk_details", ids=chunk_ids)}
        return [rows[c] for c in chunk_ids if c in rows]

    def triples(self, question: str, entities: list[dict], kg_version: str) -> list[dict]:
        """1-hop edges of the linked entities, then 1 more hop from the best neighbours."""
        prefer = preferred_relations(question)
        ids = [e["id"] for e in entities]
        hop1 = AQ.entity_triples(self.driver, ids, kg_version, prefer, per_entity=30)
        seen = set(ids)
        nbrs = []
        for t in sorted(hop1, key=lambda t: (t["relation"] not in prefer, -t["mentions"])):
            for nid in (t["source_id"], t["target_id"]):
                if nid not in seen:
                    seen.add(nid)
                    nbrs.append(nid)
        hop2 = AQ.entity_triples(self.driver, nbrs[:5], kg_version, prefer, per_entity=15) if nbrs else []
        out = {}
        for hop, rows in ((1, hop1), (2, hop2)):
            for t in rows:
                key = (t["source_id"], t["relation"], t["target_id"])
                if key in out:
                    continue
                score = (1.0 if hop == 1 else 0.6) * (1.5 if t["relation"] in prefer else 1.0) \
                    * (1 + math.log(t["mentions"] or 1)) * (t["confidence"] or 0.5)
                out[key] = {**t, "hop": hop, "score": round(score, 4)}
        return sorted(out.values(), key=lambda t: -t["score"])

    def aggregates(self, question: str, kg_version: str) -> list[dict]:
        rows = []
        for name in aggregate_intents(question):
            rows += [{**r, "query": name} for r in self.run(name, kg_version, k=8)]
        return rows

    # ------------------------------------------------------------------ context
    def build_context(self, entities, triples, aggs, passages) -> tuple[str, dict]:
        """Sections in priority order, truncated to TOKEN_BUDGET. Returns (context, evidence by short id)."""
        evidence, lines, used = {}, [], 0

        def add(line: str) -> bool:
            nonlocal used
            t = est_tokens(line)
            if used + t > TOKEN_BUDGET:
                return False
            lines.append(line)
            used += t
            return True

        if entities:
            add("## Linked entities")
            for e in entities:
                add(f"- {e['name']} ({e['label']}){': ' + e['description'] if e.get('description') else ''}")
        if aggs:
            add("## Graph aggregates (counted over the whole knowledge graph)")
            for a in aggs:
                ids = [short_id(d) for d in a["doc_ids"] if d]
                for d, sid in zip(a["doc_ids"], ids):
                    evidence.setdefault(sid, {"doc_id": d, "evidence_text": f"{a['name']}: {a['n']} {a['what']}"})
                add(f"- {a['name']}: {a['n']} {a['what']} (e.g. {', '.join(a['examples'][:4])}) "
                    f"{' '.join('[' + i + ']' for i in ids)}")
        if triples:
            add("## Facts (graph relations, each with its evidence sentence and source id)")
            for t in triples:
                sid = short_id(t["doc_id"])
                line = (f"- ({t['source']}:{t['source_label']}) -[{t['relation']}]-> ({t['target']}:{t['target_label']})"
                        f" | evidence: \"{t['evidence']}\" [{sid}]")
                if not add(line):
                    break
                evidence.setdefault(sid, {"doc_id": t["doc_id"], "evidence_text": t["evidence"]})
        if passages:
            add("## Passages (source text)")
            for p in passages:
                sid = short_id(p["doc_id"])
                if not add(f"[{sid}] {p['title']}: {excerpt(p['text'])}"):
                    break
                evidence.setdefault(sid, {"doc_id": p["doc_id"], "evidence_text": excerpt(p["text"], 60)})
        return "\n".join(lines), evidence

    def sources(self, cited: list[str], evidence: dict, answer_text: str) -> list[dict]:
        ids = list(dict.fromkeys([c.strip("[] ") for c in cited] + re.findall(r"\[([0-9a-f]{8})\]", answer_text)))
        ids = [i for i in ids if i in evidence]
        docs = self.docs(evidence[i]["doc_id"] for i in ids)
        return [{"doc_id": evidence[i]["doc_id"], "short_id": i, "title": docs.get(evidence[i]["doc_id"], {}).get("title"),
                 "url": docs.get(evidence[i]["doc_id"], {}).get("url"), "evidence_text": evidence[i]["evidence_text"]}
                for i in ids]

    def finish(self, out: dict, parsed: dict, evidence: dict) -> dict:
        text = (parsed.get("answer") or "").strip()
        if not parsed.get("found", True) or not text or text.lower().startswith("not found"):
            out.update(answer=NOT_FOUND, sources=[], found=False)
            return out
        # Citations must point at ids that were in the context; anything else (a number, an invented
        # id) is removed from the text and reported, never shown as a source (RQ5).
        invalid = [c for c in re.findall(r"\[([^\[\]]+)\]", text)
                   if any(i.strip() not in evidence for i in c.split(","))]
        for c in invalid:
            keep = [i.strip() for i in c.split(",") if i.strip() in evidence]
            text = text.replace(f"[{c}]", "[" + ", ".join(keep) + "]" if keep else "")
        text = re.sub(r"\s+([.,;])", r"\1", re.sub(r"\s{2,}", " ", text)).strip()
        cited = [c for c in parsed.get("cited_ids", []) if c.strip("[] ") in evidence]
        out.update(answer=text, sources=[{**x, "cited": True} for x in self.sources(cited, evidence, text)],
                   found=True, invalid_citations=invalid)
        return out

    def entity_provenance(self, rows: list[dict], kg_version: str, k: int = 2) -> list[dict]:
        """Rows without doc ids (e.g. an entity's description): supporting sources are the chunks that
        mention the entities named in the rows. Marked cited=False: the model did not cite them."""
        names = sorted({v.lower() for r in rows for v in r.values() if isinstance(v, str) and 0 < len(v) < 120})
        ids = [e["id"] for e in self.run("link_exact", kg_version, candidates=names)] if names else []
        if not ids:
            return []
        chunks = self.passages([r["chunk_id"] for r in self.run("mention_chunks", kg_version, ids=ids, k=k)])
        return [{"doc_id": c["doc_id"], "short_id": short_id(c["doc_id"]), "title": c["title"], "url": c["url"],
                 "evidence_text": excerpt(c["text"], 60), "cited": False, "kind": "entity_mention"} for c in chunks]

    # ------------------------------------------------------------------ strategies
    def strategy_a(self, question, kg_version, out, stats):
        chunks = self.passages(self.vector_chunks(question))
        context, evidence = self.build_context([], [], [], chunks)
        out["context"] = context
        parsed = self.llm_json("answer", "A", question, kg_version, ANSWER_SYSTEM, answer_user(question, context),
                               ANSWER_SCHEMA, stats)
        return self.finish(out, parsed, evidence)

    def strategy_b(self, question, kg_version, out, stats):
        entities = link(self.driver, self.run, question, kg_version,
                        search=lambda c: AQ.entity_search(self.driver, c, kg_version, 3))
        out["entities_used"] = [{k: e[k] for k in ("id", "name", "label", "matched")} for e in entities]
        triples = self.triples(question, entities, kg_version) if entities else []
        aggs = self.aggregates(question, kg_version)
        chunks = self.passages(self.hybrid_chunks(question, [e["id"] for e in entities], kg_version))
        context, evidence = self.build_context(entities, triples, aggs, chunks)
        out["context"] = context
        out["retrieval"] = {"triples": len(triples), "aggregates": len(aggs), "passages": len(chunks)}
        parsed = self.llm_json("answer", "B", question, kg_version, ANSWER_SYSTEM, answer_user(question, context),
                               ANSWER_SCHEMA, stats)
        return self.finish(out, parsed, evidence)

    def execute_read(self, cypher: str, kg_version: str) -> list[dict]:
        from neo4j import unit_of_work

        @unit_of_work(timeout=10)
        def work(tx):
            return [r.data() for r in tx.run(cypher, kg_version=kg_version)]

        with self.driver.session() as s:
            return s.execute_read(work)

    def strategy_c(self, question, kg_version, out, stats):
        error, rows, cypher = "", None, None
        for attempt, step in enumerate(("cypher", "cypher_retry")):
            parsed = self.llm_json(step, "C", question, kg_version, CYPHER_SYSTEM, cypher_user(question, error),
                                   CYPHER_SCHEMA, stats)
            cypher = parsed.get("cypher", "")
            try:
                cypher = check_cypher(cypher)
                rows = self.execute_read(cypher, kg_version)
                break
            except UnsafeCypher as e:
                error = f"rejected by the safety filter: {e}"
            except Exception as e:                           # syntax error, timeout, unknown property
                error = f"{type(e).__name__}: {str(e)[:300]}"
            out["cypher_errors"] = out.get("cypher_errors", []) + [error]
        out["cypher"] = cypher
        out["rows"] = len(rows or [])
        if not rows:                                          # nothing to verbalize: no LLM call
            out["context"] = "(no rows)"
            return out | {"answer": NOT_FOUND, "sources": [], "found": False}
        rows = rows[:40]
        doc_ids = set()
        for r in rows:
            for k, v in r.items():
                if "doc" in k.lower():
                    doc_ids.update(v if isinstance(v, list) else [v])
        doc_ids.discard(None)
        evidence = {}
        for r in rows:
            ev = next((v for k, v in r.items() if "evidence" in k.lower() and isinstance(v, str)), None)
            for k, v in r.items():
                if "doc" in k.lower():
                    for d in (v if isinstance(v, list) else [v]):
                        if d:
                            evidence.setdefault(short_id(d), {"doc_id": d, "evidence_text": ev or json.dumps(
                                {kk: vv for kk, vv in r.items() if "doc" not in kk.lower()}, default=str)[:300]})
        shown = [{k: ([short_id(x) for x in v if x] if isinstance(v, list) and "doc" in k.lower()
                      else short_id(v) if "doc" in k.lower() and isinstance(v, str) else v)
                  for k, v in r.items()} for r in rows]
        rows_json = json.dumps(shown, ensure_ascii=False, default=str)[:6000]
        out["context"] = rows_json
        parsed = self.llm_json("verbalize", "C", question, kg_version, VERBALIZE_SYSTEM,
                               verbalize_user(question, cypher, rows_json), ANSWER_SCHEMA, stats)
        out = self.finish(out, parsed, evidence)
        if out["found"] and not out["sources"]:
            out["sources"] = self.entity_provenance(rows, kg_version)
        return out

    # ------------------------------------------------------------------ interface
    def answer(self, question: str, strategy: str = "B", kg_version: str = "resolved") -> dict:
        strategy = strategy.upper()
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {list(STRATEGIES)}")
        check_version(kg_version)
        t0 = time.perf_counter()
        stats = {"llm_calls": 0, "cache_hits": 0, "tokens_in": 0, "tokens_out": 0, "llm_latency_s": 0.0,
                 "llm_live_s": 0.0}
        out = {"question": question, "strategy": strategy, "strategy_name": STRATEGIES[strategy],
               "kg_version": kg_version, "answer": "", "sources": [], "context": "", "cypher": None,
               "entities_used": [], "model": self.s.answer_model, "prompt_version": PROMPT_VERSION}
        out = {"A": self.strategy_a, "B": self.strategy_b, "C": self.strategy_c}[strategy](question, kg_version, out, stats)
        wall = time.perf_counter() - t0
        # latency = retrieval/DB time + the LLM time of each call (measured live, or stored with the
        # cached result), so cached re-runs report the same latency a live run would have.
        retrieval_s = max(wall - stats["llm_live_s"], 0.0)
        out.update(tokens_in=stats["tokens_in"], tokens_out=stats["tokens_out"], llm_calls=stats["llm_calls"],
                   cache_hits=stats["cache_hits"], retrieval_s=round(retrieval_s, 3),
                   latency_s=round(retrieval_s + stats["llm_latency_s"], 3), wall_s=round(wall, 3))
        return out


@lru_cache(maxsize=1)
def default_engine() -> Engine:
    from analytics.db import get_driver
    from extraction.config import settings
    from extraction.llm import LLMClient
    from fastembed import TextEmbedding
    import numpy as np

    s = settings()
    model = TextEmbedding(model_name=os.environ.get("FASTEMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                          cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"), threads=2)

    def embed(text: str) -> list[float]:
        v = np.array(list(model.embed([text]))[0], dtype=np.float32)
        return (v / max(float(np.linalg.norm(v)), 1e-12)).astype(float).tolist()

    driver = get_driver()
    with driver.session() as session:                         # also in neo4j/schema.cypher (idempotent)
        session.run("CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]").consume()
        session.run("CALL db.awaitIndexes(300)").consume()
    return Engine(driver, LLMClient(s), s, LLMCache(os.environ.get("GRAPHRAG_CACHE_DIR", "/data/cache/graphrag")), embed)
