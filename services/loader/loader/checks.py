"""Phase 3 verification: Neo4j vs gold counts, provenance, vector search, raw vs resolved sanity.
All queries come from neo4j/checks.cypher and filter on kg_version."""

import re

from . import ENTITY_LABELS, RELATION_TYPES, VERSIONS
from .cypher import named_queries

GOLD_TABLES = ([f"nodes_{l}" for l in ENTITY_LABELS] + ["nodes_Document", "nodes_Chunk"]
               + [f"edges_{r}" for r in RELATION_TYPES] + ["edges_MENTIONS", "edges_PART_OF"])
VECTOR_QUERY = "large language model developed by Google"
LLM_TERMS = re.compile(r"language model|\bLLMs?\b", re.I)                # relevant to the query topic
GOOGLE_TERMS = re.compile(r"google|gemini|deepmind|bard|palm|gemma|bert\b|t5\b", re.I)   # reported only


def table_to_neo4j(name: str) -> str:
    return name.split("_", 1)[1]                    # nodes_Person -> Person, edges_USES -> USES


def compare_counts(gold: dict, neo: dict) -> list[dict]:
    """gold/neo: {(table_or_label, kg_version): n}. Every gold row must match; anything in Neo4j
    that gold lacks is a mismatch too."""
    rows = []
    keys = {(table_to_neo4j(t), v) for t, v in gold} | set(neo)
    for name, v in sorted(keys):
        g = next((n for (t, gv), n in gold.items() if table_to_neo4j(t) == name and gv == v), 0)
        n = neo.get((name, v), 0)
        rows.append({"name": name, "kg_version": v, "gold": g, "neo4j": n, "ok": g == n})
    return rows


def _q(driver, query: str, **params) -> list[dict]:
    with driver.session() as s:
        return [r.data() for r in s.run(query, **params)]


def run_checks(driver, spark, embedder, checks_text: str) -> dict:
    from .gold import counts as gold_counts

    Q = named_queries(checks_text)
    neo = {(r["label"], r["kg_version"]): r["n"] for r in _q(driver, Q["node_counts"])}
    neo |= {(r["type"], r["kg_version"]): r["n"] for r in _q(driver, Q["edge_counts"])}
    count_rows = compare_counts(gold_counts(spark, GOLD_TABLES), neo)

    provenance = {v: (_q(driver, Q["provenance"], kg_version=v) or [None])[0] for v in VERSIONS}

    hits = _q(driver, Q["vector_search"], k=5, embedding=embedder.embed_query(VECTOR_QUERY))
    full = {h["chunk_id"]: h for h in _q(driver, "MATCH (c:Chunk) WHERE c.id IN $ids "
                                                "RETURN c.id AS chunk_id, c.text AS text",
                                         ids=[h["chunk_id"] for h in hits])}
    on_topic = lambda h, rx: bool(rx.search(full[h["chunk_id"]]["text"] + " " + h["document"]))
    relevant = sum(on_topic(h, LLM_TERMS) for h in hits)
    google = sum(on_topic(h, GOOGLE_TERMS) for h in hits)

    sanity = {}
    for v in VERSIONS:
        g = f"phase3_{v}"
        _q(driver, Q["gds_drop"], graph=g)
        proj = _q(driver, Q["gds_project"], kg_version=v, graph=g)[0]
        wcc = _q(driver, Q["gds_wcc"], graph=g)[0]
        _q(driver, Q["gds_drop"], graph=g)
        entities = proj["nodes"]
        edges = _q(driver, Q["relation_edges"], kg_version=v)[0]["n"]
        sanity[v] = {"entities": entities, "relation_edges": edges,
                     "avg_degree": round(2 * edges / entities, 3) if entities else 0,
                     "components": wcc["components"], "largest_component": wcc["largest"],
                     "largest_component_pct": round(100 * wcc["largest"] / entities, 1) if entities else 0,
                     "orphans": _q(driver, Q["orphans"], kg_version=v)[0]["n"],
                     "top_degree": _q(driver, Q["top_degree"], kg_version=v)}
        sanity[v]["orphans_pct"] = round(100 * sanity[v]["orphans"] / entities, 1) if entities else 0

    return {"counts": count_rows, "counts_ok": all(r["ok"] for r in count_rows),
            "provenance": provenance,
            "provenance_ok": all(p and p["evidence"] and p["url"] for p in provenance.values()),
            "vector_query": VECTOR_QUERY, "vector_hits": hits, "vector_relevant": relevant,
            "vector_google_related": google, "vector_ok": len(hits) == 5 and relevant == 5,
            "sanity": sanity}


def report(r: dict) -> str:
    out = ["== Neo4j vs gold counts"]
    out.append(f"  {'name':<15}{'kg_version':<11}{'gold':>8}{'neo4j':>8}  ok")
    for c in r["counts"]:
        out.append(f"  {c['name']:<15}{c['kg_version']:<11}{c['gold']:>8}{c['neo4j']:>8}  {'OK' if c['ok'] else 'MISMATCH'}")
    out.append("== Provenance (most-mentioned DEVELOPED_BY edge)")
    for v, p in r["provenance"].items():
        out.append(f"  [{v}] ({p['source']})-[:{p['relation']}]->({p['target']})  mentions={p['mentions']} "
                   f"confidence={p['confidence']} model={p['extractor_model']}/{p['prompt_version']}")
        out.append(f"        evidence: \"{p['evidence']}\"")
        out.append(f"        chunk {p['chunk_id']} of \"{p['document']}\"  {p['url']}")
    out.append(f"== Vector search: top-5 chunks for \"{r['vector_query']}\"  ({r['vector_relevant']}/5 about "
               f"language models, {r['vector_google_related']}/5 mention Google-related terms)")
    for h in r["vector_hits"]:
        out.append(f"  {h['score']:.4f}  [{h['source']}] {h['document'][:45]:<45} | {h['text'][:80]}")
    out.append("== Raw vs resolved graph (entity-relation edges only; components via GDS WCC)")
    keys = [("entities", "entities"), ("relation_edges", "relation edges"), ("avg_degree", "avg degree"),
            ("components", "connected components"), ("largest_component", "largest component"),
            ("largest_component_pct", "largest component %"), ("orphans", "orphan entities"),
            ("orphans_pct", "orphan entities %")]
    s = r["sanity"]
    out.append(f"  {'metric':<24}{'raw':>10}{'resolved':>10}{'change':>10}")
    for k, name in keys:
        a, b = s["raw"][k], s["resolved"][k]
        ch = f"{100 * (b - a) / a:+.1f}%" if a and not k.endswith("pct") else ""
        out.append(f"  {name:<24}{a:>10}{b:>10}{ch:>10}")
    out.append("  top 10 entities by degree:")
    out.append(f"  {'raw':<38}{'resolved':<38}")
    for a, b in zip(s["raw"]["top_degree"], s["resolved"]["top_degree"]):
        out.append(f"  {(a['name'][:28] + ' ' + str(a['degree'])):<38}{(b['name'][:28] + ' ' + str(b['degree'])):<38}")
    return "\n".join(out)
