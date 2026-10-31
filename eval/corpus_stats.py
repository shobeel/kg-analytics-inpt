"""Corpus and graph counts from Neo4j -> eval/results/corpus_stats.json (no LLM call), so the reports
cite measured numbers: documents per source, chunks, entities and relation edges per kg_version.

Usage: python -m eval.corpus_stats
"""

import json
from pathlib import Path

OUT = Path(__file__).parent / "results" / "corpus_stats.json"


def main() -> int:
    from analytics.db import get_driver
    with get_driver().session() as s:
        docs = {r["source"]: r["n"] for r in s.run("MATCH (d:Document) RETURN d.source AS source, count(*) AS n ORDER BY n DESC")}
        chunks = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
        ents = {r["v"]: r["n"] for r in s.run("MATCH (e:Entity) RETURN e.kg_version AS v, count(*) AS n")}
        rels = {r["v"]: r["n"] for r in s.run("MATCH (:Entity)-[r]->(:Entity) WHERE r.kg_version IS NOT NULL "
                                              "RETURN r.kg_version AS v, count(*) AS n")}
    out = {"documents": sum(docs.values()), "documents_by_source": docs, "chunks": chunks,
           "entities": ents, "relation_edges": rels}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
