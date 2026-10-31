"""python -m analytics {queries|algorithms|quality|all}: the Phase 4 runs used by make targets."""

import argparse
import json
import sys
from pathlib import Path

from . import algorithms as A
from . import quality as QL
from . import queries as Q
from .db import KG_VERSIONS, get_driver, run

RESULTS = Path(__file__).resolve().parent.parent / "eval" / "results"
MAX_SECONDS = 2.0


def library_cases(d, v: str) -> list[tuple[str, dict, str]]:
    """(query name, params, relation) for every library query, with real ids from this version."""
    ids = {n: Q.find(d, n, v)["id"] for n in ("Sam Altman", "Microsoft", "OpenAI")}
    edge = Q.sample_edges(d, "DEVELOPED_BY", v, 1)[0]
    return [("entity_by_name", {"name": "Google DeepMind", "limit": 5}, None),
            ("entity_search", {"query": Q.lucene_query("deepmnd"), "limit": 5}, None),
            ("neighbours", {"id": ids["OpenAI"], "limit": 50}, None),
            ("paths_2hop", {"source": ids["Sam Altman"], "target": ids["Microsoft"], "limit": 25}, None),
            ("paths_3hop", {"source": ids["Sam Altman"], "target": ids["Microsoft"], "limit": 25}, None),
            ("shortest_path", {"source": ids["Sam Altman"], "target": ids["Microsoft"]}, None),
            ("top_model_developers", {"limit": 10}, None), ("top_technologies", {"limit": 10}, None),
            ("top_papers", {"limit": 10}, None), ("top_affiliations", {"limit": 10}, None),
            ("top_by_documents", {"limit": 10}, None), ("cooccurrence", {"min_chunks": 2, "limit": 20}, None),
            ("cooccurring_with", {"id": ids["OpenAI"], "limit": 20}, None),
            ("edge_provenance", {"source": edge["source"], "target": edge["target"]}, "DEVELOPED_BY")]


def check_queries(d) -> bool:
    ok = True
    print(f"{'query':<22}{'version':<10}{'rows':>5}{'seconds':>9}  first row")
    for v in KG_VERSIONS:
        for name, params, rel in library_cases(d, v):
            Q.timed(d, name, v, rel, **params)                         # warm the plan cache
            rows, t = Q.timed(d, name, v, rel, **params)
            good = bool(rows) and t < MAX_SECONDS
            ok &= good
            first = json.dumps(rows[0], ensure_ascii=False, default=str)[:90] if rows else "-"
            print(f"{name:<22}{v:<10}{len(rows):>5}{t:>9.3f}  {'' if good else 'FAIL '}{first}")
    print("QUERIES_OK" if ok else "QUERIES_FAILED")
    return ok


def run_algorithms(d) -> dict:
    out = {}
    for v in KG_VERSIONS:
        r = A.run_all(d, v)
        out[v] = r
        print(f"[{v}] projection {r['projection_directed']['nodes']} nodes / {r['projection_directed']['relationships']} "
              f"edges | PageRank {r['pagerank']['ranIterations']} it (converged={r['pagerank']['didConverge']}) | "
              f"Louvain {r['louvain']['communityCount']} communities, modularity {r['louvain']['modularity']:.3f} | "
              f"WCC {r['wcc']['componentCount']} components, LCC {r['lcc']['size']} ({r['lcc']['pct']}%) | "
              f"similarity for {r['node_similarity']['entities_with_similar']} entities | {r['seconds']}s")
        written = run(d, f"""MATCH (e:Entity {{kg_version: $v}})
            RETURN count(e) AS n, count(e.pagerank_{v}) AS pagerank, count(e.degree_{v}) AS degree,
                   count(e.betweenness_{v}) AS betweenness, count(e.community_{v}) AS community,
                   count(e.component_{v}) AS component, count(e.similar_{v}) AS similar""", v=v)[0]
        print(f"[{v}] properties written: {written}")
        out[v]["written"] = written
        print(f"[{v}] top-5 PageRank: " + ", ".join(f"{x['name']} {x['pagerank']}" for x in A.top(d, "pagerank", v, 5)))
    return out


def run_quality(d) -> dict:
    r = QL.compare(d)
    print(QL.report(r))
    csv_path, md_path = QL.save(r, str(RESULTS))
    print(f"saved {csv_path} and {md_path}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser(prog="analytics")
    ap.add_argument("cmd", choices=("queries", "algorithms", "quality", "all"))
    a = ap.parse_args()
    d = get_driver()
    try:
        Q.ensure_schema(d)
        ok = True
        if a.cmd in ("queries", "all"):
            print("== Cypher library (analytics/queries.cypher), both kg_versions")
            ok &= check_queries(d)
        if a.cmd in ("algorithms", "all"):
            print("== GDS algorithms (written back as <algorithm>_<kg_version>)")
            algo = run_algorithms(d)
            ok &= all(algo[v]["written"]["pagerank"] == algo[v]["written"]["n"] for v in KG_VERSIONS)
            print("ALGORITHMS_OK" if ok else "ALGORITHMS_FAILED")
        if a.cmd in ("quality", "all"):
            print("== RQ2 graph quality, raw vs resolved")
            run_quality(d)
            print("QUALITY_OK")
        return 0 if ok else 1
    finally:
        d.close()


if __name__ == "__main__":
    sys.exit(main())
