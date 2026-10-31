"""RQ2: graph quality of the raw vs the resolved KG. Needs algorithms.run_all() for both versions
(it reads the written degree/component/community/pagerank properties). Measures only: nothing
here changes the graph.

Duplicates use the same name normalization as Phase 2 entity resolution:
  duplicate rate        entities sharing (label, normalized name) with another entity, / entities
  cross-type duplicates normalized names that occur under more than one label (Gemini as AIModel
                        and Product): ER blocks by type, so it never merges these by design
"""

import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from .algorithms import prop
from .db import check_version, run

LEGAL_SUFFIXES = {"inc", "incorporated", "ltd", "limited", "llc", "corp", "corporation", "co",
                  "company", "gmbh", "plc", "sa", "ag", "lp", "llp", "pbc"}


def norm_key(name: str) -> str:
    """Same rules as services/extraction/extraction/normalize.py (Phase 2 ER)."""
    s = unicodedata.normalize("NFKC", name).casefold()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    tokens = s.split()
    if tokens and tokens[0] == "the":
        tokens = tokens[1:]
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens)


def duplicate_stats(entities: list[dict]) -> dict:
    """entities: [{id, label, name, mentions}]."""
    same_label, by_key = defaultdict(list), defaultdict(list)
    for e in entities:
        k = norm_key(e["name"])
        same_label[(e["label"], k)].append(e)
        by_key[k].append(e)
    dup_entities = sum(len(g) - 1 for g in same_label.values() if len(g) > 1)
    cross = []
    for k, g in by_key.items():
        labels = sorted({e["label"] for e in g})
        if len(labels) > 1:
            cross.append({"normalized_name": k, "labels": labels, "entities": len(g),
                          "names": sorted({e["name"] for e in g}),
                          "mentions": sum(e["mentions"] or 0 for e in g)})
    cross.sort(key=lambda c: (-c["mentions"], c["normalized_name"]))
    n = len(entities)
    return {"duplicate_entities": dup_entities, "duplicate_rate": dup_entities / n if n else 0.0,
            "duplicate_groups": sum(1 for g in same_label.values() if len(g) > 1),
            "cross_type_groups": len(cross), "cross_type_entities": sum(c["entities"] for c in cross),
            "cross_type": cross}


def graph_stats(n: int, edges: int, degrees: list[int], components: dict, communities: list[int]) -> dict:
    return {"entities": n, "relation_edges": edges,
            "density": 2 * edges / (n * (n - 1)) if n > 1 else 0.0,
            "avg_degree": 2 * edges / n if n else 0.0,
            "orphan_pct": 100 * sum(1 for d in degrees if d == 0) / n if n else 0.0,
            "connected_components": components["count"],
            "largest_component_pct": 100 * components["largest"] / n if n else 0.0,
            "louvain_communities": len(communities),
            "louvain_communities_size_ge_2": sum(1 for s in communities if s >= 2)}


def _version_data(driver, v: str) -> dict:
    check_version(v)
    d, c, comm, pr = (prop(x, v) for x in ("degree", "component", "community", "pagerank"))
    ents = run(driver, f"""
        MATCH (e:Entity {{kg_version: $v}})
        RETURN e.id AS id, [l IN labels(e) WHERE l <> 'Entity'][0] AS label, e.name AS name,
               e.mention_count AS mentions, e.resolved_id AS resolved_id,
               e.{d} AS degree, e.{c} AS component, e.{comm} AS community, e.{pr} AS pagerank""", v=v)
    if any(e["degree"] is None for e in ents):
        raise RuntimeError(f"run algorithms.run_all(driver, '{v}') first")
    edges = run(driver, """MATCH (:Entity {kg_version: $v})-[r]->(:Entity)
                           WHERE r.kg_version = $v RETURN count(r) AS n""", v=v)[0]["n"]
    comp_sizes, comm_sizes = defaultdict(int), defaultdict(int)
    for e in ents:
        comp_sizes[e["component"]] += 1
        comm_sizes[e["community"]] += 1
    return {"entities": ents, "edges": edges,
            "components": {"count": len(comp_sizes), "largest": max(comp_sizes.values())},
            "communities": sorted(comm_sizes.values(), reverse=True)}


def pagerank_change(raw: list[dict], resolved: list[dict], k: int = 10) -> list[dict]:
    """Top-k resolved entities by PageRank, with where their best raw member ranked."""
    raw_rank = {e["id"]: i + 1 for i, e in enumerate(sorted(raw, key=lambda e: -(e["pagerank"] or 0)))}
    members = defaultdict(list)
    for e in raw:
        members[e["resolved_id"]].append(e)
    top_raw = {e["id"] for e in sorted(raw, key=lambda e: -(e["pagerank"] or 0))[:k]}
    rows = []
    for i, e in enumerate(sorted(resolved, key=lambda e: -(e["pagerank"] or 0))[:k], 1):
        m = sorted(members[e["id"]], key=lambda x: raw_rank[x["id"]])
        best = m[0] if m else None
        rows.append({"rank_resolved": i, "name": e["name"], "label": e["label"],
                     "pagerank_resolved": round(e["pagerank"], 3),
                     "raw_members": len(m), "best_raw_member": best["name"] if best else None,
                     "rank_raw": raw_rank[best["id"]] if best else None,
                     "pagerank_raw": round(best["pagerank"], 3) if best else None,
                     "in_raw_top": bool(best and best["id"] in top_raw)})
    return rows


def compare(driver) -> dict:
    data = {v: _version_data(driver, v) for v in ("raw", "resolved")}
    out = {}
    for v, d in data.items():
        degs = [e["degree"] for e in d["entities"]]
        out[v] = graph_stats(len(d["entities"]), d["edges"], degs, d["components"], d["communities"])
        out[v] |= {k: val for k, val in duplicate_stats(
            [{"id": e["id"], "label": e["label"], "name": e["name"], "mentions": e["mentions"]}
             for e in d["entities"]]).items()}
    out["pagerank_top10"] = pagerank_change(data["raw"]["entities"], data["resolved"]["entities"])
    out["pagerank_top10_overlap"] = sum(r["in_raw_top"] for r in out["pagerank_top10"])
    return out


METRICS = [("entities", "Entities", "{:.0f}"), ("relation_edges", "Relation edges", "{:.0f}"),
           ("duplicate_entities", "Same-type duplicate entities", "{:.0f}"),
           ("duplicate_rate", "Duplicate rate", "{:.2%}"),
           ("cross_type_groups", "Cross-type duplicate names", "{:.0f}"),
           ("cross_type_entities", "Entities in cross-type duplicates", "{:.0f}"),
           ("connected_components", "Connected components", "{:.0f}"),
           ("largest_component_pct", "Largest component %", "{:.1f}"),
           ("orphan_pct", "Orphan entities %", "{:.1f}"), ("density", "Density", "{:.6f}"),
           ("avg_degree", "Average degree", "{:.3f}"), ("louvain_communities", "Louvain communities", "{:.0f}"),
           ("louvain_communities_size_ge_2", "Louvain communities (size >= 2)", "{:.0f}")]


def table_rows(r: dict) -> list[dict]:
    rows = []
    for key, name, fmt in METRICS:
        a, b = r["raw"][key], r["resolved"][key]
        ch = (b - a) / a if a else None
        rows.append({"metric": name, "key": key, "raw": a, "resolved": b, "change": ch,
                     "raw_fmt": fmt.format(a), "resolved_fmt": fmt.format(b),
                     "change_fmt": "" if ch is None else f"{ch:+.1%}"})
    return rows


def save(r: dict, out_dir: str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = table_rows(r)
    csv_path, md_path = out / "rq2_graph_quality.csv", out / "rq2_graph_quality.md"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "raw", "resolved", "relative_change"])
        for x in rows:
            w.writerow([x["key"], x["raw"], x["resolved"], "" if x["change"] is None else round(x["change"], 4)])
    md = ["# RQ2: graph quality, raw vs resolved KG", "",
          "Generated by `analytics.quality` (`make test-phase4` / notebook 04). Entity graph = entities + the "
          "12 relation types (no MENTIONS). Duplicates use the Phase 2 ER name normalization.", "",
          "| Metric | Raw | Resolved | Change |", "|---|---|---|---|"]
    md += [f"| {x['metric']} | {x['raw_fmt']} | {x['resolved_fmt']} | {x['change_fmt']} |" for x in rows]
    md += ["", f"## Top-10 PageRank (resolved) and the rank of its best raw member "
               f"({r['pagerank_top10_overlap']}/10 also in the raw top 10)", "",
           "| # | Resolved entity | PageRank | Raw members | Best raw member | Raw rank | Raw PageRank |",
           "|---|---|---|---|---|---|---|"]
    md += [f"| {p['rank_resolved']} | {p['name']} ({p['label']}) | {p['pagerank_resolved']} | {p['raw_members']} | "
           f"{p['best_raw_member']} | {p['rank_raw']} | {p['pagerank_raw']} |" for p in r["pagerank_top10"]]
    for v in ("raw", "resolved"):
        md += ["", f"## Top-20 cross-type duplicates ({v}): same normalized name, different labels", "",
               "| Normalized name | Labels | Names | Entities | Mentions |", "|---|---|---|---|---|"]
        md += [f"| {c['normalized_name']} | {', '.join(c['labels'])} | {'; '.join(c['names'][:4])} | "
               f"{c['entities']} | {c['mentions']} |" for c in r[v]["cross_type"][:20]]
    md_path.write_text("\n".join(md) + "\n")
    return csv_path, md_path


def report(r: dict) -> str:
    lines = [f"{'metric':<36}{'raw':>12}{'resolved':>12}{'change':>9}"]
    lines += [f"{x['metric']:<36}{x['raw_fmt']:>12}{x['resolved_fmt']:>12}{x['change_fmt']:>9}" for x in table_rows(r)]
    lines += ["", f"Top-10 PageRank, resolved vs raw ({r['pagerank_top10_overlap']}/10 overlap):"]
    lines += [f"  {p['rank_resolved']:>2}. {p['name'][:28]:<28} pr={p['pagerank_resolved']:<7} raw rank "
              f"{p['rank_raw']} ({p['raw_members']} raw member{'s' if p['raw_members'] != 1 else ''})"
              for p in r["pagerank_top10"]]
    lines += ["", "Top-20 cross-type duplicates (resolved):"]
    lines += [f"  {c['normalized_name'][:30]:<30} {'/'.join(c['labels']):<34} mentions={c['mentions']}"
              for c in r["resolved"]["cross_type"][:20]]
    return "\n".join(lines)
