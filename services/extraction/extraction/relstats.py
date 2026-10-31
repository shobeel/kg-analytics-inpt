"""Relation statistics over the cached extractions (flags recomputed under the current rules).
Schema validation is post-hoc, so changing the rules only re-labels existing relations: no LLM calls."""

from collections import Counter

from .schema import PROMPT_DOMAIN_RANGE


def classify(r: dict) -> str:
    """One class per relation, in priority order:
    self_loop | unknown_endpoint | invalid | repaired (valid after the direction swap) | valid"""
    if r["self_loop"]:
        return "self_loop"
    if not r["endpoints_known"]:
        return "unknown_endpoint"
    if not r["schema_valid"]:
        return "invalid"
    return "repaired" if r["direction_repaired"] else "valid"


def valid_under_prompt_table(r: dict) -> bool:
    """The original extract-v2 rules, before tweaks and direction repair (baseline for the report)."""
    if r["direction_repaired"]:
        return False
    src_ok, tgt_ok = PROMPT_DOMAIN_RANGE[r["relation"]]
    return r["source_type"] in src_ok and r["target_type"] in tgt_ok


def zero_relation_entities(recs: list[dict]) -> dict:
    """Entity mentions (per chunk) that are an endpoint of no relation in that chunk."""
    total, isolated, by_type, iso_by_type = 0, 0, Counter(), Counter()
    for rec in recs:
        used = {n.strip().lower() for r in rec["relations"] for n in (r["source"], r["target"])}
        for e in rec["entities"]:
            total += 1
            by_type[e["type"]] += 1
            if e["name"].strip().lower() not in used:
                isolated += 1
                iso_by_type[e["type"]] += 1
    return {"entity_mentions": total, "zero_relation": isolated,
            "rate": round(isolated / total, 4) if total else 0.0,
            "rate_by_type": {t: round(iso_by_type[t] / n, 3) for t, n in by_type.most_common()}}


def relation_stats(recs: list[dict], top: int = 10) -> dict:
    rels = [r for rec in recs for r in rec["relations"]]
    n = len(rels)
    rate = lambda c: {k: {"count": v, "rate": round(v / n, 4) if n else 0.0} for k, v in sorted(c.items())}
    remaining = Counter((r["source_type"], r["relation"], r["target_type"]) for r in rels
                        if r["endpoints_known"] and not r["schema_valid"])
    return {
        "chunks": len(recs), "relations": n,
        "per_relation_type": dict(Counter(r["relation"] for r in rels).most_common()),
        "valid_under_original_prompt_table": sum(map(valid_under_prompt_table, rels)),
        "classes": rate(Counter(map(classify, rels))),
        "top_remaining_violations": [{"source_type": s, "relation": rel, "target_type": t, "count": c}
                                     for (s, rel, t), c in remaining.most_common(top)],
        "zero_relation_entities": zero_relation_entities(recs),
    }
