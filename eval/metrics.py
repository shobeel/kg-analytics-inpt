"""Phase 6 metrics, computed from stored results only (no LLM calls):

  entity recall (= completeness)  expected entities found in the answer text (relaxed name match)
  entity precision                expected entities found / graph entities of the expected types that the
                                  answer names (rule-based mention detection over the entity-name index;
                                  entities named in the question are ignored); F1 from both
  correctness                     judge 1-5 (eval.judge), or 1/5 by rule for 'Not found'
  faithfulness, citation precision  from the judge (RQ5)
  refusals                        'Not found' on unanswerable (correct) and on answerable (false refusal)
  latency, tokens, LLM calls, cost  per answer; cost from public list prices (eval/prices.json, RQ4)

Writes eval/results/benchmark_results.csv (one row per answer) and benchmark_summary.{json,md}.
"""

import csv
import json
import statistics
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from .extraction_scorer import relaxed_name

RESULTS = Path(__file__).parent / "results"
PRICES = Path(__file__).parent / "prices.json"
INDEX = RESULTS / "entity_name_index.json"
NOT_FOUND = "Not found in the knowledge graph."
SYSTEM_NAMES = {"direct": "Direct Cypher", "A": "A vector", "B": "B GraphRAG", "C": "C Text2Cypher"}
MAX_NGRAM = 8


def tokens(text: str) -> list[str]:
    return relaxed_name(unicodedata.normalize("NFKC", text or "")).split()


def contains_entity(answer: str, entity: str) -> bool:
    e = tokens(entity)
    if not e:
        return False
    a = tokens(answer)
    return any(a[i:i + len(e)] == e for i in range(len(a) - len(e) + 1))


def mentions(text: str, index: dict) -> list[str]:
    """Graph entity names in a text: greedy longest match over token n-grams. index: {relaxed name: labels}."""
    t, out, i = tokens(text), [], 0
    while i < len(t):
        for n in range(min(MAX_NGRAM, len(t) - i), 0, -1):
            k = " ".join(t[i:i + n])
            if k in index:
                out.append(k)
                i += n
                break
        else:
            i += 1
    return out


def entity_scores(answer: str, expected: list[str], question: str = "", index: dict | None = None) -> dict:
    if not expected:
        return {"entity_recall": None, "entity_precision": None, "entity_f1": None, "entities_found": 0}
    if answer == NOT_FOUND:
        return {"entity_recall": 0.0, "entity_precision": None, "entity_f1": 0.0, "entities_found": 0}
    found = [e for e in expected if contains_entity(answer, e)]
    recall = len(found) / len(expected)
    precision = None
    if index is not None:
        labels = set().union(*(index.get(relaxed_name(e), []) for e in expected)) if expected else set()
        in_q = set(mentions(question, index))
        exp_keys = {relaxed_name(e) for e in expected}
        named = {m for m in mentions(answer, index) if m not in in_q and (set(index[m]) & labels or m in exp_keys)}
        denom = max(len(named | {relaxed_name(e) for e in found}), len(found))
        precision = len(found) / denom if denom else (1.0 if not found else 0.0)
    f1 = (2 * precision * recall / (precision + recall) if precision is not None and precision + recall
          else (0.0 if precision is not None else None))
    return {"entity_recall": recall, "entity_precision": precision, "entity_f1": f1, "entities_found": len(found)}


def cost_usd(tokens_in: int, tokens_out: int, price: dict) -> float:
    return tokens_in / 1e6 * price["input_per_m"] + tokens_out / 1e6 * price["output_per_m"]


def pct(values: list[float], p: float) -> float | None:
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def mean(values) -> float | None:
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else None


# ------------------------------------------------------------------ per-answer rows
def result_rows(items: list[dict], runs: list[dict], judged: dict, index: dict | None, prices: dict) -> list[dict]:
    from .judge import deterministic
    by_id = {q["id"]: q for q in items}
    price = prices["answer_model"]
    rows = []
    for r in runs:
        q = by_id.get(r["question_id"])
        if q is None:
            continue
        refused = r["answer"] == NOT_FOUND or not r.get("found")
        ent = entity_scores(r["answer"], q.get("expected_entities") or [], q["question"], index)
        j = judged.get(r["key"])
        if r["system"] == "direct":
            corr, faith, cprec, judged_by = None, None, None, "n/a (gold query)"
        elif j:
            corr, faith, cprec, judged_by = j["correctness"], j.get("faithfulness"), j.get("citation_precision"), j["judged_by"]
        else:
            d = deterministic(q, r)
            corr, faith, cprec, judged_by = (d["correctness"], None, None, "rule") if d else (None, None, None, "pending")
        srcs = r.get("sources") or []
        rows.append({
            "key": r["key"], "question_id": q["id"], "category": q["category"], "system": r["system"],
            "system_name": SYSTEM_NAMES[r["system"]], "kg_version": r["kg_version"], "question": q["question"],
            "answer": " ".join(r["answer"].split())[:600], "refused": refused,
            "answerable": q["category"] != "unanswerable",
            **ent, "correctness": corr, "faithfulness": faith, "citation_precision": cprec, "judged_by": judged_by,
            "n_sources": len(srcs), "n_cited": sum(1 for s in srcs if s.get("cited", True)),
            "invalid_citations": len(r.get("invalid_citations") or []),
            "cypher_retries": len(r.get("cypher_errors") or []),
            "llm_calls": r["llm_calls"], "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
            "latency_s": r["latency_s"], "retrieval_s": r.get("retrieval_s"),
            "cost_usd": cost_usd(r["tokens_in"], r["tokens_out"], price),
        })
    return rows


# ------------------------------------------------------------------ summaries
def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    un = [r for r in rows if not r["answerable"]]
    return {
        "answers": len(rows), "questions": len({r["question_id"] for r in rows}),
        "correctness": mean(r["correctness"] for r in rows),
        "correct_ge4": mean((r["correctness"] >= 4) for r in rows if r["correctness"] is not None),
        "entity_recall": mean(r["entity_recall"] for r in ans), "entity_precision": mean(r["entity_precision"] for r in ans),
        "entity_f1": mean(r["entity_f1"] for r in ans),
        "faithfulness": mean(r["faithfulness"] for r in rows), "citation_precision": mean(r["citation_precision"] for r in rows),
        "answers_with_sources": mean((r["n_sources"] > 0) for r in ans if not r["refused"]),
        "refusal_rate_unanswerable": mean(r["refused"] for r in un),
        "false_refusal_rate": mean(r["refused"] for r in ans),
        "latency_median_s": pct([r["latency_s"] for r in rows], 0.5), "latency_p95_s": pct([r["latency_s"] for r in rows], 0.95),
        "llm_calls_per_q": mean(r["llm_calls"] for r in rows), "tokens_in_per_q": mean(r["tokens_in"] for r in rows),
        "tokens_out_per_q": mean(r["tokens_out"] for r in rows), "cost_usd_per_q": mean(r["cost_usd"] for r in rows),
        "cost_usd_total": sum(r["cost_usd"] for r in rows),
        "invalid_citations_removed": sum(r["invalid_citations"] for r in rows),
        "judge_pending": sum(r["judged_by"] == "pending" for r in rows),
    }


def group(rows: list[dict], *keys) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return {k: summarize(v) for k, v in sorted(g.items())}


def paired_raw_resolved(rows: list[dict]) -> dict:
    """Per system: questions with an answer on both versions, mean of each metric per version."""
    out = {}
    for sy in SYSTEM_NAMES:
        by = defaultdict(dict)
        for r in rows:
            if r["system"] == sy:
                by[r["question_id"]][r["kg_version"]] = r
        pairs = [v for v in by.values() if "raw" in v and "resolved" in v]
        if not pairs:
            continue
        out[sy] = {"pairs": len(pairs)}
        for m in ("correctness", "entity_recall", "entity_f1", "faithfulness", "citation_precision", "latency_s"):
            for ver in ("raw", "resolved"):
                out[sy][f"{m}_{ver}"] = mean(p[ver][m] for p in pairs)
        out[sy]["better_resolved"] = sum(1 for p in pairs if (p["resolved"]["correctness"] or 0) > (p["raw"]["correctness"] or 0))
        out[sy]["better_raw"] = sum(1 for p in pairs if (p["raw"]["correctness"] or 0) > (p["resolved"]["correctness"] or 0))
    return out


def fmt(x, kind="f2") -> str:
    if x is None:
        return "–"
    if kind == "pct":
        return f"{100 * x:.0f}%"
    if kind == "usd":
        return f"${x:.5f}"
    if kind == "int":
        return f"{x:.0f}"
    return f"{x:.2f}"


COLS = [("correctness", "Correctness (1-5)", "f2"), ("correct_ge4", "Correct (≥4)", "pct"),
        ("entity_recall", "Entity recall", "pct"), ("entity_f1", "Entity F1", "pct"),
        ("faithfulness", "Faithfulness", "pct"), ("citation_precision", "Citation precision", "pct"),
        ("false_refusal_rate", "False refusals", "pct"), ("refusal_rate_unanswerable", "Refused unanswerable", "pct"),
        ("latency_median_s", "Latency p50 (s)", "f2"), ("latency_p95_s", "Latency p95 (s)", "f2"),
        ("llm_calls_per_q", "LLM calls/q", "f2"), ("tokens_in_per_q", "Tokens in/q", "int"),
        ("tokens_out_per_q", "Tokens out/q", "int"), ("cost_usd_per_q", "Cost/q (paid)", "usd")]


def md_table(groups: dict, label_cols: list[str], cols=COLS) -> str:
    head = "| " + " | ".join(label_cols + [c[1] for c in cols]) + " |"
    sep = "|" + "---|" * (len(label_cols) + len(cols))
    lines = [head, sep]
    for k, s in groups.items():
        lab = [SYSTEM_NAMES.get(x, x) for x in k]
        lines.append("| " + " | ".join(lab + [fmt(s[c[0]], c[2]) for c in cols]) + " |")
    return "\n".join(lines)


def build_index(driver) -> dict:
    """{relaxed entity name: [labels]} from the resolved graph (names of >= 3 chars)."""
    from analytics.db import run
    idx = defaultdict(set)
    for r in run(driver, """MATCH (e:Entity {kg_version: 'resolved'})
                            RETURN e.name AS name, e.aliases AS aliases,
                                   [l IN labels(e) WHERE l <> 'Entity'][0] AS label"""):
        for n in [r["name"]] + list(r["aliases"] or []):
            k = relaxed_name(n or "")
            if len(k) >= 3 and not k.isdigit():
                idx[k].add(r["label"])
    return {k: sorted(v) for k, v in idx.items()}


def load_index(refresh: bool = False) -> dict | None:
    if INDEX.exists() and not refresh:
        return json.loads(INDEX.read_text())
    try:
        from analytics.db import get_driver
        idx = build_index(get_driver())
    except Exception as e:
        print(f"(entity index unavailable: {type(e).__name__}; entity precision not computed)")
        return None
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(idx, ensure_ascii=False))
    return idx


def hardest(rows: list[dict], n: int = 5) -> list[dict]:
    """Questions with the lowest mean correctness over the LLM systems (A, B, C) on the resolved graph."""
    by = defaultdict(list)
    for r in rows:
        if r["system"] != "direct" and r["kg_version"] == "resolved" and r["correctness"] is not None:
            by[r["question_id"]].append(r)
    direct = {(r["question_id"], r["kg_version"]): r for r in rows if r["system"] == "direct"}
    out = []
    for qid, rs in by.items():
        d_res, d_raw = direct.get((qid, "resolved")), direct.get((qid, "raw"))
        out.append({"question_id": qid, "category": rs[0]["category"], "question": rs[0]["question"],
                    "mean_correctness": mean(r["correctness"] for r in rs),
                    "per_system": {r["system"]: r["correctness"] for r in sorted(rs, key=lambda r: r["system"])},
                    "refusals": sum(r["refused"] for r in rs), "systems": len(rs),
                    "mean_entity_recall": mean(r["entity_recall"] for r in rs),
                    "direct_entity_recall_resolved": d_res["entity_recall"] if d_res else None,
                    "direct_entity_recall_raw": d_raw["entity_recall"] if d_raw else None,
                    "answers": {r["system"]: r["answer"][:300] for r in rs}})
    return sorted(out, key=lambda x: (x["mean_correctness"], x["question_id"]))[:n]


def examples(rows: list[dict], k: int = 3) -> dict:
    """Per system, the k best and k worst answers on the resolved graph (LLM systems: correctness, then
    entity F1; Direct Cypher: entity F1, since the gold query is not judged)."""
    out = {}
    for sy in SYSTEM_NAMES:
        rs = [r for r in rows if r["system"] == sy and r["kg_version"] == "resolved" and r["answerable"]]
        if not rs:
            continue
        key = (lambda r: (r["entity_f1"] or 0, r["entity_recall"] or 0)) if sy == "direct" else \
              (lambda r: (r["correctness"] or 0, r["entity_f1"] or 0))
        srt = sorted(rs, key=lambda r: (key(r), r["question_id"]))
        pick = lambda r: {f: r[f] for f in ("question_id", "category", "question", "answer", "correctness",
                                            "entity_recall", "entity_f1", "faithfulness", "refused")}
        out[sy] = {"good": [pick(r) for r in reversed(srt[-k:])], "bad": [pick(r) for r in srt[:k]]}
    return out


def llm_usage(path: Path) -> dict:
    """All LLM requests of the project per (phase, model) from the call log (incl. retries and errors)."""
    out = {}
    if not path.exists():           # logs/ is not in git: a fresh clone keeps the committed table
        prev = RESULTS / "benchmark_summary.json"
        return (json.loads(prev.read_text()).get("llm_usage") or {}) if prev.exists() else out
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = r.get("purpose", "")
        phase = ("benchmark answers" if p.startswith("graphrag") else "judge" if p.startswith("judge")
                 else "extraction / ER" if p.startswith(("extract", "adjudicate")) else "gold pre-fill" if "prefill" in p
                 else "other")
        d = out.setdefault(f"{phase}|{r.get('model')}", {"requests": 0, "ok": 0, "tokens_in": 0, "tokens_out": 0, "lat": []})
        d["requests"] += 1
        if r.get("status") == "ok":
            d["ok"] += 1
            d["tokens_in"] += r.get("input_tokens", 0)
            d["tokens_out"] += r.get("output_tokens", 0) + r.get("thinking_tokens", 0)
            d["lat"].append(r.get("latency_s", 0))
    for d in out.values():
        d["latency_median_s"] = statistics.median(d.pop("lat")) if d.get("lat") else None
    return dict(sorted(out.items()))


def judge_usage(judged: dict, prices: dict) -> dict:
    us = [j.get("judge_usage") or {} for j in judged.values() if j.get("judged_by") not in (None, "rule")]
    tin = sum(u.get("input_tokens", 0) for u in us)
    tout = sum(u.get("output_tokens", 0) + u.get("thinking_tokens", 0) for u in us)
    return {"calls": len(us), "tokens_in": tin, "tokens_out": tout, "cost_usd": cost_usd(tin, tout, prices["judge_model"])}


def direct_changes(rows: list[dict]) -> int:
    """Questions whose gold query returns a different result on the raw graph than on the resolved one."""
    d = {(r["question_id"], r["kg_version"]): r["answer"] for r in rows if r["system"] == "direct"}
    return sum(1 for (q, v), a in d.items() if v == "resolved" and (q, "raw") in d and d[(q, "raw")] != a)


def compute(items: list[dict], runs: dict, judged: dict, index: dict | None, errors: dict | None = None) -> dict:
    prices = json.loads(PRICES.read_text())
    rows = result_rows(items, list(runs.values()), judged, index, prices)
    ids = {q["id"] for q in items}
    err = [e for e in (errors or {}).values() if e["question_id"] in ids]
    jerr_path = RESULTS / "judge_errors.jsonl"
    jerr = {json.loads(l)["run_key"] for l in jerr_path.read_text().splitlines() if l.strip()} if jerr_path.exists() else set()
    llm = [r for r in rows if r["system"] != "direct"]
    return {"rows": rows,
            "by_system_version": group(rows, "system", "kg_version"),
            "by_category_system": group([r for r in rows if r["kg_version"] == "resolved"], "category", "system"),
            "raw_vs_resolved": paired_raw_resolved(rows),
            "hardest": hardest(rows), "examples": examples(rows),
            "headline_g01": {r["kg_version"]: r["answer"] for r in rows if r["question_id"] == "G01" and r["system"] == "direct"},
            "errors": {"answer_runs": len(err), "by_system": {s: sum(e["system"] == s for e in err) for s in SYSTEM_NAMES},
                       "judge_unresolved": len(jerr - set(judged))},
            "totals": {"questions": len(items), "by_category": {c: sum(q["category"] == c for q in items)
                                                                for c in sorted({q["category"] for q in items})},
                       "direct_answer_changes": direct_changes(rows), "answers": len(rows), "llm_answers": len(llm), "llm_calls": sum(r["llm_calls"] for r in llm),
                       "tokens_in": sum(r["tokens_in"] for r in llm), "tokens_out": sum(r["tokens_out"] for r in llm),
                       "answer_cost_usd": sum(r["cost_usd"] for r in llm), "judge": judge_usage(judged, prices)},
            "llm_usage": llm_usage(Path(__file__).resolve().parent.parent / "logs" / "llm_calls.jsonl"),
            "prices": prices}


def save(res: dict, judge_agreement: dict | None = None) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = res["rows"]
    if rows:
        with (RESULTS / "benchmark_results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    ser = {k: {"|".join(kk): vv for kk, vv in v.items()} for k, v in res.items() if k.startswith("by_")}
    for k in ("raw_vs_resolved", "hardest", "examples", "errors", "totals", "llm_usage", "headline_g01"):
        ser[k] = res.get(k)
    ser["judge_agreement"] = judge_agreement
    from .judge_validation import inter_report
    ser["inter_model_agreement"] = inter_report()
    (RESULTS / "benchmark_summary.json").write_text(json.dumps(ser, indent=1, default=str))
    md = ["# Benchmark summary", "", "Generated by `eval.metrics` from `benchmark_runs.jsonl` and `judge_scores.jsonl`.", "",
          "## By system and graph version", "", md_table(res["by_system_version"], ["System", "KG"]), "",
          "## By category and system (resolved graph)", "",
          md_table(res["by_category_system"], ["Category", "System"]), ""]
    (RESULTS / "benchmark_summary.md").write_text("\n".join(md))


def main() -> int:
    from . import questions as Q
    from .judge import judged_keys
    from .judge_validation import agreement_report
    from .run_benchmark import done_runs, error_runs
    items = Q.verified(Q.load())
    res = compute(items, done_runs(), judged_keys(), load_index("--refresh-index" in sys.argv), error_runs())
    agree = agreement_report()
    save(res, agree)
    print(md_table(res["by_system_version"], ["System", "KG"]))
    print(f"\n{len(res['rows'])} answers scored; judge pending: {sum(r['judged_by'] == 'pending' for r in res['rows'])}")
    print("judge agreement:", json.dumps(agree, default=str) if agree else "pending (eval/gold/judge_validation.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
