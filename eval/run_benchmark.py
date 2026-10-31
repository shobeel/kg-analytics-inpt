"""Phase 6 benchmark runner: every verified question x {Direct Cypher, A vector, B GraphRAG, C Text2Cypher}
x {resolved, raw}. Resumable: one JSONL line per finished (question, system, kg_version); GraphRAG LLM
calls are cached by the Phase 5 engine, judge calls by eval.judge. Stops cleanly on the daily quota
or when the Gemini budget for today would be exceeded, and resumes on the next run.

  plan          planned Gemini (answers) and Groq (judge) calls for what is still to do; no LLM call
  run           answers (Direct Cypher needs no LLM). Prints the plan and stops unless --confirm
  judge         Groq judge on every answer not judged yet. Prints the plan and stops unless --confirm
  status        what is done

Usage: python -m eval.run_benchmark plan|run|judge|status [--confirm] [--all-drafts] [--max-gemini N]
  --all-drafts  include unverified questions (dry runs; Direct Cypher only by default, see --systems)
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from . import questions as Q

RESULTS = Path(__file__).parent / "results"
RUNS = RESULTS / "benchmark_runs.jsonl"
SYSTEMS = {"direct": "Direct Cypher", "A": "A vector", "B": "B GraphRAG", "C": "C Text2Cypher"}
LLM_SYSTEMS = ("A", "B", "C")
VERSIONS = ("resolved", "raw")
NOT_FOUND = "Not found in the knowledge graph."
# Upper bound of Gemini calls per answer: A/B one answer call; C cypher (+1 retry) + verbalize.
MAX_CALLS = {"A": 1, "B": 1, "C": 3}
STEPS = {"A": ("answer",), "B": ("answer",), "C": ("cypher", "verbalize")}


def run_key(q: dict, system: str, kg_version: str) -> str:
    h = hashlib.sha256(json.dumps([q["question"], q["gold_query"]]).encode()).hexdigest()[:10]
    return f"{q['id']}|{system}|{kg_version}|{h}"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def done_runs() -> dict:
    """Latest finished run per key (errors are retried, so only status ok/stored counts)."""
    out = {}
    for r in read_jsonl(RUNS):
        if r.get("status") == "ok":
            out[r["key"]] = r
    return out


# ------------------------------------------------------------------ quota accounting
def quota_day_start(now: datetime | None = None) -> datetime:
    """Gemini free-tier daily quotas reset at midnight Pacific time."""
    now = now or datetime.now(timezone.utc)
    pt = now.astimezone(ZoneInfo("America/Los_Angeles"))
    return pt.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def requests_today(log_path: str, model: str, now: datetime | None = None) -> int:
    """Requests (ok + errors: both count against the quota) sent to `model` since the daily reset."""
    start = quota_day_start(now).isoformat()
    n = 0
    p = Path(log_path)
    if not p.exists():
        return 0
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("model") == model and r.get("ts", "") >= start[:23] and r.get("status") != "quota_exhausted":
            n += 1
    return n


def gemini_daily_limit() -> int:
    return int(os.environ.get("GEMINI_DAILY_LIMIT", "500"))


# ------------------------------------------------------------------ the systems
def render_rows(rows: list[dict], limit: int = 20) -> str:
    """Direct Cypher 'answer': the gold query rows as text."""
    if not rows:
        return NOT_FOUND
    lines = ["; ".join(f"{k}: {v}" for k, v in r.items() if v not in (None, [], "")) for r in rows[:limit]]
    more = f" (+{len(rows) - limit} more rows)" if len(rows) > limit else ""
    return "\n".join(lines) + more


def run_direct(driver, q: dict, kg_version: str) -> dict:
    t0 = time.perf_counter()
    rows = Q.run_query(driver, q["gold_query"], kg_version)
    lat = time.perf_counter() - t0
    return {"answer": render_rows(rows), "found": bool(rows), "rows": len(rows), "sources": [],
            "cypher": q["gold_query"], "llm_calls": 0, "cache_hits": 0, "tokens_in": 0, "tokens_out": 0,
            "latency_s": round(lat, 3), "retrieval_s": round(lat, 3), "model": None,
            "result_rows": rows[:40]}


KEEP = ("answer", "found", "sources", "cypher", "cypher_errors", "rows", "invalid_citations", "entities_used",
        "retrieval", "llm_calls", "cache_hits", "tokens_in", "tokens_out", "latency_s", "retrieval_s", "model",
        "prompt_version")


def run_llm_system(engine, q: dict, system: str, kg_version: str) -> dict:
    r = engine.answer(q["question"], system, kg_version)
    out = {k: r.get(k) for k in KEEP}
    out["found"] = r.get("found", r["answer"] != NOT_FOUND)
    out["context"] = (r.get("context") or "")[:6000]
    out["sources"] = [{k: s.get(k) for k in ("doc_id", "short_id", "title", "url", "evidence_text", "cited", "kind")}
                      for s in r.get("sources", [])]
    return out


def uncached_steps(engine, q: dict, system: str, kg_version: str) -> list[str]:
    from graphrag import PROMPT_VERSION
    from graphrag.cache import cache_key
    return [st for st in STEPS[system]
            if engine.cache.get(cache_key(system, q["question"], kg_version, PROMPT_VERSION, st)) is None]


# ------------------------------------------------------------------ plan
def error_runs() -> dict:
    """Runs that failed after the client's retries and have no later ok run: final, reported as errors."""
    ok = done_runs()
    return {r["key"]: r for r in read_jsonl(RUNS) if r.get("status") == "error" and r["key"] not in ok}


def pending(items: list[dict], systems, versions=VERSIONS) -> list[tuple]:
    """Version-major order: every resolved run comes before any raw run, so a quota cut keeps the
    most valuable results. Errored runs are final (the client already retried them)."""
    done = done_runs() | error_runs()
    return [(q, s, v) for v in versions for q in items for s in systems if run_key(q, s, v) not in done]


def plan(items: list[dict], systems, cache_probe=None) -> dict:
    """Planned calls. cache_probe(q, system, version) -> uncached GraphRAG steps (None: assume none cached)."""
    from .judge import judged_keys
    todo = pending(items, systems)
    exp = mx = 0
    for q, s, v in todo:
        if s not in LLM_SYSTEMS:
            continue
        steps = cache_probe(q, s, v) if cache_probe else list(STEPS[s])
        if s == "C":
            if "cypher" in steps:
                exp += 1 if q["category"] == "unanswerable" else 2
                mx += 3
            elif "verbalize" in steps:
                exp, mx = exp + 1, mx + 1
        elif steps:
            exp, mx = exp + 1, mx + 1
    done = done_runs()
    judged = judged_keys()
    llm_done = [r for r in done.values() if r["system"] in LLM_SYSTEMS and r["question_id"] in {q["id"] for q in items}]
    judge_now = sum(1 for r in llm_done if r["found"] and r["key"] not in judged)
    judge_later = sum(1 for q, s, v in todo if s in LLM_SYSTEMS)      # upper bound: 'Not found' needs no judge
    return {"pending_runs": len(todo), "gemini_expected": exp, "gemini_max": mx,
            "groq_judge_now": judge_now, "groq_judge_after_run_max": judge_now + judge_later}


def print_plan(p: dict, items: list[dict], s) -> None:
    used = requests_today(s.call_log, s.answer_model)
    limit = gemini_daily_limit()
    print(f"questions: {len(items)}; pending runs: {p['pending_runs']}")
    print(f"Gemini ({s.answer_model}): planned {p['gemini_expected']} calls (max {p['gemini_max']} with Text2Cypher "
          f"retries); used today {used}/{limit} -> {max(limit - used, 0)} left")
    print(f"Groq judge ({os.environ.get('JUDGE_MODEL', '?')}): {p['groq_judge_now']} calls now, "
          f"at most {p['groq_judge_after_run_max']} once every pending answer exists ('Not found' answers are not judged)")
    print("PLAN_JSON " + json.dumps({**p, "gemini_used_today": used, "gemini_daily_limit": limit}))


# ------------------------------------------------------------------ main loops
def do_run(items: list[dict], systems, max_gemini: int, confirm: bool) -> int:
    from analytics.db import get_driver
    from extraction.config import settings
    from extraction.llm import LLMCallFailed, QuotaExhausted

    s = settings()
    todo = pending(items, systems)
    need_llm = any(sy in LLM_SYSTEMS for _, sy, _ in todo)
    engine = None
    if need_llm:
        from graphrag.engine import default_engine
        engine = default_engine()
        p = plan(items, systems, lambda q, sy, v: uncached_steps(engine, q, sy, v))
        print_plan(p, items, s)
        if p["gemini_max"] and not confirm:
            print("PLAN_ONLY: re-run with --confirm (make benchmark ok=1) to make these calls")
            return 2
    driver = engine.driver if engine else get_driver()
    used0 = requests_today(s.call_log, s.answer_model)
    budget = min(max_gemini, gemini_daily_limit() - 20 - used0)     # 20 requests kept in reserve
    live = 0
    print(f"running {len(todo)} pending runs; Gemini budget this run: {max(budget, 0)}")
    for n, (q, sy, v) in enumerate(todo, 1):
        base = {"key": run_key(q, sy, v), "question_id": q["id"], "category": q["category"], "system": sy,
                "kg_version": v, "question": q["question"], "ts": datetime.now(timezone.utc).isoformat()}
        try:
            if sy == "direct":
                res = run_direct(driver, q, v)
            else:
                need = len(uncached_steps(engine, q, sy, v)) and MAX_CALLS[sy]
                if live + need > budget:
                    print(f"STOPPED_BUDGET: {live} live Gemini calls this run; the next answer could exceed today's "
                          f"budget. Re-run tomorrow (after midnight Pacific) to resume.")
                    return 3
                res = run_llm_system(engine, q, sy, v)
                live += res["llm_calls"] - res["cache_hits"]
        except QuotaExhausted as e:
            print(f"STOPPED_QUOTA: daily quota reached ({str(e)[:120]}). Results so far are saved; resume tomorrow.")
            return 3
        except LLMCallFailed as e:
            append_jsonl(RUNS, {**base, "status": "error", "error": str(e)[:500]})
            print(f"  error {base['key']}: {str(e)[:120]}")
            continue
        append_jsonl(RUNS, {**base, "status": "ok", **res})
        print(f"  [{n}/{len(todo)}] {q['id']} {sy:<6} {v:<8} calls={res['llm_calls']} (cached {res['cache_hits']}) "
              f"lat={res['latency_s']:.2f}s  {' '.join(res['answer'].split())[:70]}")
    print(f"RUN_COMPLETE: live Gemini calls this run: {live}")
    return 0


def do_judge(items: list[dict], confirm: bool) -> int:
    from .judge import Judge, judged_keys
    p = plan(items, list(SYSTEMS))
    print(f"Groq judge ({os.environ.get('JUDGE_MODEL', '?')}): {p['groq_judge_now']} calls planned")
    if p["groq_judge_now"] and not confirm:
        print("PLAN_ONLY: re-run with --confirm (make judge ok=1) to make these calls")
        return 2
    return Judge.default().judge_all(items, done_runs(), judged_keys())


def status(items: list[dict]) -> str:
    from .judge import judged_keys
    done = done_runs()
    judged = judged_keys()
    ids = {q["id"] for q in items}
    lines = [f"questions: {len(items)}", f"{'system':<15}{'resolved':>9}{'raw':>6}{'judged':>8}"]
    for sy, name in SYSTEMS.items():
        rs = [r for r in done.values() if r["system"] == sy and r["question_id"] in ids]
        lines.append(f"{name:<15}{sum(r['kg_version'] == 'resolved' for r in rs):>9}"
                     f"{sum(r['kg_version'] == 'raw' for r in rs):>6}{sum(r['key'] in judged for r in rs):>8}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(prog="eval.run_benchmark")
    ap.add_argument("cmd", choices=("plan", "run", "judge", "status"))
    ap.add_argument("--confirm", action="store_true", help="make the planned LLM calls")
    ap.add_argument("--all-drafts", action="store_true", help="include unverified questions (dry runs)")
    ap.add_argument("--systems", default="direct,A,B,C")
    ap.add_argument("--max-gemini", type=int, default=10_000, help="cap on live Gemini calls in this run")
    a = ap.parse_args()
    items = Q.load()
    items = [i for i in items if not i.get("dropped")] if a.all_drafts else Q.verified(items)
    systems = [x.strip() for x in a.systems.split(",") if x.strip() in SYSTEMS]
    if not items:
        print("no verified questions: tick them in eval/gold/questions_review.md, then make questions-apply")
        return 1
    if a.cmd == "status":
        print(status(items))
        return 0
    if a.cmd == "plan":
        from extraction.config import settings
        engine = None
        try:
            from graphrag.engine import default_engine
            engine = default_engine()
        except Exception as e:                                    # plan still works without Neo4j/fastembed
            print(f"(cache probe unavailable: {type(e).__name__}; assuming nothing cached)")
        probe = (lambda q, sy, v: uncached_steps(engine, q, sy, v)) if engine else None
        print_plan(plan(items, systems, probe), items, settings())
        return 0
    if a.cmd == "run":
        return do_run(items, systems, a.max_gemini, a.confirm)
    return do_judge(items, a.confirm)


if __name__ == "__main__":
    sys.exit(main())
