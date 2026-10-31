"""CLI:  python -m graphrag ask "Who developed Claude?" --strategy B --kg resolved [--json]
       python -m graphrag eval        (the Phase 5 acceptance questions, 3 strategies, resolved)"""

import argparse
import json
import logging
import sys
from pathlib import Path

from . import NOT_FOUND, STRATEGIES, answer

QUESTIONS = [("lookup", "Who developed Claude?"),
             ("multi-hop", "Which people work at the organization that developed AlphaFold?"),
             ("aggregation", "Which organization developed the most AI models?"),
             ("semantic", "What is DeepMind known for?"),
             ("unanswerable", "Who is the CEO of INPT?")]


def show(r: dict, context: bool = False) -> str:
    out = [f"[{r['strategy']} {r['strategy_name']} | {r['kg_version']}] {r['question']}", "", r["answer"], ""]
    if r.get("entities_used"):
        out.append("entities: " + ", ".join(f"{e['name']} ({e['label']})" for e in r["entities_used"]))
    if r.get("cypher"):
        out.append(f"cypher: {r['cypher']}")
    out.append(f"sources ({len(r['sources'])}):")
    for s in r["sources"]:
        tag = "" if s.get("cited", True) else " (supporting, not cited by the model)"
        out.append(f"  [{s['short_id']}] {s['title']} | {s['url']}{tag}\n        evidence: \"{s['evidence_text'][:220]}\"")
    if r.get("invalid_citations"):
        out.append(f"removed invalid citations: {r['invalid_citations']}")
    out.append(f"llm_calls={r['llm_calls']} (cached {r['cache_hits']}) tokens={r['tokens_in']}/{r['tokens_out']} "
               f"latency={r['latency_s']}s (retrieval {r['retrieval_s']}s)")
    if context:
        out += ["", "context:", r["context"]]
    return "\n".join(out)


def evaluate(kg: str) -> int:
    results = []
    for kind, q in QUESTIONS:
        for st in STRATEGIES:
            results.append({"kind": kind, **answer(q, st, kg)})
    Path("/app/eval/results").mkdir(parents=True, exist_ok=True)
    Path("/app/eval/results/phase5_answers.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"{'question':<13}{'strategy':<13}{'answer (summary)':<62}{'src':>4}{'llm':>5}{'tok in/out':>12}{'lat s':>7}")
    for r in results:
        summ = " ".join(r["answer"].split())[:59]
        print(f"{r['kind']:<13}{r['strategy'] + ' ' + r['strategy_name']:<13}{summ:<62}{len(r['sources']):>4}"
              f"{r['llm_calls']:>5}{str(r['tokens_in']) + '/' + str(r['tokens_out']):>12}{r['latency_s']:>7.2f}")
    print("\n== Question 2, full answers")
    for r in results:
        if r["kind"] == "multi-hop":
            print("\n" + show(r))
    ok = True
    for r in results:
        if r["kind"] == "unanswerable" and r["answer"] != NOT_FOUND:
            ok = False
            print(f"FAIL: strategy {r['strategy']} answered the unanswerable question")
        if r["kind"] != "unanswerable" and r["answer"] != NOT_FOUND and not all(s["url"] for s in r["sources"]):
            ok = False
            print(f"FAIL: strategy {r['strategy']} / {r['kind']}: source without URL")
    answered = [r for r in results if r["kind"] != "unanswerable"]
    for r in answered:
        if r["answer"] != NOT_FOUND and not r["sources"]:
            ok = False
            print(f"FAIL: strategy {r['strategy']} / {r['kind']}: answer without any source")
    print(f"\nanswered (not 'Not found'): {sum(r['answer'] != NOT_FOUND for r in answered)}/{len(answered)}; "
          f"answers with >=1 source: {sum(bool(r['sources']) for r in answered)}/{len(answered)}; "
          f"live LLM calls this run: {sum(r['llm_calls'] - r['cache_hits'] for r in results)}")
    print("EVAL_OK" if ok else "EVAL_FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(prog="graphrag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--strategy", default="B", choices=list(STRATEGIES) + [s.lower() for s in STRATEGIES])
    p.add_argument("--kg", default="resolved", choices=("raw", "resolved"))
    p.add_argument("--json", action="store_true")
    p.add_argument("--context", action="store_true", help="also print the retrieved context")
    p = sub.add_parser("eval")
    p.add_argument("--kg", default="resolved", choices=("raw", "resolved"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if a.cmd == "ask":
        r = answer(a.question, a.strategy, a.kg)
        print(json.dumps(r, indent=1, ensure_ascii=False) if a.json else show(r, a.context))
        return 0
    return evaluate(a.kg)


if __name__ == "__main__":
    sys.exit(main())
