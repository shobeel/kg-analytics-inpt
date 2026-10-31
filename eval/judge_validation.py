"""Validate the LLM judge on 15 judged answers, stratified by system and category, scored blind.

  make   eval/gold/judge_validation.md: 15 answers with the question, expected answer, system answer and
         its evidence, and a `your score:` line per answer (kept if it exists, unless --force)
  score  agreement between hand scores in that file and the judge: exact, within 1 point,
         quadratic-weighted Cohen's kappa, Spearman rho (also called by eval.metrics and the report)
  inter  inter-model agreement (not human validation): a second judge from another model family
         (VALIDATION_JUDGE_MODEL, Groq free tier) scores the same 15 answers with the same rubric and
         the same blind prompt -> eval/results/judge_validation.json. --confirm makes the calls (<= 15).
Only `inter --confirm` makes LLM calls.
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

FILE = Path(__file__).parent / "gold" / "judge_validation.md"
INTER = Path(__file__).parent / "results" / "judge_validation.json"
VALIDATION_MODEL = "qwen/qwen3.8-27b"
N = 15
SEED = 6


def sample(judged_runs: list[dict], n: int = N, seed: int = SEED) -> list[dict]:
    """Round-robin over (system, category) strata so every system and category appears."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in sorted(judged_runs, key=lambda r: r["key"]):
        strata[(r["system"], r["category"])].append(r)
    for v in strata.values():
        rng.shuffle(v)
    keys = sorted(strata)
    rng.shuffle(keys)
    out = []
    while len(out) < n and any(strata.values()):
        for k in keys:
            if strata[k] and len(out) < n:
                out.append(strata[k].pop())
    return out


def render(picks: list[dict], questions: dict) -> str:
    from .judge import RUBRIC, evidence_items
    out = ["# Judge validation: hand-score 15 answers", "",
           "Score each SYSTEM ANSWER 1-5 with the same rubric the judge uses (below), then run",
           "`make judge-agreement`. Write the number after `your score:`. The judge's scores are not shown here.", "",
           "```", RUBRIC, "```", ""]
    for i, r in enumerate(picks, 1):
        q = questions[r["question_id"]]
        out += [f"## V{i:02d} · {r['key']}", "", f"question: {q['question']}",
                f"expected answer: {q['expected_answer']}", "", "system answer:", "",
                "> " + " ".join(r["answer"].split()), ""]
        ev = evidence_items(r)
        if ev:
            out.append("evidence shown:")
            out += [f"- [{e['id']}] {e['title']}: {e['evidence']}" for e in ev]
            out.append("")
        out += ["your score: ", ""]
    return "\n".join(out)


def parse(text: str) -> dict:
    """{run_key: score} for the answers you scored."""
    out = {}
    for block in re.split(r"^## V\d\d · ", text, flags=re.M)[1:]:
        key = block.split("\n", 1)[0].strip()
        m = re.search(r"^your score:[ \t]*([1-5])\b", block, re.M)
        if m:
            out[key] = int(m.group(1))
    return out


def weighted_kappa(a: list[int], b: list[int], k: int = 5) -> float | None:
    if not a:
        return None
    n = len(a)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x - 1][y - 1] += 1
    ra = [sum(obs[i]) for i in range(k)]
    rb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = lambda i, j: (i - j) ** 2 / (k - 1) ** 2
    o = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    e = sum(w(i, j) * ra[i] * rb[j] for i in range(k) for j in range(k)) / (n * n)
    return 1 - o / e if e else None


def spearman(a: list[int], b: list[int]) -> float | None:
    if len(a) < 3:
        return None
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    sa = sum((x - ma) ** 2 for x in ra) ** 0.5
    sb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (sa * sb) if sa and sb else None


def agreement(human: dict, judge: dict) -> dict:
    keys = [k for k in human if k in judge]
    h, j = [human[k] for k in keys], [judge[k] for k in keys]
    return {"n": len(keys), "exact": sum(x == y for x, y in zip(h, j)) / len(keys) if keys else None,
            "within_1": sum(abs(x - y) <= 1 for x, y in zip(h, j)) / len(keys) if keys else None,
            "weighted_kappa": weighted_kappa(h, j), "spearman": spearman(h, j),
            "mean_human": sum(h) / len(h) if h else None, "mean_judge": sum(j) / len(j) if j else None,
            "disagreements": [{"key": k, "human": human[k], "judge": judge[k]} for k in keys if abs(human[k] - judge[k]) >= 2]}


def faithful_agreement(a: list[dict], b: list[dict], field: str) -> dict:
    """Per-answer agreement on a claim-level verdict: 'fully supported' (field == 1) yes/no, and the mean
    absolute difference of the fractions. Answers where either judge returned no value are skipped."""
    pairs = [(x[field], y[field]) for x, y in zip(a, b) if x.get(field) is not None and y.get(field) is not None]
    if not pairs:
        return {"n": 0, "binary_agreement": None, "mean_abs_diff": None}
    return {"n": len(pairs), "binary_agreement": sum((x == 1) == (y == 1) for x, y in pairs) / len(pairs),
            "mean_abs_diff": sum(abs(x - y) for x, y in pairs) / len(pairs)}


def picks() -> list[dict]:
    from .judge import judged_keys
    from .run_benchmark import done_runs
    judged = judged_keys()
    runs = [r for r in done_runs().values() if r["key"] in judged and judged[r["key"]]["judged_by"] != "rule"]
    return sample(runs) if len(runs) >= N else []


def second_judge():
    import os
    from extraction.config import settings
    from extraction.llm import LLMClient
    from .judge import Judge
    s = settings()
    model = os.environ.get("VALIDATION_JUDGE_MODEL", VALIDATION_MODEL).split(" #")[0].strip()
    if model in (s.answer_model, os.environ.get("JUDGE_MODEL", "").split(" #")[0].strip()):
        raise RuntimeError("VALIDATION_JUDGE_MODEL must differ from ANSWER_MODEL and JUDGE_MODEL")
    llm = LLMClient(s, base_url=s.prefill_base_url, api_key=s.prefill_api_key,
                    min_interval_s=float(os.environ.get("JUDGE_MIN_INTERVAL_S", "8")))
    extra = {"reasoning_effort": "none"} if model.startswith("qwen/") else {}
    return Judge(llm, model, os.environ.get("JUDGE_CACHE_DIR", "/data/cache/judge"), extra_body=extra)


def inter_model(confirm: bool) -> int:
    from . import questions as Q
    from .judge import judged_keys
    chosen = picks()
    if not chosen:
        print(f"fewer than {N} judged answers: run the judge first")
        return 1
    if not confirm:
        print(f"PLAN: {N} calls to the validation judge ({VALIDATION_MODEL}, Groq); re-run with --confirm")
        return 2
    qs = {q["id"]: q for q in Q.load()}
    judge, first = second_judge(), judged_keys()
    rows = []
    for r in chosen:
        try:
            res = judge.score(qs[r["question_id"]], r)
        except Exception as e:                     # recorded, never dropped silently
            print(f"  validation judge error {r['key']}: {str(e)[:150]}")
            rows.append({"key": r["key"], "error": str(e)[:300]})
            continue
        f = first[r["key"]]
        rows.append({"key": r["key"], "system": r["system"], "category": r["category"],
                     "judge": {k: f.get(k) for k in ("correctness", "faithfulness", "citation_precision")},
                     "second": {k: res.get(k) for k in ("correctness", "faithfulness", "citation_precision")}})
        print(f"  {r['key']}: judge={f['correctness']} second={res['correctness']}")
    ok = [x for x in rows if "error" not in x]
    rep = {"kind": "inter-model agreement, not human validation", "judge_model": next(iter(first.values()))["judged_by"]
           if first else None, "second_model": judge.model, "n_sampled": len(chosen), "errors": len(rows) - len(ok),
           "correctness": agreement({x["key"]: x["second"]["correctness"] for x in ok},
                                    {x["key"]: x["judge"]["correctness"] for x in ok}),
           "faithfulness": faithful_agreement([x["judge"] for x in ok], [x["second"] for x in ok], "faithfulness"),
           "citation_precision": faithful_agreement([x["judge"] for x in ok], [x["second"] for x in ok],
                                                    "citation_precision"),
           "rows": rows}
    INTER.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    c = rep["correctness"]
    print(f"inter-model correctness: n={c['n']} exact={c['exact']} within1={c['within_1']} kappa={c['weighted_kappa']}")
    print(f"wrote {INTER}")
    return 0


def inter_report() -> dict | None:
    return json.loads(INTER.read_text()) if INTER.exists() else None


def agreement_report() -> dict | None:
    """Hand-score agreement; None until at least one answer is hand-scored."""
    from .judge import judged_keys
    if not FILE.exists():
        return None
    human = parse(FILE.read_text())
    if not human:
        return None
    judge = {k: v["correctness"] for k, v in judged_keys().items()}
    return agreement(human, judge)


def main() -> int:
    from . import questions as Q
    from .judge import judged_keys
    from .run_benchmark import done_runs
    ap = argparse.ArgumentParser(prog="eval.judge_validation")
    ap.add_argument("cmd", choices=("make", "score", "inter"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    if a.cmd == "inter":
        return inter_model(a.confirm)
    if a.cmd == "make":
        if FILE.exists() and not a.force:
            print(f"{FILE} exists (scores in progress?); use --force to regenerate")
            return 1
        judged = judged_keys()
        runs = [r for r in done_runs().values() if r["key"] in judged and judged[r["key"]]["judged_by"] != "rule"]
        if len(runs) < N:
            print(f"only {len(runs)} judged answers so far; need {N}")
            return 1
        qs = {q["id"]: q for q in Q.load()}
        FILE.write_text(render(sample(runs), qs))
        print(f"wrote {FILE} ({N} answers)")
        return 0
    rep = agreement_report()
    print(json.dumps(rep, indent=1) if rep else f"no hand scores yet in {FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
