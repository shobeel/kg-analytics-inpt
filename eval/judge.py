"""LLM judge (Groq JUDGE_MODEL, another model family than the Gemini answer model) with a fixed rubric.

One call per answer returns:
  correctness  1-5 against the expected answer (rubric below)
  claims       the answer's atomic factual claims, each supported / not supported by the shown evidence
  sources      for every source shown with the answer: does its evidence support a claim of the answer?
-> faithfulness = supported claims / claims, citation precision = supporting sources / sources (RQ5).

Not judged (deterministic, no call): 'Not found' answers (correctness 1 on answerable questions, 5 on
unanswerable ones) and Direct Cypher (the gold query itself: scored by entity completeness only).
Results are cached by (JUDGE_VERSION, model, question, expected answer, answer, evidence), and appended
to eval/results/judge_scores.jsonl.
"""

import hashlib
import json
import os
from pathlib import Path

JUDGE_VERSION = "judge-v1"
SCORES = Path(__file__).parent / "results" / "judge_scores.jsonl"
ERRORS = Path(__file__).parent / "results" / "judge_errors.jsonl"
NOT_FOUND = "Not found in the knowledge graph."
MAX_SOURCES = 8
EVIDENCE_WORDS = 70

RUBRIC = """\
Correctness (compare the SYSTEM ANSWER with the EXPECTED ANSWER; the expected answer comes from a
verified knowledge graph query):
  5  correct and complete: every key fact/entity of the expected answer, nothing wrong
  4  correct, with minor omissions (most key entities present) or harmless extra detail
  3  partly correct: some key facts right, but important omissions or one clear error
  2  mostly wrong or off-topic, with only a small correct part
  1  wrong, or claims nothing is known although the expected answer exists
Extra facts that are not in the expected answer are not errors unless they contradict it. For a count,
the number must match to score 5. If the expected answer is "Not found in the knowledge graph.", score 5
if the system declines to answer and 1 if it asserts an answer."""

SYSTEM = f"""You are a strict, consistent evaluator of question-answering systems over a knowledge graph
about the AI ecosystem. Judge only from the texts you are given, never from your own knowledge.

{RUBRIC}

Faithfulness: split the SYSTEM ANSWER into its atomic factual claims (at most 8; ignore citation
markers). For each claim, supported=true only if the EVIDENCE shown below states it (paraphrase is fine).
Citations: for every EVIDENCE item (by its id), supports=true if it supports at least one claim of the answer.
Return JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "integer", "minimum": 1, "maximum": 5},
        "rationale": {"type": "string"},
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "claim": {"type": "string"}, "supported": {"type": "boolean"}}, "required": ["claim", "supported"]}},
        "sources": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "supports": {"type": "boolean"}}, "required": ["id", "supports"]}},
    },
    "required": ["correctness", "rationale", "claims", "sources"],
}


def clip(text: str, words: int = EVIDENCE_WORDS) -> str:
    w = " ".join((text or "").split()).split(" ")
    return " ".join(w[:words]) + (" ..." if len(w) > words else "")


def evidence_items(run: dict) -> list[dict]:
    """The sources shown to the user with this answer (what provenance offers to check it)."""
    out = []
    for i, s in enumerate(run.get("sources") or []):
        if len(out) == MAX_SOURCES:
            break
        out.append({"id": s.get("short_id") or f"s{i}", "title": s.get("title") or "",
                    "evidence": clip(s.get("evidence_text") or "")})
    return out


def user_prompt(q: dict, run: dict) -> str:
    ev = evidence_items(run)
    ev_txt = "\n".join(f"[{e['id']}] {e['title']}: {e['evidence']}" for e in ev) or "(no evidence shown)"
    return (f"QUESTION: {q['question']}\n\nEXPECTED ANSWER: {q['expected_answer']}\n"
            f"EXPECTED KEY ENTITIES: {', '.join(q.get('expected_entities') or []) or '(none)'}\n\n"
            f"SYSTEM ANSWER: {run['answer']}\n\nEVIDENCE shown with the answer:\n{ev_txt}")


def needs_judge(run: dict) -> bool:
    return run["system"] != "direct" and bool(run.get("found")) and run.get("answer") != NOT_FOUND


def deterministic(q: dict, run: dict) -> dict | None:
    """Scores that need no judge call: 'Not found' answers. None means: judge it."""
    if run.get("answer") == NOT_FOUND or not run.get("found"):
        ok = q["category"] == "unanswerable"
        return {"correctness": 5 if ok else 1, "rationale": "declined (Not found)", "claims": [], "sources": [],
                "judged_by": "rule"}
    return None


def cache_key(model: str, q: dict, run: dict) -> str:
    raw = json.dumps([JUDGE_VERSION, model, q["question"], q["expected_answer"], run["answer"],
                      evidence_items(run)], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def derive(parsed: dict, run: dict) -> dict:
    claims = parsed.get("claims") or []
    shown = {e["id"] for e in evidence_items(run)}
    src = [s for s in parsed.get("sources") or [] if s.get("id", "").strip("[] ") in shown]
    return {"correctness": max(1, min(5, int(parsed["correctness"]))),
            "faithfulness": (sum(bool(c["supported"]) for c in claims) / len(claims)) if claims else None,
            "citation_precision": (sum(bool(s["supports"]) for s in src) / len(src)) if src else None,
            "n_claims": len(claims), "n_sources_judged": len(src)}


def judged_keys() -> dict:
    out = {}
    if SCORES.exists():
        for l in SCORES.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                out[r["run_key"]] = r
    return out


class Judge:
    def __init__(self, llm, model: str, cache_dir: str, extra_body: dict | None = None):
        self.llm, self.model = llm, model
        self.extra_body = {"reasoning_effort": "low"} if extra_body is None else extra_body
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls) -> "Judge":
        from extraction.config import settings
        from extraction.llm import LLMClient
        s = settings()
        model = os.environ.get("JUDGE_MODEL", "").split(" #")[0].strip()
        if not model or not s.prefill_api_key:
            raise RuntimeError("JUDGE_MODEL and GROQ_API_KEY must be set in .env")
        if model == s.answer_model:
            raise RuntimeError("JUDGE_MODEL must differ from ANSWER_MODEL")
        llm = LLMClient(s, base_url=s.prefill_base_url, api_key=s.prefill_api_key,
                        min_interval_s=float(os.environ.get("JUDGE_MIN_INTERVAL_S", "8")))
        return cls(llm, model, os.environ.get("JUDGE_CACHE_DIR", "/data/cache/judge"))

    def score(self, q: dict, run: dict) -> dict:
        key = cache_key(self.model, q, run)
        p = self.cache / f"{key}.json"
        if p.exists():
            hit = json.loads(p.read_text())
        else:
            raw, usage = self.llm.chat_json(system=SYSTEM, user=user_prompt(q, run), schema=SCHEMA,
                                            schema_name="judge", purpose=f"judge-{JUDGE_VERSION}",
                                            ref=run["key"], model=self.model,
                                            extra_body=self.extra_body or None)
            hit = {"raw": raw, "usage": usage, "model": self.model}
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(hit, ensure_ascii=False))
            os.replace(tmp, p)
        parsed = json.loads(hit["raw"])
        return {**parsed, **derive(parsed, run), "judged_by": self.model, "judge_usage": hit["usage"]}

    def judge_all(self, items: list[dict], runs: dict, done: dict) -> int:
        from extraction.llm import LLMCallFailed, QuotaExhausted
        by_id = {q["id"]: q for q in items}
        todo = [r for r in runs.values() if r["question_id"] in by_id and needs_judge(r) and r["key"] not in done]
        print(f"judging {len(todo)} answers with {self.model}")
        SCORES.parent.mkdir(parents=True, exist_ok=True)
        for n, run in enumerate(todo, 1):
            try:
                res = self.score(by_id[run["question_id"]], run)
            except QuotaExhausted as e:
                print(f"STOPPED_QUOTA: Groq daily quota reached ({str(e)[:120]}). Saved; resume tomorrow.")
                return 3
            except (LLMCallFailed, json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"  judge error {run['key']}: {str(e)[:150]}")
                with ERRORS.open("a") as f:      # reported by eval.metrics, retried on the next run
                    f.write(json.dumps({"run_key": run["key"], "model": self.model, "error": str(e)[:500]}) + "\n")
                continue
            with SCORES.open("a") as f:
                f.write(json.dumps({"run_key": run["key"], "judge_version": JUDGE_VERSION, **res},
                                   ensure_ascii=False) + "\n")
            print(f"  [{n}/{len(todo)}] {run['key']}: correctness={res['correctness']} "
                  f"faithfulness={res['faithfulness']} citation_precision={res['citation_precision']}")
        print("JUDGE_COMPLETE")
        return 0
