"""Phase 6 unit tests: question review round trip, metrics, judge scoring rules, agreement statistics,
quota accounting, plan counting and report generation on synthetic results. No Neo4j, no LLM."""

import json
from datetime import datetime, timezone

import pytest

from eval import judge as J
from eval import judge_validation as JV
from eval import metrics as M
from eval import questions as Q
from eval import run_benchmark as R

NF = "Not found in the knowledge graph."


def q(qid="M01", cat="multi-hop", ents=("Demis Hassabis", "Shane Legg"), **kw):
    return {"id": qid, "category": cat, "question": "Which people work at the organization that developed AlphaFold?",
            "gold_query": "MATCH (e {kg_version: $kg_version}) RETURN e.name AS entity", "verified": True,
            "expected_answer": "Demis Hassabis, Shane Legg", "expected_entities": list(ents),
            "gold_rows_resolved": 2, "gold_preview": [], **kw}


def run(system="B", answer="Demis Hassabis and Shane Legg work at DeepMind [099acf82].", version="resolved",
        qq=None, **kw):
    qq = qq or q()
    return {"key": R.run_key(qq, system, version), "question_id": qq["id"], "category": qq["category"],
            "system": system, "kg_version": version, "question": qq["question"], "status": "ok", "answer": answer,
            "found": answer != NF, "sources": [{"short_id": "099acf82", "title": "AlphaFold", "url": "u",
                                                 "evidence_text": "Hassabis co-founded DeepMind with Shane Legg", "cited": True}],
            "llm_calls": 1, "cache_hits": 0, "tokens_in": 1000, "tokens_out": 50, "latency_s": 2.0, **kw}


# ------------------------------------------------------------------ question set
def test_drafts_have_the_required_mix_and_take_kg_version():
    cats = [d[1] for d in Q.DRAFTS]
    assert {c: cats.count(c) for c in Q.CATEGORIES} == Q.CATEGORIES
    assert len({d[0] for d in Q.DRAFTS}) == 30
    assert all("$kg_version" in d[3] for d in Q.DRAFTS)
    assert all(d[5].get("anchor") and d[5]["hops"] in (2, 3) for d in Q.DRAFTS if d[1] == "multi-hop")


def test_check_drafts_flags_problems():
    items = [q(cat=c, qid=f"X{i:02d}") for i, c in enumerate(Q.CATEGORIES)]
    items[1]["anchor_in_lcc"] = False
    items[4]["gold_rows_resolved"] = 3                         # unanswerable with rows
    probs = Q.check_drafts(items)
    assert any("largest connected component" in p for p in probs)
    assert any("unanswerable but" in p for p in probs)


def test_review_round_trip_applies_ticks_and_edits():
    items = [q("L01", "lookup"), q("M01"), q("S01", "semantic")]
    text = Q.render_review(items, [])
    text = text.replace("## L01 · lookup\n\n[ ] OK", "## L01 · lookup\n\n[x] OK", 1)
    blk = text.index("## M01")
    text = text[:blk] + text[blk:].replace("[ ] FIX", "[x] FIX", 1).replace(
        "expected_entities: Demis Hassabis ; Shane Legg", "expected_entities: Demis Hassabis ; Mustafa Suleyman", 1)
    out, summary = Q.apply_review(items, Q.parse_review(text))
    by = {i["id"]: i for i in out}
    assert by["L01"]["verified"] and by["M01"]["verified"] and not by["S01"]["verified"]
    assert by["M01"]["expected_entities"] == ["Demis Hassabis", "Mustafa Suleyman"]
    assert summary["OK"] == 1 and summary["FIX"] == 1 and summary["unticked"] == 1
    assert [i["id"] for i in Q.verified(out)] == ["L01", "M01"]


def test_review_drop_and_ambiguous():
    items = [q("L01", "lookup"), q("L02", "lookup")]
    text = Q.render_review(items, [])
    text = text.replace("[ ] DROP", "[x] DROP", 1)
    i2 = text.index("## L02")
    text = text[:i2] + text[i2:].replace("[ ] OK   [ ] FIX", "[x] OK   [x] FIX", 1)
    out, summary = Q.apply_review(items, Q.parse_review(text))
    assert out[0]["dropped"] and not out[0]["verified"]
    assert not out[1]["verified"] and summary["ambiguous"] == ["L02"]


def test_expected_entities_and_template():
    rows = [{"via": "Cohere", "entity": "Takane"}, {"via": "Cohere", "entity": "Aya Vision"}, {"entity": "Takane"}]
    ents = Q.expected_from_rows(rows)
    assert ents == ["Takane", "Aya Vision"]
    assert Q.fill_template("{first} ({n})", [{"entity": "DeepMind", "n": 34}], ["DeepMind"]) == "DeepMind (34)"


# ------------------------------------------------------------------ metrics
INDEX = {"demis hassabis": ["Person"], "shane legg": ["Person"], "mustafa suleyman": ["Person"],
         "deepmind": ["Organization"], "alphafold": ["AIModel"], "google": ["Organization"], "jaan tallinn": ["Person"]}


def test_contains_entity_is_token_based_and_relaxed():
    assert M.contains_entity("Founded by Demis  Hassabis.", "Demis Hassabis")
    assert M.contains_entity("Mistral AI, Inc. is in Paris", "Mistral AI")
    assert not M.contains_entity("Claude Code is a tool", "Code Red")
    assert not M.contains_entity("GPT-40", "GPT-4")


def test_mentions_prefers_longest_match():
    idx = {"claude": ["AIModel"], "claude code": ["Product"]}
    assert M.mentions("Claude Code and Claude", idx) == ["claude code", "claude"]


def test_entity_scores_precision_ignores_question_entities():
    s = M.entity_scores("Demis Hassabis and Jaan Tallinn work at DeepMind, which developed AlphaFold.",
                        ["Demis Hassabis", "Shane Legg"], "Who works at the organization that developed AlphaFold?", INDEX)
    assert s["entity_recall"] == 0.5
    assert s["entity_precision"] == 0.5          # Demis found; Jaan Tallinn is a wrong Person; DeepMind is not a Person
    assert s["entity_f1"] == pytest.approx(0.5)


def test_entity_scores_not_found_and_unanswerable():
    assert M.entity_scores(NF, ["Anthropic"])["entity_recall"] == 0.0
    assert M.entity_scores(NF, [])["entity_recall"] is None


def test_percentiles_and_cost():
    assert M.pct([1, 2, 3, 4, 5], 0.5) == 3
    assert M.pct([1, 2, 3, 4, 5], 0.95) == pytest.approx(4.8)
    assert M.cost_usd(1_000_000, 100_000, {"input_per_m": 0.3, "output_per_m": 2.5}) == pytest.approx(0.55)


def test_result_rows_and_summaries():
    qa, qu = q(), q("U01", "unanswerable", ents=())
    runs = [run("B"), run("A", NF), run("direct", "entity: Demis Hassabis", llm_calls=0, tokens_in=0, tokens_out=0),
            run("B", NF, qq=qu), run("B", "INPT's CEO is X.", version="raw", qq=qu),
            run("B", version="raw")]
    judged = {runs[0]["key"]: {"correctness": 4, "faithfulness": 1.0, "citation_precision": 1.0, "judged_by": "j"},
              runs[4]["key"]: {"correctness": 1, "faithfulness": 0.0, "citation_precision": 0.0, "judged_by": "j"},
              runs[5]["key"]: {"correctness": 3, "faithfulness": 0.5, "citation_precision": 1.0, "judged_by": "j"}}
    prices = {"answer_model": {"input_per_m": 0.3, "output_per_m": 2.5}}
    rows = M.result_rows([qa, qu], runs, judged, INDEX, prices)
    by = {(r["system"], r["question_id"], r["kg_version"]): r for r in rows}
    assert by[("A", "M01", "resolved")]["correctness"] == 1            # Not found on an answerable question
    assert by[("B", "U01", "resolved")]["correctness"] == 5            # correct refusal
    assert by[("direct", "M01", "resolved")]["correctness"] is None
    assert by[("direct", "M01", "resolved")]["entity_recall"] == 0.5
    s = M.summarize([r for r in rows if r["system"] == "B"])
    assert s["refusal_rate_unanswerable"] == 0.5 and s["false_refusal_rate"] == 0.0
    rr = M.paired_raw_resolved(rows)
    assert rr["B"]["pairs"] == 2 and rr["B"]["better_resolved"] == 2


# ------------------------------------------------------------------ judge
def test_judge_rules_and_derivation():
    qa, qu = q(), q("U01", "unanswerable", ents=())
    assert J.deterministic(qa, run(answer=NF))["correctness"] == 1
    assert J.deterministic(qu, run(answer=NF, qq=qu))["correctness"] == 5
    assert J.deterministic(qa, run()) is None
    assert not J.needs_judge(run("direct")) and not J.needs_judge(run(answer=NF)) and J.needs_judge(run())
    d = J.derive({"correctness": 7, "claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}],
                  "sources": [{"id": "099acf82", "supports": True}, {"id": "deadbeef", "supports": True}]}, run())
    assert d["correctness"] == 5 and d["faithfulness"] == 0.5
    assert d["citation_precision"] == 1.0 and d["n_sources_judged"] == 1     # ids not shown are ignored


def test_judge_prompt_and_cache_key():
    r = run()
    p = J.user_prompt(q(), r)
    assert "EXPECTED ANSWER: Demis Hassabis, Shane Legg" in p and "[099acf82] AlphaFold" in p
    assert J.cache_key("m", q(), r) == J.cache_key("m", q(), dict(r))
    assert J.cache_key("m", q(), r) != J.cache_key("m", q(), {**r, "answer": "other"})


def test_judge_all_with_fake_llm(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "SCORES", tmp_path / "scores.jsonl")

    class FakeLLM:
        calls = 0

        def chat_json(self, **kw):
            FakeLLM.calls += 1
            return json.dumps({"correctness": 4, "rationale": "ok", "claims": [{"claim": "x", "supported": True}],
                               "sources": [{"id": "099acf82", "supports": True}]}), {"input_tokens": 1, "output_tokens": 1,
                                                                                     "latency_s": 0.1}

    jd = J.Judge(FakeLLM(), "fake", str(tmp_path / "cache"))
    runs = {r["key"]: r for r in (run(), run(answer=NF, system="A"), run("direct"))}
    assert jd.judge_all([q()], runs, {}) == 0
    assert FakeLLM.calls == 1                                       # Not found and Direct are not judged
    jd.judge_all([q()], runs, {})                                   # second pass: cached
    assert FakeLLM.calls == 1
    assert J.judged_keys()[run()["key"]]["correctness"] == 4


# ------------------------------------------------------------------ judge validation
def test_agreement_statistics():
    assert JV.weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert JV.spearman([1, 2, 3, 4], [2, 3, 4, 5]) == pytest.approx(1.0)
    a = JV.agreement({"a": 5, "b": 3, "c": 1}, {"a": 5, "b": 4, "c": 4, "d": 2})
    assert a["n"] == 3 and a["exact"] == pytest.approx(1 / 3) and a["within_1"] == pytest.approx(2 / 3)
    assert a["disagreements"] == [{"key": "c", "human": 1, "judge": 4}]


def test_validation_sample_is_stratified_and_parsed():
    runs = [run(sy, qq=q(f"{c[0].upper()}0{i}", c)) for sy in "ABC" for c in ("lookup", "multi-hop") for i in range(4)]
    picks = JV.sample(runs, 15)
    assert len(picks) == 15 and len({p["key"] for p in picks}) == 15
    assert {p["system"] for p in picks} == set("ABC") and {p["category"] for p in picks} == {"lookup", "multi-hop"}
    md = JV.render(picks, {p["question_id"]: q(p["question_id"], p["category"]) for p in picks})
    assert "judged_by" not in md and "correctness=" not in md            # blind: no judge scores in the file
    md = md.replace("your score: ", "your score: 4", 2)
    assert list(JV.parse(md).values()) == [4, 4]


# ------------------------------------------------------------------ runner: quota, plan, rendering
def test_quota_day_starts_at_midnight_pacific():
    now = datetime(2026, 10, 29, 14, 0, tzinfo=timezone.utc)
    assert R.quota_day_start(now) == datetime(2026, 10, 29, 7, 0, tzinfo=timezone.utc)
    now = datetime(2026, 11, 3, 5, 0, tzinfo=timezone.utc)                # PST, before the reset
    assert R.quota_day_start(now) == datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc)


def test_requests_today_counts_only_today_and_model(tmp_path):
    log = tmp_path / "calls.jsonl"
    rows = [{"ts": "2026-10-29T06:59:00.000+00:00", "model": "g", "status": "ok"},
            {"ts": "2026-10-29T08:00:00.000+00:00", "model": "g", "status": "ok"},
            {"ts": "2026-10-29T09:00:00.000+00:00", "model": "g", "status": "error"},
            {"ts": "2026-10-29T09:00:00.000+00:00", "model": "other", "status": "ok"}]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    assert R.requests_today(str(log), "g", datetime(2026, 10, 29, 14, tzinfo=timezone.utc)) == 2


def test_render_rows():
    assert R.render_rows([]) == NF
    assert R.render_rows([{"entity": "Anthropic", "n": None}]) == "entity: Anthropic"
    assert "(+5 more rows)" in R.render_rows([{"entity": i} for i in range(25)])


def test_plan_counts_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RUNS", tmp_path / "runs.jsonl")
    monkeypatch.setattr(J, "SCORES", tmp_path / "scores.jsonl")
    items = [q(), q("U01", "unanswerable", ents=())]
    p = R.plan(items, list(R.SYSTEMS))
    # per version: A 1 + B 1 + C 2 (answerable) / A 1 + B 1 + C 1 (unanswerable) -> (4 + 3) x 2
    assert p["pending_runs"] == 16 and p["gemini_expected"] == 14 and p["gemini_max"] == 20
    R.append_jsonl(R.RUNS, run("B"))
    p = R.plan(items, list(R.SYSTEMS))
    assert p["pending_runs"] == 15 and p["groq_judge_now"] == 1
    cached = R.plan(items, list(R.SYSTEMS), cache_probe=lambda qq, s, v: [])
    assert cached["gemini_expected"] == 0


# ------------------------------------------------------------------ report
def test_report_builds_from_synthetic_results(tmp_path, monkeypatch):
    from eval import report as REP
    monkeypatch.setattr(REP, "FIGURES", tmp_path)
    qa, qu = q(), q("U01", "unanswerable", ents=())
    runs = [run(s, version=v) for s in ("A", "B", "C") for v in ("raw", "resolved")] + \
           [run("direct", "entity: Demis Hassabis", version=v, llm_calls=0) for v in ("raw", "resolved")] + \
           [run("B", NF, qq=qu), run("C", "Jaan Tallinn works at DeepMind", invalid_citations=["66848"])]
    judged = {r["key"]: {"correctness": 4, "faithfulness": 1.0, "citation_precision": 1.0, "judged_by": "j"}
              for r in runs if r["system"] != "direct" and r["answer"] != NF}
    res = M.compute([qa, qu], {r["key"] + str(i): r for i, r in enumerate(runs)}, judged, INDEX)
    figs = REP.figures(res, res["rows"])
    assert len(figs) >= 4 and all((tmp_path / f).stat().st_size > 5000 for f in figs)
    md = REP.build(res, runs, [qa, qu], figs)
    for section in ("## In one page", "## RQ1", "## RQ2", "## RQ3", "## RQ4", "## RQ5", "## Good and bad answers",
                    "## The five hardest questions", "## Judge validation", "## Cost", "## Limitations"):
        assert section in md
    for threat in ("Model-made gold sets", "Inter-model judge validation", "LLM-as-judge bias", "Single run",
                   "Small question set", "Free-tier constraints", "Domain-specific corpus"):
        assert threat in md
    assert "Fabricated citation caught" in md and "WORKS_AT investor error" in md


def test_pending_runs_resolved_before_raw_and_errors_are_final(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RUNS", tmp_path / "runs.jsonl")
    items = [q(), q("L01", "lookup")]
    order = [(qq["id"], v) for qq, s, v in R.pending(items, ["direct", "A"])]
    first_raw = next(i for i, (_, v) in enumerate(order) if v == "raw")
    assert all(v == "resolved" for _, v in order[:first_raw]) and all(v == "raw" for _, v in order[first_raw:])
    R.append_jsonl(R.RUNS, {**run("A"), "status": "error", "error": "boom"})
    assert run("A")["key"] in R.error_runs()
    assert all(not (qq["id"] == "M01" and s == "A" and v == "resolved") for qq, s, v in R.pending(items, ["A"]))


def test_inter_model_faithfulness_agreement():
    a = [{"faithfulness": 1.0}, {"faithfulness": 0.5}, {"faithfulness": None}]
    b = [{"faithfulness": 1.0}, {"faithfulness": 1.0}, {"faithfulness": 1.0}]
    r = JV.faithful_agreement(a, b, "faithfulness")
    assert r["n"] == 2 and r["binary_agreement"] == pytest.approx(0.5) and r["mean_abs_diff"] == pytest.approx(0.25)


def test_report_number_check(tmp_path):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "scripts" / "check_report_numbers.py"
    if not path.exists():
        pytest.skip("scripts/ is not in this repository")
    spec = importlib.util.spec_from_file_location("chk", path)
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    toks = [t for _, t in chk.report_numbers("RQ1 F1 0.84 and `G01 34` and gpt-oss-120b cost $0.12, 73% of 1,000")]
    assert toks == ["0.84", "$0.12", "73%", "1,000"]
    assert "73" in chk.variants(0.73) and "0.84" in chk.variants(0.8412)
