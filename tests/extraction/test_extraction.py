import json

import numpy as np
import pytest

from extraction.cache import ResultCache, chunk_hash
from extraction.er import Thresholds, resolve_type
from extraction.extract import evidence_in_chunk, extract_chunk, postprocess
from extraction.llm import _is_daily_quota, _retry_delay
from extraction.normalize import is_acronym_pair, norm_key, version_signature
from extraction.schema import ExtractionResult

CHUNK = {"chunk_id": "c1", "title": "t",
         "text": "GPT-4 was developed by OpenAI.  It is used by Microsoft."}
GOOD = {"entities": [{"name": "GPT-4", "type": "AIModel", "description": "model"},
                     {"name": "OpenAI", "type": "Organization", "description": ""},
                     {"name": "openai", "type": "Organization", "description": "dup"}],
        "relations": [{"source": "GPT-4", "relation": "DEVELOPED_BY", "target": "OpenAI",
                       "evidence_text": "GPT-4 was developed by  OpenAI", "confidence": 0.95},
                      {"source": "OpenAI", "relation": "TRAINED_ON", "target": "GPT-4",
                       "evidence_text": "made up sentence", "confidence": 0.7}]}


class FakeLLM:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    def chat_json(self, **kw):
        self.calls += 1
        return self.outputs.pop(0), {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0,
                                     "total_tokens": 15, "latency_s": 0.1, "cost_usd": 0.0}


def test_postprocess_flags_evidence_schema_and_dedupes():
    out = postprocess(ExtractionResult.model_validate(GOOD), CHUNK["text"])
    assert [e["name"] for e in out["entities"]] == ["GPT-4", "OpenAI"]          # case-dup removed
    r1, r2 = out["relations"]
    assert r1["evidence_verified"] and r1["schema_valid"] and r1["endpoints_known"]
    assert not r2["evidence_verified"] and not r2["schema_valid"]               # flagged, not dropped


def test_invalid_json_is_retried_once_then_accepted():
    llm = FakeLLM(["{not json", json.dumps(GOOD)])
    rec = extract_chunk(llm, CHUNK)
    assert llm.calls == 2 and rec["status"] == "ok" and rec["attempts"] == 2 and len(rec["usage"]) == 2


def test_two_invalid_outputs_give_invalid_status():
    bad = json.dumps({"entities": [{"name": "X", "type": "NotAType", "description": ""}], "relations": []})
    rec = extract_chunk(FakeLLM([bad, bad]), CHUNK)
    assert rec["status"] == "invalid" and rec["entities"] == [] and "ValidationError" in rec["error"]


def test_evidence_check_is_whitespace_and_case_insensitive():
    assert evidence_in_chunk("gpt-4 WAS developed by openai.", CHUNK["text"])
    assert not evidence_in_chunk("GPT-5 was developed by OpenAI", CHUNK["text"])


def test_quota_and_retry_delay_parsing():
    daily = "429 Quota exceeded for metric generate_content_free_tier_requests, quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    minute = "429 quotaId GenerateRequestsPerMinutePerProjectPerModel-FreeTier 'retryDelay': '37s'"
    assert _is_daily_quota(daily) and not _is_daily_quota(minute)
    assert _retry_delay(minute) == 37.0 and _retry_delay("no hint") is None


def test_cache_roundtrip(tmp_path):
    c = ResultCache(str(tmp_path), "gemini-x", "v1")
    key = chunk_hash("some text")
    assert c.get(key) is None
    c.put(key, {"chunk_hash": key, "status": "ok"})
    assert c.get(key)["status"] == "ok" and len(c.all()) == 1


def test_normalization_rules():
    assert norm_key("The Hugging Face, Inc.") == "hugging face"
    assert norm_key("xAI (company)") == "xai"
    assert norm_key("Google DeepMind Ltd") == norm_key("google deepmind")
    assert version_signature("GPT-4") != version_signature("GPT-4o") != version_signature("GPT-3")
    assert version_signature("Llama 2 70B") == frozenset({"2", "70b"})
    assert is_acronym_pair("llms", "large language models") and not is_acronym_pair("gpt", "openai")


def _groups(names):
    return [{"key": norm_key(n), "name": n, "type": "AIModel", "description": "", "version": version_signature(n)}
            for n in names]


def test_resolution_auto_merge_version_guard_and_llm_adjudication():
    names = ["HuggingFace", "Hugging Face Hub", "GPT-4", "GPT-4o", "BERT", "BERT model"]
    g = _groups(names)
    emb = np.eye(len(g), dtype=np.float32)
    emb[2, 3] = emb[3, 2] = 0.95                     # GPT-4 vs GPT-4o look similar
    emb[4, 5] = emb[5, 4] = 0.9                      # BERT vs BERT model: ambiguous -> LLM
    asked = []

    def adjudicate(pairs):
        asked.extend((p["a"]["name"], p["b"]["name"]) for p in pairs)
        return {i: True for i in range(len(pairs))}

    uf, log, _ = resolve_type(g, emb, Thresholds(), adjudicate)
    decisions = {(names[r["i"]], names[r["j"]]): r["decision"] for r in log}
    assert decisions[("GPT-4", "GPT-4o")] == "reject_version_mismatch"
    assert uf.find(2) != uf.find(3)
    assert ("BERT", "BERT model") in asked and uf.find(4) == uf.find(5)
    assert decisions[("BERT", "BERT model")] == "merge_llm"


def test_nospace_variant_merges_without_llm():
    g = _groups(["Hugging Face", "HuggingFace"])
    uf, log, _ = resolve_type(g, np.eye(2, dtype=np.float32), Thresholds(), adjudicate=None)
    assert uf.find(0) == uf.find(1) and log[0]["decision"] == "merge_auto_string"


def test_quota_stop_is_clean_with_pending_work(monkeypatch, tmp_path):
    """A daily-quota error mid-run cancels pending chunks without crashing (regression)."""
    import extraction.extract as ex
    from extraction.config import Settings
    from extraction.llm import QuotaExhausted

    chunks = [{"chunk_id": f"c{i}", "doc_id": "d", "source": "rss", "chunk_index": i,
               "text": f"Chunk {i} text.", "title": "t", "url": "u"} for i in range(12)]
    calls = {"n": 0}

    def fake_extract(llm, chunk):
        calls["n"] += 1
        if calls["n"] > 2:
            raise QuotaExhausted("429 GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        return {"status": "ok", "attempts": 1, "usage": [{"input_tokens": 1, "output_tokens": 1,
                "thinking_tokens": 0, "total_tokens": 2, "latency_s": 0.0, "cost_usd": 0.0}],
                "entities": [], "relations": []}

    class FakeSpark:
        def stop(self):
            pass

    monkeypatch.setattr(ex, "get_spark", lambda app: FakeSpark())
    monkeypatch.setattr(ex, "read_chunks", lambda spark, hdfs: chunks)
    monkeypatch.setattr(ex, "extract_chunk", fake_extract)
    monkeypatch.setattr(ex, "LLMClient", lambda s: object())
    s = Settings(api_key="k", extraction_model="m", cache_dir=str(tmp_path), concurrency=1)
    summary = ex.run(s, write=False)
    assert summary["quota_exhausted"] and summary["chunks_extracted_total"] == 2
    assert summary["chunks_remaining"] == 10


def test_check_relations_repairs_direction_flags_self_loops_and_applies_tweaks():
    from extraction.extract import check_relations, recheck
    from extraction.relstats import classify, valid_under_prompt_table

    text = "Baidu released ERNIE 4.5. Common Crawl stores data on Amazon S3. Anthropic was founded by Dario Amodei."
    ents = [{"name": n, "type": t} for n, t in (("Baidu", "Organization"), ("ERNIE 4.5", "AIModel"),
            ("Common Crawl", "Organization"), ("Amazon S3", "Product"), ("Anthropic", "Organization"),
            ("Dario Amodei", "Person"))]
    rel = lambda s, r, t, ev: {"source": s, "relation": r, "target": t, "evidence_text": ev, "confidence": 0.9}
    out = check_relations(ents, [
        rel("Baidu", "DEVELOPED_BY", "ERNIE 4.5", "Baidu released ERNIE 4.5"),          # reversed
        rel("Common Crawl", "USES", "Amazon S3", "stores data on Amazon S3"),           # tweak: USES -> Product
        rel("Anthropic", "DEVELOPED_BY", "Dario Amodei", "Anthropic was founded by Dario Amodei"),  # triple
        rel("Baidu", "DEVELOPED_BY", "baidu", "Baidu released ERNIE 4.5"),              # self-loop
        rel("Baidu", "COMPETES_WITH", "Amazon S3", "Baidu released ERNIE 4.5")], text)  # symmetric: no swap
    rev, uses, founded, loop, comp = out
    assert (rev["source"], rev["target"], rev["direction_repaired"], rev["schema_valid"]) == \
        ("ERNIE 4.5", "Baidu", True, True)
    assert uses["schema_valid"] and founded["schema_valid"] and not uses["direction_repaired"]
    assert loop["self_loop"] and classify(loop) == "self_loop"
    assert comp["schema_valid"] and comp["source"] == "Baidu" and not comp["direction_repaired"]
    assert [classify(r) for r in (rev, uses, founded)] == ["repaired", "valid", "valid"]
    assert not valid_under_prompt_table(rev) and not valid_under_prompt_table(uses)
    # recomputing an already repaired record is stable
    again = recheck({"entities": ents, "relations": out}, text)["relations"]
    assert [(r["source"], r["direction_repaired"]) for r in again] == [(r["source"], r["direction_repaired"]) for r in out]


def test_prompt_table_is_frozen():
    from extraction.prompts import EXTRACTION_SYSTEM
    assert "- USES: (AIModel | Organization | Paper | Product | Technology) -> (AIModel | Dataset | Technology)" \
        in EXTRACTION_SYSTEM


def test_groq_retry_delay_parsing():
    assert _retry_delay("Please try again in 14.7075s. Need more tokens?") == 14.7075
