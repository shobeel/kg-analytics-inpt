import json

import pytest

from graphrag import NOT_FOUND
from graphrag.cache import LLMCache, cache_key
from graphrag.engine import Engine, aggregate_intents, est_tokens, preferred_relations, rrf, short_id
from graphrag.linking import candidates, keywords
from graphrag.safety import MAX_LIMIT, UnsafeCypher, check_cypher

V = "MATCH (e:Entity {kg_version: $kg_version}) RETURN e.name"


# ------------------------------------------------------------------ Cypher safety filter
@pytest.mark.parametrize("q", [
    "MATCH (n) WHERE n.kg_version = $kg_version DETACH DELETE n",
    "CREATE (n:Entity {kg_version: $kg_version}) RETURN n",
    "MATCH (n {kg_version: $kg_version}) SET n.name = 'x' RETURN n",
    "MERGE (n:Entity {kg_version: $kg_version}) RETURN n",
    "MATCH (n {kg_version: $kg_version}) REMOVE n.name RETURN n",
    "CALL dbms.components() YIELD name RETURN name, $kg_version",
    "CALL apoc.cypher.runWrite('CREATE ()', {}) YIELD value RETURN value, $kg_version",
    "MATCH (n {kg_version: $kg_version}) RETURN n; MATCH (m) DELETE m",
    "LOAD CSV FROM 'file:///x' AS row RETURN row, $kg_version",
    "MATCH (n {kg_version: $kg_version}) FOREACH (x IN [1] | CREATE ()) RETURN n",
    "DROP INDEX entity_names",
    "match (n {kg_version: $kg_version}) detach delete n",                 # lower case
    "MATCH (n) RETURN n",                                                    # no kg_version
])
def test_unsafe_cypher_is_rejected(q):
    with pytest.raises(UnsafeCypher):
        check_cypher(q)


def test_safe_cypher_gets_a_limit():
    assert check_cypher(V).endswith(f"LIMIT {MAX_LIMIT}")
    assert check_cypher(V + " LIMIT 10").endswith("LIMIT 10")
    assert check_cypher(V + " LIMIT 5000").endswith(f"LIMIT {MAX_LIMIT}")
    assert check_cypher(V + " LIMIT $n").endswith(f"LIMIT {MAX_LIMIT}")
    assert check_cypher("```cypher\n" + V + ";\n```").endswith(f"LIMIT {MAX_LIMIT}")


def test_names_inside_strings_do_not_trigger_or_hide_keywords():
    q = "MATCH (e:Entity {kg_version: $kg_version}) WHERE e.name = 'Create ML' OR e.name = 'Set' RETURN e"
    assert check_cypher(q).startswith("MATCH")
    with pytest.raises(UnsafeCypher):                    # a write hidden after a comment is still found
        check_cypher("MATCH (e {kg_version: $kg_version}) // harmless\n DELETE e")
    ok = "CALL db.index.fulltext.queryNodes('entity_names', 'x') YIELD node WHERE node.kg_version = $kg_version RETURN node"
    assert "LIMIT" in check_cypher(ok)


# ------------------------------------------------------------------ linking and intents
def test_candidates_keep_names_and_skip_question_words():
    assert candidates("Who developed Claude?") == ["Claude"]
    c = candidates("Which people work at the organization that developed AlphaFold?")
    assert c == ["AlphaFold"]
    assert "DeepMind" in candidates("What is DeepMind known for?")
    assert candidates("Which organization developed the most AI models?") == []
    assert candidates("Who is the CEO of INPT?") == ["INPT"]
    assert "GPT-4" in candidates("Is GPT-4 based on the Transformer?")
    assert "Google DeepMind" in candidates("Where is Google DeepMind located?")


def test_keywords_and_intents():
    assert keywords("What is DeepMind known for?") == ["DeepMind"]
    assert "WORKS_AT" in preferred_relations("Which people work at the organization that developed AlphaFold?")
    assert "DEVELOPED_BY" in preferred_relations("Who developed Claude?")
    assert aggregate_intents("Which organization developed the most AI models?") == ["agg_model_developers"]
    assert aggregate_intents("Who developed Claude?") == []


def test_rrf_short_id_tokens():
    assert rrf([["a", "b", "c"], ["c", "b"], ["b"]])[0] == "b"
    assert short_id("3f2a9c1b00112233") == "3f2a9c1b" and est_tokens("one two three") == 4


def test_cache_key_and_roundtrip(tmp_path):
    k1 = cache_key("B", "Who  developed Claude?", "resolved", "v1", "answer")
    assert k1 == cache_key("B", "Who developed Claude?", "resolved", "v1", "answer")       # whitespace-insensitive
    assert k1 != cache_key("B", "Who developed Claude?", "raw", "v1", "answer")
    assert k1 != cache_key("C", "Who developed Claude?", "resolved", "v1", "answer")
    c = LLMCache(str(tmp_path))
    c.put(k1, {"raw": "{}"})
    assert c.get(k1) == {"raw": "{}"} and c.get("missing") is None


# ------------------------------------------------------------------ engine with fakes
class FakeLLM:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    def chat_json(self, **kw):
        self.calls += 1
        return json.dumps(self.outputs.pop(0)), {"input_tokens": 100, "output_tokens": 10, "thinking_tokens": 0,
                                                  "latency_s": 0.5}


class S:
    answer_model = "fake-model"


def engine(tmp_path, outputs):
    e = Engine(driver=None, llm=FakeLLM(outputs), settings=S(), cache=LLMCache(str(tmp_path)), embed_fn=None)
    e.docs = lambda ids: {d: {"doc_id": d, "title": "T", "url": "https://x/" + d} for d in ids}
    return e


def test_text2cypher_rejects_write_retries_once_then_not_found(tmp_path):
    e = engine(tmp_path, [{"cypher": "MATCH (n) DETACH DELETE n"}, {"cypher": V}])
    executed = []
    e.execute_read = lambda q, v: executed.append(q) or []
    r = e.answer("Who is the CEO of INPT?", "C", "resolved")
    assert r["answer"] == NOT_FOUND and r["sources"] == []
    assert len(executed) == 1 and "DELETE" not in executed[0]                     # the write never ran
    assert "safety filter" in r["cypher_errors"][0] and r["llm_calls"] == 2       # no verbalize call on 0 rows


def test_text2cypher_verbalizes_rows_with_sources_and_caches(tmp_path):
    out = [{"cypher": V}, {"answer": "Claude was developed by Anthropic [aaaaaaaa].", "found": True,
                            "cited_ids": ["aaaaaaaa"]}]
    e = engine(tmp_path, out)
    e.execute_read = lambda q, v: [{"model": "Claude", "org": "Anthropic", "evidence": "Claude ... by Anthropic",
                                    "doc_id": "aaaaaaaa1111"}]
    r = e.answer("Who developed Claude?", "C", "resolved")
    assert r["found"] and r["sources"][0]["doc_id"] == "aaaaaaaa1111" and r["sources"][0]["url"]
    assert r["sources"][0]["evidence_text"] == "Claude ... by Anthropic"
    assert r["llm_calls"] == 2 and r["cache_hits"] == 0 and r["tokens_in"] == 200
    r2 = e.answer("Who developed Claude?", "C", "resolved")                       # second run: all cached
    assert r2["cache_hits"] == 2 and e.llm.calls == 2 and r2["answer"] == r["answer"]


def test_not_found_answers_drop_sources(tmp_path):
    e = engine(tmp_path, [])
    r = e.finish({}, {"answer": "Not found in the knowledge graph.", "found": False, "cited_ids": ["x"]}, {})
    assert r["answer"] == NOT_FOUND and r["sources"] == []
    r = e.finish({}, {"answer": "X [bbbbbbbb] and [zzzzzzzz]", "found": True, "cited_ids": []},
                 {"bbbbbbbb": {"doc_id": "bbbbbbbb99", "evidence_text": "ev"}})
    assert [s["short_id"] for s in r["sources"]] == ["bbbbbbbb"]                 # unknown ids are dropped


def test_context_respects_token_budget(tmp_path):
    e = engine(tmp_path, [])
    triples = [{"source": f"S{i}", "source_label": "AIModel", "relation": "DEVELOPED_BY", "target": "O",
                "target_label": "Organization", "evidence": "word " * 50, "doc_id": f"{i:08d}xx"} for i in range(200)]
    ctx, ev = e.build_context([], triples, [], [])
    assert est_tokens(ctx) <= 2500 + 80 and 0 < len(ev) < 200


def test_invalid_citations_are_removed_and_reported(tmp_path):
    e = engine(tmp_path, [])
    r = e.finish({}, {"answer": "DeepMind is a lab [66848]. It built AlphaFold [aaaaaaaa, 12345678].", "found": True,
                      "cited_ids": ["66848", "aaaaaaaa"]}, {"aaaaaaaa": {"doc_id": "aaaaaaaa00", "evidence_text": "ev"}})
    assert r["answer"] == "DeepMind is a lab. It built AlphaFold [aaaaaaaa]."
    assert r["invalid_citations"] == ["66848", "aaaaaaaa, 12345678"]
    assert [x["short_id"] for x in r["sources"]] == ["aaaaaaaa"] and r["sources"][0]["cited"]


def test_fastapi_ask_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    import graphrag.app as app_mod
    monkeypatch.setattr(app_mod, "answer", lambda q, s, v: {"answer": NOT_FOUND, "strategy": s, "kg_version": v,
                                                            "sources": []})
    c = TestClient(app_mod.app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/ask", json={"question": "Who is the CEO of INPT?", "strategy": "c", "kg_version": "raw"})
    assert r.status_code == 200 and r.json()["answer"] == NOT_FOUND and r.json()["strategy"] == "c"
    assert c.post("/ask", json={"question": "x?", "kg_version": "both"}).status_code == 422
