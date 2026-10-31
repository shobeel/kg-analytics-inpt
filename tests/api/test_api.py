"""Phase 7 API unit tests: authentication and roles, request validation, endpoint shapes, read-only
Cypher, metrics and JSON logs. Neo4j and the GraphRAG engine are replaced by fakes (no LLM call)."""

import json

import pytest
from fastapi.testclient import TestClient

from api import auth
from api import main as M
from api.monitoring import METRICS

ADMIN, VIEWER = "admin-key-0123456789abcdef", "viewer-key-0123456789abcdef"


class FakeEngine:
    def __init__(self):
        self.calls = []

    def answer(self, question, strategy, kg_version):
        self.calls.append((question, strategy, kg_version))
        if strategy == "C" and "boom" in question:
            raise ValueError("bad strategy input")
        return {"question": question, "strategy": strategy, "kg_version": kg_version, "answer": "Anthropic [abc123].",
                "sources": [{"short_id": "abc123", "title": "Claude", "url": "u", "evidence_text": "e", "cited": True}],
                "context": "SECRET PROMPT CONTEXT", "entities_used": [], "llm_calls": 1, "cache_hits": 0,
                "tokens_in": 900, "tokens_out": 40, "latency_s": 1.2}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEYS_ADMIN", ADMIN)
    monkeypatch.setenv("API_KEYS_VIEWER", VIEWER + ", short")          # keys under 16 chars are ignored
    monkeypatch.setenv("API_LOG", str(tmp_path / "api.jsonl"))
    queries = []

    def fake_read(cypher, **params):
        queries.append((cypher, params))
        if "RETURN 1 AS ok" in cypher:
            return [{"ok": 1}]
        if "pagerank" in cypher and "community, size" not in cypher:
            return [{"id": "e1", "name": "Anthropic", "label": "Organization", "pagerank": 11.88, "community": 65}]
        if "top_members" in cypher:
            return [{"community": 65, "size": 40, "top_members": ["Anthropic", "Claude"]}]
        if "toLower(e.name)" in cypher:
            return [{"id": "e1", "name": "Anthropic", "label": "Organization"}] if params["q"] != "nobody" else []
        if "r.key AS edge_id" in cypher:
            return [{"edge_id": "e2|DEVELOPED_BY|e1", "relation": "DEVELOPED_BY", "source": "e2", "target": "e1",
                     "id": "e2", "name": "Claude", "label": "AIModel", "mentions": 3, "evidence": "x"}]
        if "UNWIND range" in cypher:
            return [{"source": "Claude", "relation": "DEVELOPED_BY", "target": "Anthropic", "evidence": "Claude is ...",
                     "confidence": 0.9, "extractor_model": "m", "chunk_id": "c", "doc_id": "d", "document": "Claude",
                     "url": "u"}]
        return []

    eng = FakeEngine()
    monkeypatch.setattr(M, "read", fake_read)
    monkeypatch.setattr(M, "get_engine", lambda: eng)
    results = tmp_path / "results"
    results.mkdir()
    (results / "rq2_graph_quality.csv").write_text("metric,raw,resolved\nEntities,4403,4071\n")
    monkeypatch.setattr(M, "RESULTS", results)
    METRICS.reset()
    c = TestClient(M.app, raise_server_exceptions=False)
    c.queries, c.engine, c.log = queries, eng, tmp_path / "api.jsonl"
    return c


def H(key):
    return {"X-API-Key": key}


# ------------------------------------------------------------------ auth
def test_role_resolution_and_short_keys_ignored(monkeypatch):
    monkeypatch.setenv("API_KEYS_ADMIN", ADMIN)
    monkeypatch.setenv("API_KEYS_VIEWER", VIEWER + ",short")
    assert auth.role_of(ADMIN) == "admin" and auth.role_of(VIEWER) == "viewer"
    assert auth.role_of("short") is None and auth.role_of("") is None and auth.role_of(None) is None


def test_health_needs_no_key(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "neo4j": "ok"}


@pytest.mark.parametrize("method,path,body", [("post", "/ask", {"question": "Who developed Claude?"}),
                                              ("get", "/analytics/top", None), ("get", "/metrics", None)])
def test_missing_or_wrong_key_is_401(client, method, path, body):
    for headers in ({}, H("not-a-valid-key-at-all")):
        r = getattr(client, method)(path, json=body, headers=headers) if body else getattr(client, method)(path, headers=headers)
        assert r.status_code == 401


def test_viewer_can_ask_and_read_but_not_admin_endpoints(client):
    assert client.post("/ask", json={"question": "Who developed Claude?"}, headers=H(VIEWER)).status_code == 200
    for path in ("/analytics/top", "/analytics/communities", "/analytics/quality", "/graph/neighbours?entity=Anthropic",
                 "/provenance/e2|DEVELOPED_BY|e1"):
        assert client.get(path, headers=H(VIEWER)).status_code == 200, path
    assert client.get("/metrics", headers=H(VIEWER)).status_code == 403
    assert client.post("/admin/metrics/reset", headers=H(VIEWER)).status_code == 403
    assert client.get("/metrics", headers=H(ADMIN)).status_code == 200
    assert client.post("/admin/metrics/reset", headers=H(ADMIN)).json() == {"reset": True}


# ------------------------------------------------------------------ endpoints
def test_ask_validates_input_and_hides_context(client):
    r = client.post("/ask", json={"question": "Who developed Claude?", "strategy": "b", "kg_version": "raw"},
                    headers=H(VIEWER))
    assert r.status_code == 200 and "context" not in r.json()
    assert client.engine.calls[-1] == ("Who developed Claude?", "B", "raw")
    for bad in ({"question": "Hi", "strategy": "B"}, {"question": "Who developed Claude?", "strategy": "D"},
                {"question": "Who developed Claude?", "kg_version": "prod"}):
        assert client.post("/ask", json=bad, headers=H(VIEWER)).status_code == 422
    assert client.post("/ask", json={"question": "boom question", "strategy": "C"}, headers=H(VIEWER)).status_code == 400


def test_analytics_and_graph_endpoints_filter_on_kg_version(client):
    top = client.get("/analytics/top?kg_version=raw&limit=5", headers=H(VIEWER)).json()
    assert top[0]["name"] == "Anthropic"
    assert client.get("/analytics/top?kg_version=prod", headers=H(VIEWER)).status_code == 422
    assert client.get("/analytics/quality", headers=H(VIEWER)).json() == [{"metric": "Entities", "raw": "4403", "resolved": "4071"}]
    g = client.get("/graph/neighbours?entity=Anthropic", headers=H(VIEWER)).json()
    assert g["center"]["id"] == "e1" and g["edges"][0]["edge_id"] == "e2|DEVELOPED_BY|e1"
    assert client.get("/graph/neighbours?entity=nobody", headers=H(VIEWER)).status_code == 404
    for cypher, params in client.queries:
        if "RETURN 1" not in cypher:
            assert "kg_version" in cypher and params.get("v") in ("raw", "resolved")


def test_queries_are_read_only(client):
    client.get("/analytics/top", headers=H(VIEWER))
    client.get("/analytics/communities", headers=H(VIEWER))
    client.get("/graph/neighbours?entity=Anthropic", headers=H(VIEWER))
    client.get("/provenance/e2|DEVELOPED_BY|e1", headers=H(VIEWER))
    from graphrag.safety import WRITE, strip_literals
    for cypher, _ in client.queries:
        assert not WRITE.search(strip_literals(cypher)), cypher


def test_provenance_rejects_malformed_or_unknown_relation(client):
    assert client.get("/provenance/e2|DROP_ALL|e1", headers=H(VIEWER)).status_code == 400
    assert client.get("/provenance/just-an-id", headers=H(VIEWER)).status_code == 400
    p = client.get("/provenance/e2|DEVELOPED_BY|e1", headers=H(VIEWER)).json()
    assert p["mentions"][0]["url"] == "u" and p["kg_version"] == "resolved"


# ------------------------------------------------------------------ monitoring
def test_metrics_count_requests_tokens_and_logs_are_json_without_keys(client):
    client.post("/ask", json={"question": "Who developed Claude?", "strategy": "A"}, headers=H(VIEWER))
    client.get("/analytics/top", headers={"X-API-Key": "wrong-key-wrong-key-wrong"})
    text = client.get("/metrics", headers=H(ADMIN)).text
    assert 'api_requests_total{method="POST",route="/ask",status="200"} 1' in text
    assert 'api_requests_total{method="GET",route="/analytics/top",status="401"} 1' in text
    assert 'api_llm_total{strategy="A",kind="tokens_in"} 900' in text
    assert 'api_request_seconds_count{route="/ask"} 1' in text
    lines = [json.loads(l) for l in client.log.read_text().splitlines()]
    assert {l["event"] for l in lines} >= {"request", "ask"}
    assert all(ADMIN not in json.dumps(l) and VIEWER not in json.dumps(l) for l in lines)
    ask = next(l for l in lines if l["event"] == "request" and l["route"] == "/ask")
    assert ask["role"] == "viewer" and ask["status"] == 200 and "latency_ms" in ask


def test_internal_errors_are_500_and_counted(client, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(M, "read", broken)
    assert client.get("/analytics/top", headers=H(VIEWER)).status_code == 500
    assert client.get("/health").json()["status"] == "degraded"
    assert 'api_errors_total{route="/analytics/top"} 1' in client.get("/metrics", headers=H(ADMIN)).text
