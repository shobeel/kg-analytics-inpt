"""Thin HTTP client for the Phase 7 API (viewer key from UI_API_KEY; the key is never shown)."""

import os

import requests

API = os.environ.get("API_URL", "http://api:8000").rstrip("/")
TIMEOUT = 120


def _headers() -> dict:
    return {"X-API-Key": os.environ.get("UI_API_KEY", "")}


def get(path: str, **params):
    r = requests.get(f"{API}{path}", params=params, headers=_headers(), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def ask(question: str, strategy: str, kg_version: str) -> dict:
    r = requests.post(f"{API}/ask", json={"question": question, "strategy": strategy, "kg_version": kg_version},
                      headers=_headers(), timeout=TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {r.json().get('detail', r.text)[:300]}")
    return r.json()


def error_text(e: Exception) -> str:
    if isinstance(e, requests.HTTPError) and e.response is not None:
        try:
            return f"{e.response.status_code}: {e.response.json().get('detail')}"
        except ValueError:
            return f"{e.response.status_code}: {e.response.text[:200]}"
    return str(e)[:300]
