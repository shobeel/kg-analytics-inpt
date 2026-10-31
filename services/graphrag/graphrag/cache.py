"""LLM result cache keyed by (strategy, question, kg_version, prompt_version, step): re-runs cost
no calls. The stored usage (tokens, latency) is reported again on a hit, so comparisons stay stable."""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional


def cache_key(strategy: str, question: str, kg_version: str, prompt_version: str, step: str) -> str:
    raw = json.dumps([strategy, " ".join(question.split()), kg_version, prompt_version, step])
    return hashlib.sha256(raw.encode()).hexdigest()


class LLMCache:
    def __init__(self, root: str):
        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[dict]:
        p = self.dir / f"{key}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, key: str, value: dict) -> None:
        p = self.dir / f"{key}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False))
        os.replace(tmp, p)
