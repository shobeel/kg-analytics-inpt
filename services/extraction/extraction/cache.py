"""File cache of extraction results keyed by chunk hash (+ model + prompt version), so an
interrupted or quota-limited run resumes without repeating any call."""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional


def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ResultCache:
    def __init__(self, root: str, model: str, prompt_version: str):
        self.dir = Path(root) / model.replace("/", "_") / prompt_version
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[dict]:
        p = self._path(key)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, key: str, value: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False))
        os.replace(tmp, p)                      # atomic: never a half-written cache entry

    def all(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted(self.dir.glob("*/*.json"))]
