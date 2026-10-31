"""fastembed (ONNX Runtime, no torch) chunk embeddings, same model as Phase 2's entity embeddings.

all-MiniLM-L6-v2 reads at most 256 word pieces (~190 words), and a chunk has up to ~600 words,
so a chunk is embedded as the normalized mean of its 180-word windows, each prefixed with the
document title ("contextual chunk header": on the smoke query it moved 1/5 -> 4/5 hits containing
Google-related terms). Cached by hash of the embedded text, so a re-load computes nothing."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

WINDOW_WORDS = 180


def windows(text: str, size: int = WINDOW_WORDS) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), size)] or [""]


def mean_pool(vectors: np.ndarray) -> np.ndarray:
    v = vectors.mean(axis=0)
    return v / max(float(np.linalg.norm(v)), 1e-12)


class ChunkEmbedder:
    def __init__(self, model_name: str, cache_dir: str, embed_fn=None):
        self.model_name = model_name
        self.cache_path = Path(cache_dir) / model_name.replace("/", "_") / "chunks.json"
        self._embed_fn = embed_fn                      # injectable for tests
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.computed = 0

    def _fn(self):
        if self._embed_fn is None:
            from fastembed import TextEmbedding
            model = TextEmbedding(model_name=self.model_name, cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"),
                                  threads=2)
            self._embed_fn = lambda texts: np.array(list(model.embed(texts, batch_size=32)), dtype=np.float32)
        return self._embed_fn

    def release(self) -> None:
        """Free the ONNX model (lite stack: the loader shares 1.5 GB with a Spark driver)."""
        import gc
        self._embed_fn = None
        gc.collect()

    def embed_query(self, text: str) -> list[float]:
        v = self._fn()([text])[0]
        return (v / max(float(np.linalg.norm(v)), 1e-12)).astype(float).tolist()

    def embed_chunks(self, texts: list[str], titles: list[str] = None) -> list[list[float]]:
        titles = titles or [""] * len(texts)
        keys = [hashlib.sha256(f"{ti}\n{t}".encode()).hexdigest() for ti, t in zip(titles, texts)]
        todo = [(k, ti, t) for k, ti, t in zip(keys, titles, texts) if k not in self.cache]
        if todo:
            spans, flat = [], []
            for _, ti, t in todo:
                w = [f"{ti}. {x}" if ti else x for x in windows(t)]
                spans.append((len(flat), len(flat) + len(w)))
                flat += w
            vecs = self._fn()(flat)
            for (k, _, _), (a, b) in zip(todo, spans):
                self.cache[k] = [round(float(x), 7) for x in mean_pool(vecs[a:b])]
            self.computed += len(todo)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.cache))
            os.replace(tmp, self.cache_path)
        return [self.cache[k] for k in keys]
