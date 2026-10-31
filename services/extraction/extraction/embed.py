"""Local embeddings with fastembed (ONNX Runtime; no torch). The model is downloaded once into
the fastembed_cache volume."""

import os

import numpy as np


class Embedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name,
                                   cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"))

    def embed(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """L2-normalized float32 matrix (rows = texts), so dot product = cosine."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vecs = np.array(list(self.model.embed(texts, batch_size=batch_size)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)
