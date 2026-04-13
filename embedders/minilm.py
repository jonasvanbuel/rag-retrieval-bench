"""MiniLM embedder.

Wraps sentence-transformers all-MiniLM-L6-v2 (80 MB, local, no API key).
Output: 384-dimensional L2-normalised float32 vectors.

Fast enough for the full corpus chunked at 512 words on CPU.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from embedders import BaseEmbedder

MODEL_ID = "all-MiniLM-L6-v2"


class MiniLMEmbedder(BaseEmbedder):
    name = "minilm"

    def __init__(self):
        self._model = SentenceTransformer(MODEL_ID)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=64,
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
