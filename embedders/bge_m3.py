"""BGE-M3 embedder.

Wraps BAAI/bge-m3 via sentence-transformers (dense-only output, 2.3 GB model).
Output: 1024-dimensional L2-normalised float32 vectors.

Runs on CPU to use BGE-M3's full 8192-token context window without truncation.
Apple Silicon MPS causes "Invalid buffer size: 64.00 GiB" for attention matrices
on sequences longer than ~800 tokens, so CPU is the honest option for this corpus.
Embedding is a one-time cost cached to disk — expect ~30–60 min on CPU.

batch_size=4 avoids RAM pressure on CPU with long academic chunks.
First load takes 30–60 s — the pre-cache stage in runner.py pays this once.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from embedders import BaseEmbedder

MODEL_ID = "BAAI/bge-m3"


class BGEM3Embedder(BaseEmbedder):
    name = "bge_m3"

    def __init__(self):
        print(f"Loading {MODEL_ID} on CPU (this may take a minute)...", end=" ", flush=True)
        # Force CPU: avoids the Apple Silicon MPS "Invalid buffer size: 64 GiB"
        # error that would otherwise require capping input at 512 tokens.
        # BGE-M3 supports up to 8192 tokens; we use its full context window.
        self._model = SentenceTransformer(MODEL_ID, device="cpu")
        self._model.max_seq_length = 8192
        print("ready")

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=4,
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
