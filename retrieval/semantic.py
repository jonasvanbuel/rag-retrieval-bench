"""Semantic (vector) retriever.

Cosine similarity via numpy matrix multiply.  Works because all embeddings
are L2-normalised: cosine(a, b) == dot(a, b) when ||a||=||b||=1.

The embedding matrix is passed in at construction time and lives in memory
for the duration of the benchmark run — no disk I/O at query time.
"""
import numpy as np

from chunkers import Chunk
from embedders import BaseEmbedder
from retrieval import BaseRetriever, RetrievedChunk


class SemanticRetriever(BaseRetriever):
    name = "semantic"

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray,
                 embedder: BaseEmbedder):
        # embeddings: shape (n_chunks, dim), L2-normalised float32
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        q_vec = self.embedder.embed_query(query)          # shape (dim,)
        scores = self.embeddings @ q_vec                  # shape (n_chunks,) — cosine sim
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]), rank=r + 1)
            for r, i in enumerate(top_indices)
        ]
