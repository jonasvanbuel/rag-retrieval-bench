"""Hybrid retriever — Reciprocal Rank Fusion (RRF).

Combines ranked lists from a SemanticRetriever and a BM25Retriever using RRF:

  score(doc) = Σ_i  1 / (k + rank_i(doc))

where k=60 is the standard smoothing constant (Cormack et al.).

Chunk identity is tracked via (paper_id, chunk_index) tuples so lookup is O(1)
rather than list.index() which is O(n).
"""
from collections import defaultdict

from chunkers import Chunk
from retrieval import BaseRetriever, RetrievedChunk
from retrieval.semantic import SemanticRetriever
from retrieval.bm25 import BM25Retriever

RRF_K = 60


class HybridRetriever(BaseRetriever):
    name = "hybrid"

    def __init__(self, semantic: SemanticRetriever, bm25: BM25Retriever,
                 rrf_k: int = RRF_K):
        self.semantic = semantic
        self.bm25 = bm25
        self.rrf_k = rrf_k
        # Build O(1) lookup: (paper_id, chunk_index) → Chunk object
        self._chunk_map: dict[tuple[str, int], Chunk] = {
            (c.paper_id, c.chunk_index): c
            for c in semantic.chunks
        }

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        candidate_k = max(top_k * 4, 20)
        sem_results  = self.semantic.retrieve(query, top_k=candidate_k)
        bm25_results = self.bm25.retrieve(query,     top_k=candidate_k)

        rrf_scores: dict[tuple[str, int], float] = defaultdict(float)

        for rank, rc in enumerate(sem_results, start=1):
            key = (rc.chunk.paper_id, rc.chunk.chunk_index)
            rrf_scores[key] += 1.0 / (self.rrf_k + rank)

        for rank, rc in enumerate(bm25_results, start=1):
            key = (rc.chunk.paper_id, rc.chunk.chunk_index)
            rrf_scores[key] += 1.0 / (self.rrf_k + rank)

        top = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                chunk=self._chunk_map[key],
                score=score,
                rank=r + 1,
            )
            for r, (key, score) in enumerate(top)
        ]
