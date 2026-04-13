"""Re-ranking retriever.

Wraps a HybridRetriever: retrieves a larger candidate pool, then re-scores
the top candidates with a cross-encoder and returns the reranked top_k.

Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2 (~90 MB, local).
Downloaded automatically by sentence-transformers on first use.

The model is lazy-loaded so configs that don't use re-ranking never pay
the loading cost.
"""
from sentence_transformers import CrossEncoder

from retrieval import BaseRetriever, RetrievedChunk
from retrieval.hybrid import HybridRetriever

RERANKER_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES  = 20   # re-rank this many candidates from hybrid before slicing to top_k


class RerankerRetriever(BaseRetriever):
    name = "reranker"

    def __init__(self, hybrid: HybridRetriever,
                 rerank_candidates: int = RERANK_CANDIDATES):
        self.hybrid = hybrid
        self.rerank_candidates = rerank_candidates
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(RERANKER_MODEL)
        return self._model

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        candidates = self.hybrid.retrieve(query, top_k=self.rerank_candidates)
        pairs = [(query, rc.chunk.text) for rc in candidates]
        ce_scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(ce_scores, candidates), key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(chunk=rc.chunk, score=float(score), rank=r + 1)
            for r, (score, rc) in enumerate(ranked[:top_k])
        ]
