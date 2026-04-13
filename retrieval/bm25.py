"""BM25 retriever.

Wraps rank-bm25's BM25Okapi.  The same _tokenize() function is applied at
both index time and query time to guarantee vocabulary consistency.
"""
import re
from rank_bm25 import BM25Okapi

from chunkers import Chunk
from retrieval import BaseRetriever, RetrievedChunk


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self.index = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        tokens = _tokenize(query)
        scores = self.index.get_scores(tokens)  # shape (n_chunks,)
        top_k = min(top_k, len(self.chunks))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]), rank=r + 1)
            for r, i in enumerate(ranked[:top_k])
        ]
