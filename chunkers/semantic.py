"""Semantic chunker.

Embeds every sentence with MiniLM, then scans adjacent-sentence cosine
similarities.  A chunk boundary is inserted where similarity drops below the
threshold (topic shift) or the word-count hard cap is hit.

Always uses MiniLM for the similarity scan, regardless of which embedding
model is used for the evaluation — this keeps chunk boundaries independent
of the axis being measured.
"""
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer

from chunkers import BaseChunker, Chunk
from chunkers.sentence import _ensure_nltk

THRESHOLD    = 0.45   # cosine similarity drop below this triggers a split
TARGET_WORDS = 512    # hard word-count cap; prevents runaway long chunks
EMBED_MODEL  = "all-MiniLM-L6-v2"


class SemanticChunker(BaseChunker):
    name = "semantic"

    def __init__(self, threshold: float = THRESHOLD, target_words: int = TARGET_WORDS):
        self.threshold = threshold
        self.target_words = target_words
        self._model: SentenceTransformer | None = None
        _ensure_nltk()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(EMBED_MODEL)
        return self._model

    def chunk(self, text: str, paper_id: str) -> list[Chunk]:
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return []

        embeddings = self.model.encode(
            sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Cosine similarity between each adjacent sentence pair (dot product, already normalised)
        sims = [float(np.dot(embeddings[i], embeddings[i + 1]))
                for i in range(len(embeddings) - 1)]

        chunks: list[Chunk] = []
        current: list[str] = []
        current_words = 0
        idx = 0

        for i, sent in enumerate(sentences):
            current.append(sent)
            current_words += len(sent.split())

            is_last = (i == len(sentences) - 1)
            topic_shift = (i < len(sims)) and (sims[i] < self.threshold)
            word_cap = current_words >= self.target_words

            if (topic_shift or word_cap) and not is_last:
                chunks.append(Chunk(
                    text=" ".join(current),
                    paper_id=paper_id,
                    chunk_index=idx,
                    metadata={"word_count": current_words},
                ))
                idx += 1
                current = []
                current_words = 0

        if current:
            chunks.append(Chunk(
                text=" ".join(current),
                paper_id=paper_id,
                chunk_index=idx,
                metadata={"word_count": current_words},
            ))

        return chunks
