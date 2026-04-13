"""Fixed-size chunker.

Splits text by word count with 50 % overlap.  Simple and deterministic —
the baseline every other chunker is measured against.

  chunk_size = 512 words
  overlap    = 256 words  (50 %)
  step       = 256 words
"""
from chunkers import BaseChunker, Chunk

CHUNK_SIZE = 512   # words
OVERLAP    = 256   # 50 % overlap → step = 256 words


class FixedSizeChunker(BaseChunker):
    name = "fixed_size"

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, paper_id: str) -> list[Chunk]:
        words = text.split()
        if not words:
            return []

        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        idx = 0
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunks.append(Chunk(
                text=" ".join(chunk_words),
                paper_id=paper_id,
                chunk_index=idx,
                metadata={"word_count": len(chunk_words)},
            ))
            idx += 1
            start += step

        return chunks
