"""Sentence-boundary chunker.

Groups NLTK sentences into ~512-word chunks.  No sentence is split mid-way;
sentences accumulate until the word-count target is reached, then a new chunk
starts.  No overlap — sentence integrity is the design tradeoff here.
"""
import nltk

from chunkers import BaseChunker, Chunk

TARGET_WORDS = 512


def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


class SentenceChunker(BaseChunker):
    name = "sentence"

    def __init__(self, target_words: int = TARGET_WORDS):
        self.target_words = target_words
        _ensure_nltk()

    def chunk(self, text: str, paper_id: str) -> list[Chunk]:
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current: list[str] = []
        current_words = 0
        idx = 0

        for sent in sentences:
            sent_words = len(sent.split())
            if current_words + sent_words > self.target_words and current:
                chunks.append(Chunk(
                    text=" ".join(current),
                    paper_id=paper_id,
                    chunk_index=idx,
                    metadata={"word_count": current_words, "sentence_count": len(current)},
                ))
                idx += 1
                current = []
                current_words = 0
            current.append(sent)
            current_words += sent_words

        if current:
            chunks.append(Chunk(
                text=" ".join(current),
                paper_id=paper_id,
                chunk_index=idx,
                metadata={"word_count": current_words, "sentence_count": len(current)},
            ))

        return chunks
