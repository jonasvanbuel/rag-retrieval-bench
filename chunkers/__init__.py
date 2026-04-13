"""Chunkers package.

Shared types used across the entire pipeline:
  Chunk           — a single piece of text with provenance
  BaseChunker     — abstract base; subclasses implement chunk()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any


@dataclass
class Chunk:
    text: str
    paper_id: str        # ArXiv ID, e.g. "2005.11401"
    chunk_index: int     # 0-based position within this paper
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata carries optional diagnostics: word_count, sentence_count, etc.


class BaseChunker(ABC):
    name: str = "base"

    @abstractmethod
    def chunk(self, text: str, paper_id: str) -> list[Chunk]:
        """Split one paper's text into chunks."""
        ...

    def chunk_all(self, papers: dict[str, str]) -> list[Chunk]:
        """Chunk a {paper_id: text} mapping; returns a flat list of all chunks."""
        result: list[Chunk] = []
        for paper_id, text in papers.items():
            result.extend(self.chunk(text, paper_id))
        return result
