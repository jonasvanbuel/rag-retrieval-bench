"""Retrieval package.

Shared types:
  RetrievedChunk  — a retrieved chunk with its score and rank
  BaseRetriever   — abstract base; subclasses implement retrieve()
"""
from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod

from chunkers import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float   # higher = more relevant (cosine sim, BM25 score, or RRF score)
    rank: int      # 1-based rank in the returned list


class BaseRetriever(ABC):
    name: str = "base"

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        ...
