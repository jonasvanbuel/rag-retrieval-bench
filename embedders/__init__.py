"""Embedders package.

Shared interface for all embedding models:
  BaseEmbedder    — abstract base; subclasses implement embed() and embed_query()

All implementations must return L2-normalised float32 arrays so that
cosine similarity reduces to a plain dot product in the retrieval layer.
"""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    name: str = "base"

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts.  Returns shape (n, dim) float32 array."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string.  Returns shape (dim,) float32 array."""
        ...
