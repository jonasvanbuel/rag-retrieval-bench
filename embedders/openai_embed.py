"""OpenAI embedder.

Wraps text-embedding-3-small via the OpenAI SDK.

Output: 1536-dimensional float32 vectors (pre-normalised by the API).
Batches requests at BATCH_SIZE to stay within API limits.

Requires:
  OPENAI_API_KEY  in the environment (see .env.example).
"""
import os
import re
import numpy as np
from openai import OpenAI

from embedders import BaseEmbedder

MODEL_ID    = "text-embedding-3-small"
BATCH_SIZE  = 150    # academic chunks ~900 tokens avg; 150 × 900 ≈ 135K < 300K token/request limit
MAX_TOKENS  = 8000   # hard cap per input; API limit is 8192 but leave headroom

# Some academic papers discuss tokenisation and literally contain OpenAI special-token
# markers (e.g. <|endofprompt|>).  The API rejects these, so we strip them first.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]{1,40}\|>")

try:
    import tiktoken as _tiktoken
    _enc = _tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False


def _sanitise(text: str) -> str:
    """Remove special-token markers and truncate to MAX_TOKENS.

    Dense PDF tables (e.g. benchmark result matrices) can produce chunks with
    20 000+ tokens despite a low word count — tiktoken truncation is the only
    reliable guard against the 8 192-token per-input API limit.
    """
    text = _SPECIAL_TOKEN_RE.sub("", text)
    if not text.strip():
        text = " "
    if _HAS_TIKTOKEN:
        tokens = _enc.encode(text, disallowed_special=())
        if len(tokens) > MAX_TOKENS:
            text = _enc.decode(tokens[:MAX_TOKENS])
    else:
        # Fallback: ~4 chars per token
        text = text[:MAX_TOKENS * 4]
    return text


class OpenAIEmbedder(BaseEmbedder):
    name = "openai"

    def __init__(self):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = MODEL_ID

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        clean = [_sanitise(t) for t in texts]
        resp = self._client.embeddings.create(input=clean, model=self._model)
        return np.array([item.embedding for item in resp.data], dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
        return np.vstack([self._embed_batch(b) for b in batches])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_batch([text])[0]
