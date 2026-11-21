"""Embedding providers.

`HashingEmbedder` is a deterministic, offline embedder (signed feature hashing over
word unigrams and bigrams). It needs no API key, which keeps tests and CI evals
reproducible. `OpenAIEmbedder` is used when OPENAI_API_KEY is set.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

TOKEN = re.compile(r"[a-z0-9_]+")
STOPWORDS = frozenset(
    "a an the and or of to in on for is are be with by how do does what which i you it this that can my your".split()
)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def stem(token: str) -> str:
    """Very small suffix stripper so 'duplicates'/'duplicate' and 'retried'/'retry' share features."""
    if len(token) <= 4:
        return token
    if token.endswith(("ies", "ied")):
        return token[:-3] + "y"
    if token.endswith(("sses", "shes", "ches", "xes", "zes")):
        return token[:-2]
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class HashingEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for feat in features:
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class OpenAIEmbedder:
    """Embeddings from any OpenAI-compatible endpoint (OpenAI, Gemini, Mistral, Ollama, ...)."""

    def __init__(
        self,
        model: str,
        dim: int,
        api_key: str | None = None,
        base_url: str | None = None,
        send_dimensions: bool = True,
    ) -> None:
        from langchain_openai import OpenAIEmbeddings

        self.dim = dim
        kwargs: dict = {"model": model}
        if send_dimensions:  # otherwise EMBEDDING_DIM must equal the model's native size
            kwargs["dimensions"] = dim
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
            # Non-OpenAI servers expect strings, not the tiktoken token arrays langchain sends by default.
            kwargs["check_embedding_ctx_length"] = False
        self._client = OpenAIEmbeddings(**kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
