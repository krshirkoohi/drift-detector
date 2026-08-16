"""Embedding providers for the Drift Detector.

Providers turn text into fixed-size vectors. The detector only ever sees
vectors, so swapping providers never touches detection logic.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from typing import Protocol, Sequence

import numpy as np


class EmbeddingProvider(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


# Function words carry no topical signal; without a real semantic model they
# dominate the token sum and drown out content words, so the test provider
# filters them.
_STOPWORDS = frozenset(
    "a an and are as at be but by for from had has have he her his i if in into is it its of on or "
    "our she so that the their them they this to was we were while will with you your".split()
)


def l2_normalise(v: np.ndarray) -> np.ndarray:
    """Rescale vectors to unit length so only direction (meaning) matters."""
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


class DeterministicProvider:
    """Offline, dependency-free provider for tests, CI, and demos.

    Each token is hashed to a stable pseudo-random unit vector; a text's
    embedding is the normalised sum of its token vectors, so texts sharing
    vocabulary land near each other. Not a semantic model: use it to test
    plumbing and thresholds, not real detection quality.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self._cache: dict[str, np.ndarray] = {}

    def _token_vec(self, token: str) -> np.ndarray:
        if token not in self._cache:
            seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            self._cache[token] = rng.standard_normal(self.dim)
        return self._cache[token]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, text in enumerate(texts):
            for t in re.findall(r"[a-z0-9']+", text.lower()):
                if t not in _STOPWORDS:
                    out[i] += self._token_vec(t)
        return l2_normalise(out)


class GeminiProvider:
    """Gemini embedding API provider (network required)."""

    def __init__(self, model: str = "gemini-embedding-001", api_key: str | None = None, dim: int = 768):
        self.model = model
        self.dim = dim
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:batchEmbedContents?key={self.api_key}"
        )
        body = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": self.dim,
                }
                for t in texts
            ]
        }
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        vecs = np.array([e["values"] for e in data["embeddings"]])
        return l2_normalise(vecs)


class OpenAICompatibleProvider:
    """Any OpenAI-compatible /v1/embeddings endpoint (OpenAI, Ollama, LM Studio, vLLM)."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dim: int = 1536,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": list(texts)}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        vecs = np.array([d["embedding"] for d in data["data"]])
        return l2_normalise(vecs)


class LocalTransformerProvider:
    """Offline real neural transformer embedding provider (e.g. all-MiniLM-L6-v2, roberta-base)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", dim: int | None = None):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._dim = dim

    def _lazy_init(self) -> None:
        if self.tokenizer is None or self.model is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
            except ImportError as err:
                raise RuntimeError(
                    "LocalTransformerProvider requires 'torch' and 'transformers' packages. "
                    f"Please install them or use provider='test'/'deterministic' for offline hash testing. Error: {err}"
                ) from err
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.eval()
                self._dim = self.model.config.hidden_size
            except Exception as err:
                raise RuntimeError(
                    f"Failed to load local embedding model '{self.model_name}': {err}. "
                    "Ensure model weights are cached or network access is available."
                ) from err

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._lazy_init()
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self._lazy_init()
        import torch

        if not texts:
            return np.zeros((0, self.dim))

        inputs = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs[0]
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()
            return l2_normalise(mean_pooled)


def get_provider(name: str, **kwargs) -> EmbeddingProvider:
    name = name.lower()
    if name in ("test", "deterministic", "hash"):
        return DeterministicProvider(**kwargs)
    if name in ("local", "transformer", "transformers", "hf", "roberta", "minilm", "sbert", "local_transformer", "local-transformer"):
        return LocalTransformerProvider(**kwargs)
    if name == "gemini":
        return GeminiProvider(**kwargs)
    if name in ("openai", "openai-compatible", "ollama"):
        return OpenAICompatibleProvider(**kwargs)
    raise ValueError(f"Unknown provider: '{name}'. Valid providers: 'local' (real neural model), 'gemini', 'openai', or 'test'/'deterministic' (offline hash testing).")

