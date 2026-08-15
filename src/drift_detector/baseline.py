"""Baseline store: loads reference samples, computes centroid and auto thresholds."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embedding import EmbeddingProvider, l2_normalise


@dataclass
class Baseline:
    centroid: np.ndarray
    cosine_threshold: float
    euclidean_threshold: float
    n_samples: int


class BaselineStore:
    """Loads baseline texts, embeds them, computes the centroid and p95 thresholds."""

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    @staticmethod
    def load_texts(path: str | Path) -> list[str]:
        data = json.loads(Path(path).read_text())
        if isinstance(data, dict):
            data = data.get("samples", [])
        return [s["text"] if isinstance(s, dict) else s for s in data]

    def build(self, texts: list[str], percentile: float = 95.0) -> Baseline:
        if len(texts) < 3:
            raise ValueError("Need at least 3 baseline samples")
        emb = l2_normalise(self.provider.embed(texts))
        centroid = l2_normalise(emb.mean(axis=0, keepdims=True))[0]
        cos = 1.0 - emb @ centroid
        euc = np.linalg.norm(emb - centroid, axis=1)
        return Baseline(
            centroid=centroid,
            cosine_threshold=float(np.percentile(cos, percentile)),
            euclidean_threshold=float(np.percentile(euc, percentile)),
            n_samples=len(texts),
        )

    def build_from_file(self, path: str | Path, percentile: float = 95.0) -> Baseline:
        return self.build(self.load_texts(path), percentile)
