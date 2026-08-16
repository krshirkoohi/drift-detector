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
    mean_cosine_distance: float = 0.0
    mean_euclidean_distance: float = 0.0


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

    def build(
        self,
        texts: list[str],
        percentile: float = 95.0,
    ) -> Baseline:
        if len(texts) < 3:
            raise ValueError("Need at least 3 baseline samples")
        emb = l2_normalise(self.provider.embed(texts))
        centroid = l2_normalise(emb.mean(axis=0, keepdims=True))[0]
        cos = 1.0 - emb @ centroid
        euc = np.linalg.norm(emb - centroid, axis=1)

        mu_cos = float(np.mean(cos))
        mu_euc = float(np.mean(euc))
        raw_cos = float(np.percentile(cos, percentile))
        raw_euc = float(np.percentile(euc, percentile))

        # Apply small-sample safety floors to prevent overfitting when sample count N < 10
        is_neural = getattr(self.provider, "model_name", None) is not None or getattr(self.provider, "model", None) is not None
        if getattr(self.provider, "model_name", None) is not None:
            eff_floor_cos, eff_floor_euc = 0.85, 1.20
        elif is_neural:
            eff_floor_cos, eff_floor_euc = 0.70, 1.00
        else:
            eff_floor_cos, eff_floor_euc = 0.45, 0.65

        if len(texts) < 10:
            cos_threshold = max(raw_cos, eff_floor_cos)
            euc_threshold = max(raw_euc, eff_floor_euc)
        else:
            cos_threshold = raw_cos
            euc_threshold = raw_euc

        return Baseline(
            centroid=centroid,
            cosine_threshold=cos_threshold,
            euclidean_threshold=euc_threshold,
            n_samples=len(texts),
            mean_cosine_distance=mu_cos,
            mean_euclidean_distance=mu_euc,
        )

    def build_from_file(self, path: str | Path, percentile: float = 95.0) -> Baseline:
        return self.build(self.load_texts(path), percentile)
