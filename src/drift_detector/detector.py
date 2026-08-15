"""Drift detection engine: per-turn distance scoring gated behind a sustained-trend rule."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .baseline import Baseline
from .embedding import EmbeddingProvider, l2_normalise


class PageHinkley:
    """Streaming change detector: alarms on sustained upward shift, forgives blips.

    Drift is a divergence, not a blip (V1 design decision): the alarm only fires
    when the PH statistic exceeds lambda AND the current value sits above the
    running mean for `sustain` consecutive turns. A one-turn spike that recovers
    resets the streak and is forgiven.
    """

    def __init__(self, delta: float = 0.005, lam: float = 0.1, burn_in: int = 3, sustain: int = 2):
        self.delta, self.lam, self.burn_in, self.sustain = delta, lam, burn_in, sustain
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.cum = 0.0
        self.cum_min = 0.0
        self.exceed_streak = 0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum += x - self.mean - self.delta
        self.cum_min = min(self.cum_min, self.cum)
        elevated = self.statistic > self.lam and x > self.mean
        self.exceed_streak = self.exceed_streak + 1 if elevated else 0
        return self.n > self.burn_in and self.exceed_streak >= self.sustain

    @property
    def statistic(self) -> float:
        return self.cum - self.cum_min


@dataclass
class TurnScore:
    turn: int
    cosine_distance: float
    euclidean_distance: float
    threshold_breach: bool
    trend_alarm: bool
    drifted: bool

    @property
    def badge(self) -> str:
        if self.drifted:
            return "drift detected"
        if self.threshold_breach:
            return "threshold breach"
        return "nominal"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["badge"] = self.badge
        return d


class DriftDetector:
    """Scores each response against the baseline centroid.

    - Embeds the response only (the PRD thesis: output quality is the signal).
    - L2-normalises, then computes cosine and Euclidean distance to the centroid.
    - Static threshold breach is reported per turn; the drift verdict is gated
      behind the Page-Hinkley sustained-trend rule so one-turn blips are forgiven.
    """

    def __init__(
        self,
        baseline: Baseline,
        provider: EmbeddingProvider,
        metric: str = "cosine",
        use_trend: bool = True,
        ph_delta: float = 0.005,
        ph_lambda: float = 0.1,
    ):
        self.baseline = baseline
        self.provider = provider
        self.metric = metric
        self.use_trend = use_trend
        self.ph = PageHinkley(delta=ph_delta, lam=ph_lambda)
        self.turn = 0
        self.history: list[TurnScore] = []

    def score(self, text: str) -> TurnScore:
        self.turn += 1
        v = l2_normalise(self.provider.embed([text]))[0]
        cos = float(1.0 - v @ self.baseline.centroid)
        euc = float(np.linalg.norm(v - self.baseline.centroid))
        if self.metric == "cosine":
            d, thr = cos, self.baseline.cosine_threshold
        else:
            d, thr = euc, self.baseline.euclidean_threshold
        breach = d > thr
        alarm = self.ph.update(d) if self.use_trend else False
        drifted = alarm if self.use_trend else breach
        ts = TurnScore(self.turn, round(cos, 4), round(euc, 4), breach, alarm, drifted)
        self.history.append(ts)
        return ts

    def summary(self) -> dict:
        key = "cosine_distance" if self.metric == "cosine" else "euclidean_distance"
        ds = [getattr(t, key) for t in self.history]
        drifted = [t for t in self.history if t.drifted]
        return {
            "turns": len(self.history),
            "drifted_turns": len(drifted),
            "drift_rate": round(len(drifted) / len(self.history), 3) if self.history else 0.0,
            "mean_distance": round(float(np.mean(ds)), 4) if ds else 0.0,
            "peak_distance": round(float(np.max(ds)), 4) if ds else 0.0,
            "metric": self.metric,
            "trend_rule": self.use_trend,
        }
