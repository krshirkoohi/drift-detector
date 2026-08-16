"""Drift detection engine: per-turn distance scoring gated behind a sustained-trend rule."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from typing import Optional
import time

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
    compacted_reset: bool = False
    notice: Optional[str] = None

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


# DriftResult is the typed API contract alias for TurnScore
DriftResult = TurnScore


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
        self.initial_baseline: Baseline = baseline
        self.baseline: Baseline = baseline
        self.provider = provider
        self.metric = metric
        self.use_trend = use_trend
        self.ph_delta = ph_delta
        self.ph_lambda = ph_lambda
        self.ph = PageHinkley(delta=ph_delta, lam=ph_lambda)
        self.turn = 0
        self.history: list[TurnScore] = []
        self.prev_history_len: Optional[int] = None
        self.prev_prompt_tokens: Optional[int] = None
        self.has_drifted: bool = False

    @classmethod
    def from_examples(
        cls,
        baseline_texts: list[str],
        provider: EmbeddingProvider,
        metric: str = "cosine",
        use_trend: bool = True,
        threshold: Optional[float] = None,
        ph_delta: float = 0.005,
        ph_lambda: float = 0.1,
    ) -> DriftDetector:
        """Initialize the DriftDetector directly from a list of baseline texts."""
        from .baseline import BaselineStore
        store = BaselineStore(provider)
        baseline = store.build(baseline_texts)
        if threshold is not None:
            if metric == "cosine":
                baseline.cosine_threshold = threshold
            else:
                baseline.euclidean_threshold = threshold
        return cls(
            baseline=baseline,
            provider=provider,
            metric=metric,
            use_trend=use_trend,
            ph_delta=ph_delta,
            ph_lambda=ph_lambda,
        )

    def reset_page_hinkley(self) -> None:
        """Reset Page-Hinkley streaming accumulators, mean, and latch state."""
        self.ph.reset()
        self.has_drifted = False

    def reset(self) -> None:
        """Reset turn count, history, and streaming detector state (preserving baseline anchor)."""
        self.reset_page_hinkley()
        self.turn = 0
        self.history = []
        self.prev_history_len = None
        self.prev_prompt_tokens = None

    def rebase(
        self,
        new_anchor: Baseline | list[str] | str,
        reason: str = "explicit_task_transition",
    ) -> str:
        """Explicitly rebase the semantic mission anchor.

        This should ONLY be called on deliberate task changes, user scope transitions,
        or explicitly validated multi-sample re-baselining events.
        """
        from .baseline import BaselineStore
        self.reset_page_hinkley()
        self.turn = 0

        if isinstance(new_anchor, Baseline):
            self.baseline = new_anchor
        elif isinstance(new_anchor, str):
            vecs = l2_normalise(self.provider.embed([new_anchor.strip()]))
            centroid = vecs[0]
            self.baseline = Baseline(
                centroid=centroid,
                cosine_threshold=max(self.baseline.cosine_threshold, 0.45),
                euclidean_threshold=max(self.baseline.euclidean_threshold, 0.65),
                n_samples=1,
            )
        elif isinstance(new_anchor, (list, tuple)):
            if len(new_anchor) < 3:
                vecs = l2_normalise(self.provider.embed(list(new_anchor)))
                centroid = l2_normalise(np.mean(vecs, axis=0, keepdims=True))[0]
                self.baseline = Baseline(
                    centroid=centroid,
                    cosine_threshold=max(self.baseline.cosine_threshold, 0.45),
                    euclidean_threshold=max(self.baseline.euclidean_threshold, 0.65),
                    n_samples=len(new_anchor),
                )
            else:
                store = BaselineStore(self.provider)
                self.baseline = store.build(list(new_anchor))
        else:
            raise TypeError(f"Unsupported anchor type: {type(new_anchor)}")

        return f"[driftd] Semantic mission anchor rebased ({reason})"

    def handle_compaction(
        self,
        compacted_summary: Optional[str] = None,
        rebase_anchor: bool = False,
    ) -> str:
        """Reset transient Page-Hinkley accumulators after chat compaction while preserving the mission anchor by default.

        Per V1 Operating Brief:
        Compaction resets transient trend accumulators and history counters, but preserves
        the original task/mission anchor centroid by default so gradual drift cannot silently
        renormalise against recent drifted summaries.
        """
        self.reset_page_hinkley()
        self.turn = 0

        if rebase_anchor and compacted_summary and compacted_summary.strip():
            self.rebase(compacted_summary.strip(), reason="compaction_rebase_requested")
            return "[driftd] Chat compacted: accumulators reset · re-baselined on compacted summary"

        return "[driftd] Chat compacted: accumulators reset (mission anchor preserved)"

    def score(
        self,
        text: str,
        history_len: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        is_compacted: bool = False,
        compacted_summary: Optional[str] = None,
    ) -> TurnScore:
        compacted_reset = False
        notice = None

        # 1. Explicit compaction signal
        if is_compacted or (compacted_summary is not None and compacted_summary.strip()):
            self.handle_compaction(compacted_summary=compacted_summary)
            compacted_reset = True
            notice = "[driftd] Chat compacted: detector reset"

        # 2. History truncation detection: len(history) < prev_len
        elif history_len is not None and self.prev_history_len is not None and history_len < self.prev_history_len:
            self.handle_compaction(compacted_summary=compacted_summary)
            compacted_reset = True
            notice = f"[driftd] Chat compacted (history truncated {self.prev_history_len} -> {history_len}): detector reset"

        # 3. Token drop detection: abrupt drop in prompt tokens across turns
        elif (
            prompt_tokens is not None
            and self.prev_prompt_tokens is not None
            and self.prev_prompt_tokens > 200
            and prompt_tokens < int(self.prev_prompt_tokens * 0.6)
        ):
            self.handle_compaction(compacted_summary=compacted_summary)
            compacted_reset = True
            notice = f"[driftd] Chat compacted (tokens dropped {self.prev_prompt_tokens} -> {prompt_tokens}): detector reset"

        # Track lengths for next turn
        if history_len is not None:
            self.prev_history_len = history_len
        if prompt_tokens is not None:
            self.prev_prompt_tokens = prompt_tokens

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
        if drifted:
            self.has_drifted = True
        ts = TurnScore(
            turn=self.turn,
            cosine_distance=round(cos, 4),
            euclidean_distance=round(euc, 4),
            threshold_breach=breach,
            trend_alarm=alarm,
            drifted=drifted,
            compacted_reset=compacted_reset,
            notice=notice,
        )
        self.history.append(ts)
        return ts

    def check_response(
        self,
        response_text: str,
        history_len: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        is_compacted: bool = False,
        compacted_summary: Optional[str] = None,
    ) -> dict:
        """Unified check_response interface returning full dictionary metrics."""
        t0 = time.time()
        score = self.score(
            response_text,
            history_len=history_len,
            prompt_tokens=prompt_tokens,
            is_compacted=is_compacted,
            compacted_summary=compacted_summary,
        )
        latency = (time.time() - t0) * 1000
        return {
            "turn_index": score.turn,
            "cosine_distance": score.cosine_distance,
            "euclidean_distance": score.euclidean_distance,
            "threshold": self.baseline.cosine_threshold if self.metric == "cosine" else self.baseline.euclidean_threshold,
            "metric": self.metric,
            "is_drifting": score.drifted,
            "drift_detected": score.drifted,
            "threshold_breach": score.threshold_breach,
            "trend_alarm": score.trend_alarm,
            "compacted_reset": score.compacted_reset,
            "notice": score.notice,
            "latency_ms": round(latency, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ph_running_mean": round(self.ph.mean, 4) if self.use_trend else None,
            "ph_running_sum": round(self.ph.cum, 4) if self.use_trend else None,
            "ph_min_sum": round(self.ph.cum_min, 4) if self.use_trend else None,
            "ph_statistic": round(self.ph.statistic, 4) if self.use_trend else None,
            "ph_threshold": self.ph_lambda if self.use_trend else None,
            "ph_delta": self.ph_delta if self.use_trend else None,
        }

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
            "has_drifted": self.has_drifted,
        }

