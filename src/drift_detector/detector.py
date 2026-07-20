"""
detector.py — Legacy compatibility wrapper for the drift detection engine.

Provides DriftDetector and TurnScore, which delegate to the new DriftSession API under the hood.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import numpy as np

from .baseline import BaselineStore
from .embeddings import EmbeddingAdapter
from .session import DriftSession

@dataclass
class TurnScore:
    """Legacy container for a single turn's metrics."""
    turn: int
    cosine_distance: float
    euclidean_distance: float
    threshold_breach: bool
    trend_alarm: bool
    drifted: bool

    def to_dict(self) -> dict:
        """Convert the TurnScore to a dictionary."""
        return asdict(self)


class DriftDetector:
    """
    Legacy wrapper that provides backward compatibility for DriftDetector.
    
    Delegates all operations to the new DriftSession class.
    """
    def __init__(
        self,
        baseline_store: BaselineStore,
        api_key: Optional[str] = None,
        threshold: Optional[float] = None,
        metric: str = "cosine",
        log_dir: Optional[str] = None,
        use_trend: bool = False,
        embedding_adapter: Optional[EmbeddingAdapter] = None,
        ph_sustain: int = 1,
        ph_burn_in: int = 0
    ):
        self.baseline_store = baseline_store
        self.metric = metric.lower()
        self.log_dir = log_dir
        self.use_trend = use_trend
        
        # Initialise adapter
        if embedding_adapter is None:
            if not api_key:
                raise ValueError("Either api_key or embedding_adapter must be provided.")
            from .embeddings import GeminiEmbeddingAdapter
            self.embedding_adapter = GeminiEmbeddingAdapter(api_key)
        else:
            self.embedding_adapter = embedding_adapter

        # Ensure baseline store has computed centroid
        if self.baseline_store.centroid is None:
            self.baseline_store.compute_centroid(adapter=self.embedding_adapter)
            
        # Standard calibration
        self._session = DriftSession.initialise(
            known_good_responses=self.baseline_store.examples,
            embedding_adapter=self.embedding_adapter,
            name=self.baseline_store.name,
            metric=self.metric,
            threshold=threshold,
            use_trend=self.use_trend,
            ph_sustain=ph_sustain,
            ph_burn_in=ph_burn_in,
            log_dir=self.log_dir
        )
        
        # Sync attributes for backwards compatibility
        self.threshold = self._session.threshold
        self.ph_delta = self._session.ph_delta
        self.ph_threshold = self._session.ph_threshold
        
    @property
    def has_drifted(self) -> bool:
        """Return whether drift has been detected in this session."""
        return self._session.has_drifted
        
    @property
    def ph_n(self) -> int:
        """Get the current turn index (Page-Hinkley n)."""
        return self._session.ph_n
        
    @ph_n.setter
    def ph_n(self, value: int) -> None:
        self._session.ph_n = value

    @property
    def ph_running_mean(self) -> float:
        """Get Page-Hinkley running mean distance."""
        return self._session.ph_running_mean
        
    @ph_running_mean.setter
    def ph_running_mean(self, value: float) -> None:
        self._session.ph_running_mean = value

    @property
    def ph_running_sum(self) -> float:
        """Get Page-Hinkley running sum."""
        return self._session.ph_running_sum
        
    @ph_running_sum.setter
    def ph_running_sum(self, value: float) -> None:
        self._session.ph_running_sum = value

    @property
    def ph_min_sum(self) -> float:
        """Get Page-Hinkley running minimum sum."""
        return self._session.ph_min_sum
        
    @ph_min_sum.setter
    def ph_min_sum(self, value: float) -> None:
        self._session.ph_min_sum = value
        
    @property
    def history(self) -> List[TurnScore]:
        """Convert new session history back to legacy TurnScore list for compatibility."""
        legacy_history = []
        for v in self._session.history:
            threshold_breach = v.distance > v.threshold
            legacy_history.append(TurnScore(
                turn=v.turn_index,
                cosine_distance=v.cosine_distance,
                euclidean_distance=v.euclidean_distance,
                threshold_breach=threshold_breach,
                trend_alarm=v.trend_alarm if v.trend_alarm is not None else False,
                drifted=v.drift_detected
            ))
        return legacy_history

    def check_response(self, response_text: str) -> Dict[str, Any]:
        """Analyse a single response against the baseline (legacy API)."""
        v = self._session.observe(response_text)
        
        result = {
            "timestamp": time_format_now(),
            "response_snippet": response_text[:100] + ("..." if len(response_text) > 100 else ""),
            "metric": self.metric,
            "cosine_distance": v.cosine_distance,
            "euclidean_distance": v.euclidean_distance,
            "threshold": v.threshold,
            "is_drifting": v.drift_detected,
            "latency_ms": v.latency_ms
        }
        
        if self.use_trend:
            result.update({
                "ph_running_mean": v.ph_running_mean,
                "ph_running_sum": v.ph_running_sum,
                "ph_min_sum": v.ph_min_sum,
                "ph_statistic": v.trend_statistic,
                "ph_threshold": v.ph_threshold,
                "ph_delta": v.ph_delta,
                "trend_alarm": v.trend_alarm
            })
            
        return result

    def score(self, text: str) -> TurnScore:
        """Score a response text, returning a TurnScore (legacy API)."""
        v = self._session.observe(text)
        threshold_breach = v.distance > v.threshold
        return TurnScore(
            turn=v.turn_index,
            cosine_distance=v.cosine_distance,
            euclidean_distance=v.euclidean_distance,
            threshold_breach=threshold_breach,
            trend_alarm=v.trend_alarm if v.trend_alarm is not None else False,
            drifted=v.drift_detected
        )

    def summary(self) -> dict:
        """Return summary statistics of the session history (legacy API)."""
        return self._session.summary()


def time_format_now() -> str:
    """Helper to return current time formatted as UTC ISO-8601 string."""
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
