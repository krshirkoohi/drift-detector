"""
models.py — Data models and structures for drift detection.
"""
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

@dataclass
class DriftVerdict:
    """
    The verdict and metrics returned after scoring a response for semantic drift.
    """
    distance: float
    trend_statistic: float
    drift_detected: bool
    recommend_fresh_chat: bool
    
    # Diagnostic fields for debugging and visual dashboards
    turn_index: int
    threshold: float
    metric: str
    cosine_distance: float
    euclidean_distance: float
    ph_running_mean: Optional[float] = None
    ph_running_sum: Optional[float] = None
    ph_min_sum: Optional[float] = None
    ph_threshold: Optional[float] = None
    ph_delta: Optional[float] = None
    trend_alarm: Optional[bool] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert the verdict to a standard dictionary representation."""
        return asdict(self)
