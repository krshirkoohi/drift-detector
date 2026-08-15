from __future__ import annotations

from typing import List, Optional

from .baseline import BaselineStore
from .detector import DriftDetector as InnerDetector
from .detector import TurnScore as DriftResult
from .embedding import EmbeddingProvider

class DriftDetector:
    """Core DriftDetector typed API contract."""
    
    def __init__(self, inner: InnerDetector):
        self._inner = inner

    @classmethod
    def from_examples(
        cls,
        baseline_texts: List[str],
        provider: EmbeddingProvider,
        metric: str = "cosine",
        use_trend: bool = True,
        threshold: Optional[float] = None
    ) -> DriftDetector:
        """Initialize the DriftDetector from a list of baseline texts."""
        store = BaselineStore(provider)
        baseline = store.build(baseline_texts)
        
        # Override threshold if explicitly provided
        if threshold is not None:
            if metric == "cosine":
                baseline.cosine_threshold = threshold
            else:
                baseline.euclidean_threshold = threshold
                
        inner = InnerDetector(
            baseline=baseline,
            provider=provider,
            metric=metric,
            use_trend=use_trend
        )
        return cls(inner)

    def score(self, response: str) -> DriftResult:
        """Score a single response and return the drift result."""
        return self._inner.score(response)
        
    def summary(self) -> dict:
        """Return a summary of the session."""
        return self._inner.summary()
