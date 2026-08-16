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

    def score(
        self,
        response: str,
        history_len: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        is_compacted: bool = False,
        compacted_summary: Optional[str] = None,
    ) -> DriftResult:
        """Score a single response and return the drift result."""
        return self._inner.score(
            response,
            history_len=history_len,
            prompt_tokens=prompt_tokens,
            is_compacted=is_compacted,
            compacted_summary=compacted_summary,
        )

    def handle_compaction(
        self,
        compacted_summary: Optional[str] = None,
        new_baseline: Optional[object] = None,
    ) -> str:
        """Reset detector accumulators and re-seed baseline after chat compaction."""
        return self._inner.handle_compaction(
            compacted_summary=compacted_summary,
            new_baseline=new_baseline,
        )

    def reset_page_hinkley(self) -> None:
        """Reset Page-Hinkley accumulators and latch state."""
        self._inner.reset_page_hinkley()

    def reset(self) -> None:
        """Reset session turn count and state."""
        self._inner.reset()

    def check_response(self, *args, **kwargs) -> dict:
        """Unified check_response interface."""
        return self._inner.check_response(*args, **kwargs)
        
    def summary(self) -> dict:
        """Return a summary of the session."""
        return self._inner.summary()

