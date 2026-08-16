"""Core DriftDetector API contracts and re-exports."""
from __future__ import annotations

from .detector import DriftDetector, DriftResult, PageHinkley, TurnScore
from .embedding import EmbeddingProvider

__all__ = [
    "DriftDetector",
    "DriftResult",
    "PageHinkley",
    "TurnScore",
    "EmbeddingProvider",
]
