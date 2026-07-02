"""
Drift Detector MVP package.
"""

from .baseline import BaselineStore
from .detector import DriftDetector, TurnScore
from .embeddings import (
    EmbeddingAdapter,
    GeminiEmbeddingAdapter,
    LocalEmbeddingAdapter,
    DeterministicEmbeddingAdapter,
    OpenAICompatibleEmbeddingAdapter,
    get_adapter,
)
from .harness import AgentHarness, TurnRecord, SessionSummary

__all__ = [
    "BaselineStore",
    "DriftDetector",
    "TurnScore",
    "EmbeddingAdapter",
    "GeminiEmbeddingAdapter",
    "LocalEmbeddingAdapter",
    "DeterministicEmbeddingAdapter",
    "OpenAICompatibleEmbeddingAdapter",
    "get_adapter",
    "AgentHarness",
    "TurnRecord",
    "SessionSummary",
]
