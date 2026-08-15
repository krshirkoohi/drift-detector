"""drift_detector: semantic drift detection for LLM agent sessions."""
from .baseline import Baseline, BaselineStore
from .detector import DriftDetector, PageHinkley, TurnScore
from .embedding import (
    DeterministicProvider,
    EmbeddingProvider,
    GeminiProvider,
    OpenAICompatibleProvider,
    get_provider,
    l2_normalise,
)

__version__ = "0.2.0"

__all__ = [
    "Baseline",
    "BaselineStore",
    "DriftDetector",
    "PageHinkley",
    "TurnScore",
    "DeterministicProvider",
    "EmbeddingProvider",
    "GeminiProvider",
    "OpenAICompatibleProvider",
    "get_provider",
    "l2_normalise",
]
