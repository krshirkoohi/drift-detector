"""drift_detector: semantic drift detection for LLM agent sessions."""
from .baseline import Baseline, BaselineStore
from .detector import DriftDetector, DriftResult, PageHinkley, TurnScore
from .embedding import (
    DeterministicProvider,
    EmbeddingProvider,
    GeminiProvider,
    LocalTransformerProvider,
    OpenAICompatibleProvider,
    get_provider,
    l2_normalise,
)

__version__ = "0.2.0"

__all__ = [
    "Baseline",
    "BaselineStore",
    "DriftDetector",
    "DriftResult",
    "PageHinkley",
    "TurnScore",
    "DeterministicProvider",
    "LocalTransformerProvider",
    "EmbeddingProvider",
    "GeminiProvider",
    "OpenAICompatibleProvider",
    "get_provider",
    "l2_normalise",
]
