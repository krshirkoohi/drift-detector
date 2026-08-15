import pytest

from drift_detector.core import DriftDetector, DriftResult
from drift_detector.embedding import DeterministicProvider

def test_drift_detector_api_contracts():
    # 1. Provide a dummy deterministic provider for testing
    provider = DeterministicProvider(dim=16)
    
    # 2. Provide some baseline texts
    baseline_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn canine leaps above the sleepy hound.",
        "That agile red fox hurdles over the tired dog."
    ]
    
    # 3. Instantiate from_examples
    detector = DriftDetector.from_examples(
        baseline_texts=baseline_texts,
        provider=provider,
        metric="cosine",
        use_trend=False,
    )
    
    # Assert instantiation
    assert isinstance(detector, DriftDetector)
    
    # 4. Score a response
    response_text = "The quick brown fox is very active."
    result = detector.score(response_text)
    
    # Assert typed API DriftResult
    assert isinstance(result, DriftResult)
    assert hasattr(result, "cosine_distance")
    assert hasattr(result, "drifted")
    assert result.badge in ["drift detected", "threshold breach", "nominal"]
    
    # Ensure it's not throwing errors and computes successfully
    assert result.cosine_distance >= 0.0

def test_drift_detector_with_threshold_override():
    provider = DeterministicProvider(dim=16)
    baseline_texts = [
        "One two three four.",
        "Two three four five.",
        "Three four five six."
    ]
    
    detector = DriftDetector.from_examples(
        baseline_texts=baseline_texts,
        provider=provider,
        threshold=0.99 # explicit threshold
    )
    
    # Highly different text should trigger drift if the threshold was tight, 
    # but 0.99 is very loose.
    result = detector.score("Random off-topic text that shares no words.")
    
    assert isinstance(result.cosine_distance, float)
