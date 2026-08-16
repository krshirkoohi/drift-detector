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


def test_get_provider_mappings():
    from drift_detector.embedding import (
        get_provider,
        DeterministicProvider,
        LocalTransformerProvider,
        GeminiProvider,
        OpenAICompatibleProvider,
    )
    
    assert isinstance(get_provider("test"), DeterministicProvider)
    assert isinstance(get_provider("deterministic"), DeterministicProvider)
    assert isinstance(get_provider("hash"), DeterministicProvider)
    assert isinstance(get_provider("local"), LocalTransformerProvider)
    assert isinstance(get_provider("transformer"), LocalTransformerProvider)
    assert isinstance(get_provider("gemini"), GeminiProvider)
    assert isinstance(get_provider("openai"), OpenAICompatibleProvider)
    
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("nonexistent_provider_abc")


def test_local_transformer_provider_fail_fast():
    from drift_detector.embedding import LocalTransformerProvider
    
    # Nonexistent model should raise clear RuntimeError (no silent fallback)
    provider = LocalTransformerProvider(model_name="nonexistent/fake-model-12345")
    with pytest.raises(RuntimeError, match="Failed to load local embedding model"):
        provider.embed(["Some test text"])


def test_calibration_support_and_lifecycle_states():
    provider = DeterministicProvider(dim=16)
    
    # 1. High sample support (>=10 samples)
    high_samples = [f"Sample sentence {i} discussing distributed systems." for i in range(12)]
    det_high = DriftDetector.from_examples(high_samples, provider=provider)
    assert det_high.is_calibrated is True
    assert det_high.calibration_support == "high"
    assert det_high.confidence == "high"
    assert det_high.lifecycle_state == "monitoring"
    
    score_high = det_high.score("Valid distributed consensus message.")
    assert score_high.calibrated is True
    assert score_high.calibration_support == "high"
    assert score_high.confidence == "high"
    assert score_high.lifecycle_state == "monitoring"
    
    summary_high = det_high.summary()
    assert summary_high["calibrated"] is True
    assert summary_high["calibration_support"] == "high"
    assert summary_high["confidence"] == "high"
    assert summary_high["lifecycle_state"] == "monitoring"
    
    # 2. Moderate sample support (3-9 samples)
    mod_samples = [f"Sample sentence {i} on caching." for i in range(4)]
    det_mod = DriftDetector.from_examples(mod_samples, provider=provider)
    assert det_mod.is_calibrated is True
    assert det_mod.calibration_support == "moderate"
    assert det_mod.confidence == "moderate"
    assert det_mod.lifecycle_state == "monitoring"
    
    # 3. Low sample support (dynamic single summary / <3 samples)
    det_mod.rebase("Single summary sentence.", reason="test_single_summary")
    assert det_mod.is_calibrated is False
    assert det_mod.calibration_support == "low"
    assert det_mod.confidence == "low"
    assert det_mod.lifecycle_state == "calibrating"
    
    score_low = det_mod.score("Off topic turn.")
    assert score_low.calibrated is False
    assert score_low.calibration_support == "low"
    assert score_low.confidence == "low"
    assert score_low.lifecycle_state == "calibrating"
    assert score_low.badge == "calibrating"


