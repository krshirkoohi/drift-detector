"""Real-world multi-turn AI coding session validation test suite.

Validates detector accuracy against genuine multi-turn development transcripts:
1. Focused Refactoring: 20 turns on-task, zero drift alerts.
2. Transient Detour: 20 turns on-task with 2-turn tangent, forgiven by Page-Hinkley.
3. Severe Derailment: 10 turns on-task followed by 10 turns of baking recipes, sustained drift alarm triggered.
"""
import json
from pathlib import Path
import pytest

from drift_detector.core import DriftDetector, DriftResult
from drift_detector.embedding import LocalTransformerProvider, DeterministicProvider


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "coding_sessions.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def real_neural_provider():
    """Real offline PyTorch transformer model (sentence-transformers/all-MiniLM-L6-v2)."""
    return LocalTransformerProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")


def test_real_session_focused_refactoring_never_drifts(real_neural_provider):
    """Verify 20-turn focused refactoring session stays nominal with zero false alarms."""
    data = load_fixtures()["session_refactoring_focused"]
    baseline_texts = data["baseline_examples"]
    turns = data["turns"]

    detector = DriftDetector.from_examples(
        baseline_texts=baseline_texts,
        provider=real_neural_provider,
        metric="cosine",
        use_trend=True,
    )

    results: list[DriftResult] = []
    for turn in turns:
        res = detector.score(turn["text"])
        results.append(res)
        assert res.drifted is False, f"False positive drift alert on focused turn {turn['turn']}"

    summary = detector.summary()
    assert summary["turns"] == 20
    assert summary["drifted_turns"] == 0
    assert summary["has_drifted"] is False
    assert summary["confidence"] in ["moderate", "high"]


def test_real_session_temporary_tangent_is_forgiven(real_neural_provider):
    """Verify a 2-turn tangential query (git command syntax) is forgiven by Page-Hinkley."""
    data = load_fixtures()["session_temporary_tangent"]
    baseline_texts = data["baseline_examples"]
    turns = data["turns"]

    detector = DriftDetector.from_examples(
        baseline_texts=baseline_texts,
        provider=real_neural_provider,
        metric="cosine",
        use_trend=True,
    )

    results: list[DriftResult] = []
    for turn in turns:
        res = detector.score(turn["text"])
        results.append(res)

    # Turns 7 and 8 were git tangents
    # Page-Hinkley blip forgiveness should prevent session-level sustained drift
    assert results[6].drifted is False  # Turn 7
    assert results[7].drifted is False  # Turn 8

    # Final summary should show zero sustained drift across the session
    summary = detector.summary()
    assert summary["turns"] == 20
    assert summary["drifted_turns"] == 0
    assert summary["has_drifted"] is False


def test_real_session_severe_derailment_triggers_drift(real_neural_provider):
    """Verify sustained off-topic tangent (French baking recipes) triggers drift alert."""
    data = load_fixtures()["session_severe_derailment"]
    baseline_texts = data["baseline_examples"]
    turns = data["turns"]

    detector = DriftDetector.from_examples(
        baseline_texts=baseline_texts,
        provider=real_neural_provider,
        metric="cosine",
        use_trend=True,
    )

    results: list[DriftResult] = []
    drift_first_detected_turn = None

    for turn in turns:
        res = detector.score(turn["text"])
        results.append(res)
        if res.drifted and drift_first_detected_turn is None:
            drift_first_detected_turn = turn["turn"]

    # Initial 10 turns on e-commerce checkout should be nominal
    for i in range(10):
        assert results[i].drifted is False, f"False drift on initial on-topic turn {i+1}"

    # Sustained baking turns should trigger drift alert
    assert drift_first_detected_turn is not None, "Failed to detect severe derailment"
    assert 12 <= drift_first_detected_turn <= 16, f"Drift detected at turn {drift_first_detected_turn}, expected [12..16]"

    summary = detector.summary()
    assert summary["has_drifted"] is True
    assert summary["drifted_turns"] >= 4
