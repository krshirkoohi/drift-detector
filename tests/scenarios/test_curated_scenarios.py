"""Curated development scenario test suite.

Note: These are hand-crafted development fixtures used for baseline validation
and candidate parameter exploration (candidate_v1), NOT held-out real-world transcripts.

Scenarios:
1. Focused Refactoring: 20 turns on-task development scenario.
2. Transient Detour: 20 turns on-task with a 2-turn git command tangent.
3. Severe Derailment: 10 turns on-task followed by 10 turns of French baking recipes.
"""
import json
from pathlib import Path
import pytest

from drift_detector.core import DriftDetector, DriftResult
from drift_detector.embedding import LocalTransformerProvider


def load_fixtures():
    fixture_path = Path(__file__).parent / "fixtures" / "curated_scenarios.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def real_neural_provider():
    """Real offline PyTorch transformer model (sentence-transformers/all-MiniLM-L6-v2)."""
    return LocalTransformerProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")


def test_curated_session_focused_refactoring_stays_nominal(real_neural_provider):
    """Verify 20-turn focused refactoring development scenario stays nominal."""
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
        assert res.drifted is False, f"Unexpected drift alert on focused turn {turn['turn']}"

    summary = detector.summary()
    assert summary["turns"] == 20
    assert summary["drifted_turns"] == 0
    assert summary["has_drifted"] is False
    assert summary["calibration_support"] in ["moderate", "high"]


def test_curated_session_temporary_tangent_is_forgiven(real_neural_provider):
    """Verify a 2-turn tangential query (git command syntax) is forgiven by Page-Hinkley in scenario."""
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

    # Final summary should show zero sustained drift across the scenario
    summary = detector.summary()
    assert summary["turns"] == 20
    assert summary["drifted_turns"] == 0
    assert summary["has_drifted"] is False


def test_curated_session_severe_derailment_triggers_drift(real_neural_provider):
    """Verify sustained off-topic tangent (French baking recipes) triggers drift alert in scenario."""
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
        assert results[i].drifted is False, f"Unexpected drift on initial on-topic turn {i+1}"

    # Sustained baking turns should trigger drift alert
    assert drift_first_detected_turn is not None, "Failed to detect severe derailment"
    assert 12 <= drift_first_detected_turn <= 16, f"Drift detected at turn {drift_first_detected_turn}, expected [12..16]"

    summary = detector.summary()
    assert summary["has_drifted"] is True
    assert summary["drifted_turns"] >= 4


def test_cloud_openai_compatible_live_embeddings():
    """Verify DriftDetector against live cloud embedding API across the severe derailment trajectory.

    Note: Tests cloud provider plumbing and trajectory evaluation when credentials are present.
    """
    import os
    from pathlib import Path
    from drift_detector.embedding import OpenAICompatibleProvider

    env_file = Path("~/.gemini/secrets.env").expanduser()
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("export OPEN_ROUTER="):
                key = line[19:].strip("\"'")
                os.environ["OPEN_ROUTER"] = key

    api_key = os.environ.get("OPEN_ROUTER") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("No cloud embedding API key found in environment or secrets.env")

    base_url = "https://openrouter.ai/api/v1" if "OPEN_ROUTER" in os.environ else "https://api.openai.com/v1"
    model = "openai/text-embedding-3-small" if "OPEN_ROUTER" in os.environ else "text-embedding-3-small"

    provider = OpenAICompatibleProvider(base_url=base_url, model=model, api_key=api_key)

    data = load_fixtures()["session_severe_derailment"]
    detector = DriftDetector.from_examples(
        baseline_texts=data["baseline_examples"],
        provider=provider,
        metric="cosine",
        use_trend=True,
    )

    results: list[DriftResult] = []
    first_drift = None
    for turn in data["turns"]:
        res = detector.score(turn["text"])
        results.append(res)
        if res.drifted and first_drift is None:
            first_drift = turn["turn"]

    # Initial on-topic turns stay nominal
    for i in range(10):
        assert results[i].drifted is False, f"Cloud false alarm on on-topic turn {i+1}"

    # Baking derailment triggers drift
    assert first_drift is not None, "Cloud provider failed to trigger on sustained derailment"
    assert first_drift >= 11

    summary = detector.summary()
    assert summary["has_drifted"] is True
