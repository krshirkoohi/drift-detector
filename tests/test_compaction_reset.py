import json
import pytest
import numpy as np

from drift_detector.core import DriftDetector, DriftResult
from drift_detector.detector import PageHinkley, TurnScore
from drift_detector.baseline import BaselineStore, Baseline
from drift_detector.embedding import DeterministicProvider
from drift_detector.harness import AgentHarness
from drift_detector.mcp_server import (
    drift_toggle,
    drift_compact_reset,
    drift_evaluate_turn,
    drift_get_status,
)


@pytest.fixture
def provider():
    return DeterministicProvider(dim=32)


@pytest.fixture
def sample_baseline(provider):
    texts = [
        "Distributed database replication and consensus protocols like Raft and Paxos.",
        "High-performance caching mechanisms, memory indexing, and horizontal sharding.",
        "Leader election algorithms, Byzantine fault tolerance, and log replication.",
    ]
    return BaselineStore(provider).build(texts)


def test_page_hinkley_reset():
    """Verify PageHinkley resets all accumulators and latch states to zero."""
    ph = PageHinkley(delta=0.005, lam=0.05, burn_in=1, sustain=2)
    
    # Establish baseline mean
    ph.update(0.1)
    # Push elevated values to trigger alarm and elevate accumulators
    ph.update(0.9)
    alarm = ph.update(0.95)
    assert alarm is True
    assert ph.statistic > 0
    assert ph.exceed_streak >= 2
    assert ph.n == 3
    
    # Reset
    ph.reset()
    assert ph.n == 0
    assert ph.mean == 0.0
    assert ph.cum == 0.0
    assert ph.cum_min == 0.0
    assert ph.exceed_streak == 0
    assert ph.statistic == 0.0


def test_detector_explicit_compaction_reset(provider, sample_baseline):
    """Verify detector handle_compaction resets PH state while preserving the original mission anchor."""
    det = DriftDetector(sample_baseline, provider, metric="cosine", use_trend=True)
    orig_centroid = np.copy(det.baseline.centroid)
    
    # Cause divergence / drift
    det.score("Distributed consensus Raft log replication.")
    det.score("How to bake chocolate chip cookies with butter and sugar.")
    det.score("Chocolate cookie recipes with vanilla extract.")
    score3 = det.score("Baking pastry and sourdough bread.")
    assert score3.drifted is True
    assert det.ph.statistic > 0
    
    # Perform compaction reset
    msg = det.handle_compaction()
    assert "[driftd] Chat compacted" in msg
    assert det.ph.statistic == 0.0
    assert det.ph.exceed_streak == 0
    assert det.has_drifted is False
    # Anchor preserved
    np.testing.assert_allclose(det.baseline.centroid, orig_centroid)
    
    # Next score should not inherit previous cumulative sum
    score_after = det.score("Raft consensus log replication algorithm.")
    assert score_after.drifted is False
    assert score_after.compacted_reset is False


def test_detector_compaction_preserves_anchor_default(provider, sample_baseline):
    """Verify detector preserves mission anchor by default even when summary is present."""
    det = DriftDetector(sample_baseline, provider, metric="cosine", use_trend=True)
    orig_centroid = np.copy(det.baseline.centroid)
    
    summary = "Building frontend UI components and React styling with Tailwind CSS."
    msg = det.handle_compaction(compacted_summary=summary, rebase_anchor=False)
    assert "[driftd] Chat compacted" in msg
    assert "anchor preserved" in msg
    
    # Mission anchor is preserved
    np.testing.assert_allclose(det.baseline.centroid, orig_centroid)
    
    # Off-topic turn still scores far from original mission centroid
    ui_score = det.score("Building frontend UI components and React styling.")
    assert ui_score.cosine_distance > 0.40


def test_detector_rebase_explicit(provider, sample_baseline):
    """Verify detector re-seeds centroid when rebase() is explicitly invoked."""
    det = DriftDetector(sample_baseline, provider, metric="cosine", use_trend=True)
    
    summary = "Building frontend UI components and React styling with Tailwind CSS."
    msg = det.rebase(summary, reason="explicit_ui_task_switch")
    assert "rebased" in msg
    
    # Now frontend UI responses should be nominal against the new centroid
    ui_score = det.score("Building frontend UI components and React styling.")
    assert ui_score.cosine_distance < 0.20
    assert ui_score.drifted is False


def test_auto_compaction_history_truncation(provider, sample_baseline):
    """Verify automatic compaction reset when history length drops (len(history) < prev_len)."""
    det = DriftDetector(sample_baseline, provider, metric="cosine", use_trend=True)
    
    # Turn 1: 10 messages
    det.score("Initial system architecture discussion.", history_len=10)
    # Turn 2: 15 messages (off-topic)
    det.score("Off-topic gardening advice and organic soil.", history_len=15)
    
    # Turn 3: Context compacted -> history drops to 3 messages
    score3 = det.score("Continuing system design review.", history_len=3)
    assert score3.compacted_reset is True
    assert score3.notice is not None
    assert "[driftd] Chat compacted" in score3.notice


def test_auto_compaction_token_drop(provider, sample_baseline):
    """Verify automatic compaction reset when prompt tokens drop abruptly."""
    det = DriftDetector(sample_baseline, provider, metric="cosine", use_trend=True)
    
    # Turn 1: 8,000 prompt tokens
    det.score("Discussing database sharding.", prompt_tokens=8000)
    # Turn 2: 9,500 prompt tokens
    det.score("Off topic recipe talk.", prompt_tokens=9500)
    
    # Turn 3: Compaction occurred -> prompt tokens dropped to 1,200
    score3 = det.score("Back to database sharding.", prompt_tokens=1200)
    assert score3.compacted_reset is True
    assert score3.notice is not None
    assert "tokens dropped" in score3.notice


def test_core_api_compaction_methods(provider):
    """Verify core DriftDetector wrapper exposes compaction reset methods."""
    baseline_texts = [
        "Distributed database replication and consensus protocols.",
        "High-performance caching mechanisms and horizontal sharding.",
        "Leader election algorithms and fault-tolerant log state machines.",
    ]
    detector = DriftDetector.from_examples(baseline_texts, provider=provider, use_trend=True)
    
    # Score turns
    detector.score("Distributed database replication and consensus protocols.")
    detector.score("Off-topic cooking conversation.")
    
    # Compaction reset via wrapper
    msg = detector.handle_compaction(compacted_summary="Distributed key-value storage engine implementation.", rebase_anchor=True)
    assert "[driftd] Chat compacted" in msg
    
    score = detector.score("Distributed key-value storage engine implementation.", history_len=2)
    assert score.cosine_distance < 0.2
    assert score.drifted is False



def test_session_compaction_handling(provider):
    """Verify DriftDetector handles compaction resets and auto-detection."""
    detector = DriftDetector.from_examples(
        baseline_texts=[
            "Distributed consensus and Raft state machine replication.",
            "Database clustering and quorum writes.",
            "Fault tolerance in distributed leader election.",
        ],
        provider=provider,
        metric="cosine",
        use_trend=True,
    )
    
    # Initial on-topic turn
    detector.score("Distributed consensus and state machines.")
    # Divergent turns
    detector.score("Gardening tips for springtime flowers.")
    detector.score("Planting tomatoes and organic fertilizer.")
    
    # Compaction with rebase
    detector.handle_compaction(compacted_summary="The topic is now focused on microservice containerisation with Docker.", rebase_anchor=True)
    assert detector.ph.cum == 0.0
    assert detector.ph.mean == 0.0
    assert detector.has_drifted is False
    
    verdict = detector.score("Building container images and running Docker containers.")
    assert verdict.drifted is False


def test_harness_compact_command_hook(provider, sample_baseline):
    """Verify AgentHarness detects /compact hook and resets detector."""
    inner = DriftDetector(sample_baseline, provider, metric="cosine", use_trend=True)
    harness = AgentHarness(detector=inner, verbose=False)
    harness.start_session("test-compaction-harness")
    
    # Turn 1
    harness.process_turn("User query", "Consensus protocols in distributed systems.")
    # Turn 2: off-topic
    harness.process_turn("User query", "Baking cookies and chocolate pastry.")
    
    # Turn 3: User triggers /compact
    record = harness.process_turn("/compact The agent is designing a distributed key-value store.", "Understood, continuing key-value store design.")
    
    assert record.is_drifting is False
    harness.end_session()


def test_mcp_evaluate_turn_with_compaction():
    """Verify FastMCP evaluate_turn handles history truncation and /compact prompts."""
    drift_toggle("reset")
    drift_toggle("on")
    
    # Calibrate initial turns
    drift_evaluate_turn("Distributed consensus Raft protocol.", user_prompt="Explain Raft.")
    drift_evaluate_turn("Leader election and log replication.", user_prompt="Explain leader election.")
    
    # Turn with /compact user prompt
    res = drift_evaluate_turn(
        "Continuing the architectural discussion.",
        user_prompt="/compact Summary of previous distributed consensus discussion.",
    )
    # Status should reflect reset / nominal
    assert "driftd" in res or "nominal" in res or "status" in res


def test_proxy_compaction_detection():
    """Verify Proxy _score_turn detects history truncation and /compact."""
    from drift_detector import proxy
    proxy.PROVIDER = DeterministicProvider(dim=32)
    proxy.CONFIG["baseline_n"] = 3
    proxy.SESSIONS.clear()
    
    sid = "test-proxy-session"
    
    # Turn 1: Build baseline sample 1
    r1 = proxy._score_turn(sid, "Distributed consensus Raft replication.", history_len=2)
    assert r1["phase"] == "collecting-baseline"
    
    # Turn 2: Build baseline sample 2
    r2 = proxy._score_turn(sid, "Leader election state machines.", history_len=4)
    assert r2["phase"] == "collecting-baseline"

    # Turn 3: Build baseline sample 3 -> baseline-ready
    r3 = proxy._score_turn(sid, "Fault-tolerant log consensus.", history_len=6)
    assert r3["phase"] == "baseline-ready"
    
    # Turn 4: Scoring turn
    r4 = proxy._score_turn(sid, "Raft log replication mechanism.", history_len=8)
    assert r4["phase"] == "scoring"
    assert r4.get("compacted_reset", False) is False
    
    # Turn 5: Context compacted -> history drops from 8 to 2 messages
    r5 = proxy._score_turn(
        sid,
        "Continuing database discussion after compaction.",
        history_len=2,
        is_compacted=True,
        compacted_summary="Summary: Focus on Raft distributed consensus.",
    )
    assert r5["phase"] == "scoring"
    assert r5.get("compacted_reset") is True


