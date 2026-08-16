import json
import pytest

from drift_detector.mcp_server import (
    drift_toggle,
    drift_attach_session,
    drift_evaluate_turn,
    drift_compact_reset,
    drift_rebase,
    drift_get_status,
)


def test_mcp_toggle():
    # 1. Turn OFF
    msg = drift_toggle("off", session_id="test-toggle")
    assert "OFF" in msg

    # 2. Evaluate turn while disabled -> zero cost bypass
    res = json.loads(drift_evaluate_turn("Some response", user_prompt="Some query", session_id="test-toggle"))
    assert res["status"] == "disabled"
    assert res["drifted"] is False

    # 3. Turn ON
    msg = drift_toggle("on", session_id="test-toggle")
    assert "ON" in msg


def test_mcp_auto_baseline_and_drift_lifecycle():
    sid = "test-auto-lifecycle"
    drift_attach_session(provider="test", session_id=sid)

    # Turn 1: Warmup 1/3 (calibrating)
    res1 = json.loads(drift_evaluate_turn("The quarterly earnings report shows solid growth.", user_prompt="Tell me about financial performance.", session_id=sid))
    assert res1["status"] == "calibrating"
    assert res1["warmup_turn"] == 1
    assert res1["warmup_target"] == 3

    # Turn 2: Warmup 2/3 (calibrating)
    res2 = json.loads(drift_evaluate_turn("Operating budget and costs remained within projected limits.", user_prompt="What about operational costs?", session_id=sid))
    assert res2["status"] == "calibrating"
    assert res2["warmup_turn"] == 2

    # Turn 3: Warmup 3/3 -> Baseline calibrated via BaselineStore, transition to monitoring
    res3 = drift_evaluate_turn("Revenue forecasting and cash flow analysis are aligned.", user_prompt="What about cash flow?", session_id=sid)
    assert "driftd" in res3 or "status" in res3

    # Check status summary
    status_raw = drift_get_status(session_id=sid)
    status = json.loads(status_raw)
    assert status["status"] == "ON"
    assert status["calibration_support"] in ["moderate", "high"]
    assert status["lifecycle_state"] == "monitoring"
    assert status["turns"] == 1
    assert status["baseline_samples"] == 3


def test_mcp_compaction_reset_preserves_anchor_by_default():
    sid = "test-compaction-mcp"
    drift_attach_session(provider="test", session_id=sid)

    # 3 warmup turns
    drift_evaluate_turn("Initial topic sentence A about software architecture.", session_id=sid)
    drift_evaluate_turn("Initial topic sentence B about microservice design.", session_id=sid)
    drift_evaluate_turn("Initial topic sentence C about database clustering.", session_id=sid)

    # Trigger default compaction (rebase_anchor=False)
    compact_summary = "Summary: The user and agent discussed backend architecture."
    msg = drift_compact_reset(compact_summary, rebase_anchor=False, session_id=sid)
    assert "mission anchor preserved" in msg

    # Trigger explicit rebase compaction (rebase_anchor=True)
    msg_rebase = drift_compact_reset(compact_summary, rebase_anchor=True, session_id=sid)
    assert "re-baselined on compacted summary" in msg_rebase


def test_mcp_multi_session_isolation():
    # Session A: Financial domain
    drift_attach_session(provider="test", session_id="session-fin")
    drift_evaluate_turn("Finance and investment portfolio analytics.", session_id="session-fin")
    drift_evaluate_turn("Stock market options trading algorithms.", session_id="session-fin")
    drift_evaluate_turn("Bond yields and fixed income instruments.", session_id="session-fin")

    # Session B: Culinary domain
    drift_attach_session(provider="test", session_id="session-food")
    drift_evaluate_turn("Sourdough bread flour hydration and starter culture.", session_id="session-food")
    drift_evaluate_turn("French croissant lamination with cold butter.", session_id="session-food")
    drift_evaluate_turn("Artisan pizza dough fermentation at room temp.", session_id="session-food")

    status_fin = json.loads(drift_get_status(session_id="session-fin"))
    status_food = json.loads(drift_get_status(session_id="session-food"))

    assert status_fin["session_id"] == "session-fin"
    assert status_food["session_id"] == "session-food"
    assert status_fin["lifecycle_state"] == "monitoring"
    assert status_food["lifecycle_state"] == "monitoring"
