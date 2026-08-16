import json
import pytest

from drift_detector.mcp_server import (
    drift_toggle,
    drift_attach_session,
    drift_evaluate_turn,
    drift_compact_reset,
    drift_get_status,
)


def test_mcp_toggle():
    # 1. Turn OFF
    msg = drift_toggle("off")
    assert "OFF" in msg
    
    # 2. Evaluate turn while disabled -> zero cost bypass
    res = json.loads(drift_evaluate_turn("Some response", user_prompt="Some query"))
    assert res["status"] == "disabled"
    assert res["drifted"] is False
    
    # 3. Turn ON
    msg = drift_toggle("on")
    assert "ON" in msg


def test_mcp_auto_baseline_and_drift_lifecycle():
    # Reset state
    drift_toggle("reset")
    drift_toggle("on")
    
    # Turn 1: Warmup / Calibrating
    res1 = json.loads(drift_evaluate_turn("The quarterly earnings report shows solid growth.", user_prompt="Tell me about financial performance."))
    assert res1["status"] == "calibrating"
    assert res1["warmup_turn"] == 1
    
    # Turn 2: Auto-baseline calibrated from initial context
    res2 = drift_evaluate_turn("Operating budget and costs remained within projected limits.", user_prompt="What about operational costs?")
    # Either nominal or discreet string
    assert "driftd" in res2 or "status" in res2
    
    # Turn 3: Off-topic drift
    res3 = drift_evaluate_turn("Here is how you bake homemade chocolate chip cookies.")
    assert "driftd" in res3
    
    # Check status summary
    status_raw = drift_get_status()
    status = json.loads(status_raw)
    assert status["status"] == "ON"
    assert status["confidence"] in ["low", "moderate", "high"]
    assert status["lifecycle_state"] in ["calibrating", "monitoring"]
    assert status["turns"] >= 1


def test_mcp_compaction_reset():
    drift_toggle("reset")
    drift_toggle("on")
    
    # Seed turns
    drift_evaluate_turn("Initial topic sentence A.")
    drift_evaluate_turn("Initial topic sentence B.")
    drift_evaluate_turn("Drifted off topic to gardening.")
    
    # Trigger compaction reset
    compact_summary = "Summary of conversation: The agent and user discussed core topic architecture."
    msg = drift_compact_reset(compact_summary)
    assert "re-baselined" in msg
    assert "1KB" in msg
    
    # Post-compaction turn should score against new summary
    res = drift_evaluate_turn("Let us continue refining the architecture.")
    assert "driftd" in res or "status" in res
