#!/usr/bin/env python3
"""
Regression test runner for the drift-detector project.

This script loads the regression scenarios from a JSON fixture file,
initialises the DriftDetector, evaluates each response, checks if the
classification matches the expected label, and outputs a tabular summary.
It exits with a non-zero code if any test fails, and 0 if all pass.
"""

import os
import sys
import json
from typing import List, Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from drift_detector import BaselineStore, DriftDetector

def run_tests() -> bool:
    """
    Run regression tests against the drift detector using the fixtures.
    
    Returns:
        True if all scenarios pass, False otherwise.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        return False

    # Define paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fixtures_path = os.path.join(base_dir, "tests", "fixtures", "regression_fixtures.json")
    baseline_path = os.path.join(base_dir, "baselines", "default.json")

    # Load fixtures
    if not os.path.exists(fixtures_path):
        print(f"❌ Error: Fixture file not found at {fixtures_path}", file=sys.stderr)
        return False

    with open(fixtures_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    # Load baseline store and initialise detector
    print(f"Loading baseline from: {baseline_path}")
    store = BaselineStore(baseline_path)
    
    # We will test using the default cosine metric with auto-calibrated threshold.
    detector = DriftDetector(store, api_key, threshold=None, metric="cosine")
    print("✅ Drift detector initialised successfully.")
    print(f"📊 Auto-calibrated Cosine Threshold (95th %ile): {detector.threshold:.4f}\n")

    # Table header
    header_fmt = "{:<20} | {:<40} | {:<8} | {:<8} | {:<9} | {:<9} | {:<6}"
    row_fmt = "{:<20} | {:<40} | {:<8} | {:<8.4f} | {:<9.4f} | {:<9} | {:<6}"
    
    print("-" * 110)
    print(header_fmt.format("Scenario ID", "Response Snippet", "Expected", "Distance", "Threshold", "Predicted", "Status"))
    print("-" * 110)

    all_passed = True
    for scenario in scenarios:
        scenario_id = scenario["id"]
        text = scenario["text"]
        expected_label = scenario["expected_label"]
        
        # Evaluate response
        result = detector.check_response(text)
        
        distance = result["cosine_distance"]
        threshold = result["threshold"]
        is_drifting = result["is_drifting"]
        
        predicted_label = "drifted" if is_drifting else "clean"
        status = "PASS" if predicted_label == expected_label else "FAIL"
        
        if status == "FAIL":
            all_passed = False
            
        snippet = text[:37] + "..." if len(text) > 40 else text
        
        print(row_fmt.format(
            scenario_id,
            snippet,
            expected_label,
            distance,
            threshold,
            predicted_label,
            status
        ))

    print("-" * 110)
    
    # Run sequential trend test scenarios
    print("\n====================================================")
    print("      RUNNING SEQUENTIAL TREND TEST SCENARIOS       ")
    print("====================================================\n")
    
    # 1. One-turn spike sequence
    # 2 clean responses, 1 drifted response, 1 clean response
    # This should NOT trigger the sustained trend alarm on any turn.
    spike_sequence = [
        "This project implements a drift detector that measures the cosine distance of LLM output embeddings from a baseline centroid. It aims to flag degradation as context fills, ensuring low overhead and inline warnings in CLI sessions.",
        "This project implements a drift detector that measures the cosine distance of LLM output embeddings from a baseline centroid. It aims to flag degradation as context fills, ensuring low overhead and inline warnings in CLI sessions.",
        "Preheat the oven to 180 degrees Celsius and prepare your baking sheet for chocolate chip cookies.",
        "This project implements a drift detector that measures the cosine distance of LLM output embeddings from a baseline centroid. It aims to flag degradation as context fills, ensuring low overhead and inline warnings in CLI sessions."
    ]
    
    # 2. Sustained trend sequence
    # 3 responses with steadily increasing semantic distance
    # This SHOULD trigger the sustained trend alarm on the final response.
    trend_sequence = [
        "This project implements a drift detector that measures the cosine distance of LLM output embeddings from a baseline centroid. It aims to flag degradation as context fills, ensuring low overhead and inline warnings in CLI sessions.",
        "We need to monitor performance metrics of the servers, checking CPU load, RAM usage, and network bandwidth in real-time.",
        "Preheat the oven to 180 degrees Celsius and prepare your baking sheet for chocolate chip cookies."
    ]
    
    print("Running 'One-turn Spike' sequence (expected: NO sustained trend alarm)...")
    detector_trend_spike = DriftDetector(store, api_key, threshold=None, metric="cosine", use_trend=True)
    spike_alarms = []
    for idx, text in enumerate(spike_sequence):
        res = detector_trend_spike.check_response(text)
        is_drifting = res["is_drifting"]
        spike_alarms.append(is_drifting)
        print(f"  Turn {idx+1}: Cosine Distance: {res['cosine_distance']:.4f} | Drifting/Alarm: {is_drifting}")
        print(f"    PH Statistic: {res.get('ph_statistic'):.6f} | PH Threshold: {res.get('ph_threshold'):.6f} | Delta: {res.get('ph_delta'):.6f}")
        
    if any(spike_alarms):
        print("❌ FAIL: 'One-turn Spike' sequence triggered a trend alarm!")
        all_passed = False
    else:
        print("✅ PASS: 'One-turn Spike' sequence did not trigger the trend alarm.")
        
    print("\nRunning 'Sustained Trend' sequence (expected: alarm triggers on final turn)...")
    detector_trend_sustained = DriftDetector(store, api_key, threshold=None, metric="cosine", use_trend=True)
    trend_alarms = []
    for idx, text in enumerate(trend_sequence):
        res = detector_trend_sustained.check_response(text)
        is_drifting = res["is_drifting"]
        trend_alarms.append(is_drifting)
        print(f"  Turn {idx+1}: Cosine Distance: {res['cosine_distance']:.4f} | Drifting/Alarm: {is_drifting}")
        print(f"    PH Statistic: {res.get('ph_statistic'):.6f} | PH Threshold: {res.get('ph_threshold'):.6f} | Delta: {res.get('ph_delta'):.6f}")
        
    # We expect False, False, True for the trend alarms
    expected_trend = [False, False, True]
    if trend_alarms == expected_trend:
        print("✅ PASS: 'Sustained Trend' sequence triggered alarm exactly as expected (True only on turn 3).")
    else:
        print(f"❌ FAIL: 'Sustained Trend' sequence alarm sequence was {trend_alarms}, expected {expected_trend}!")
        all_passed = False

    print("\n" + "=" * 52)
    
    if all_passed:
        print("\n🎉 All regression tests passed successfully!")
    else:
        print("\n❌ Some regression tests failed. Please review the output above.")
        
    return all_passed

def main():
    """Entry point for the regression test script."""
    success = run_tests()
    if not success:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
