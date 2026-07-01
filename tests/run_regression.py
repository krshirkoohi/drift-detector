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
