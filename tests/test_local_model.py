#!/usr/bin/env python3
import sys
import os
import json

# Add src directory to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from drift_detector import BaselineStore, DriftDetector, LocalEmbeddingAdapter

def run_local_model_tests() -> bool:
    baseline_path = os.path.join(os.path.dirname(__file__), "..", "baselines", "default.json")
    fixtures_path = os.path.join(os.path.dirname(__file__), "fixtures", "regression_fixtures.json")
    
    print("====================================================")
    # en-GB spelling: Initialising
    print("      INITIALISING LOCAL MODEL DRIFT DETECTOR       ")
    print("====================================================")
    print(f"Loading baseline from: {baseline_path}")
    print("Loading local model: roberta-base (offline)...")
    
    try:
        # Load baseline and instantiate local adapter
        store = BaselineStore(baseline_path)
        adapter = LocalEmbeddingAdapter("roberta-base")
        
        # Instantiate detector with local adapter
        detector = DriftDetector(
            baseline_store=store,
            threshold=None,  # Auto-calibrate
            metric="cosine",
            embedding_adapter=adapter
        )
        print("✅ Drift detector initialised successfully.")
        print(f"📊 Auto-calibrated Cosine Threshold (95th %ile): {detector.threshold:.4f}\n")
    except Exception as e:
        print(f"❌ Error during initialisation: {e}")
        return False

    if not os.path.exists(fixtures_path):
        print(f"❌ Error: Fixtures file not found at: {fixtures_path}")
        return False
        
    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    print("-" * 110)
    print(f"{'Scenario ID':20} | {'Response Snippet':40} | {'Expected':8} | {'Distance':8} | {'Threshold':9} | {'Predicted':9} | {'Status'}")
    print("-" * 110)

    all_passed = True
    for scenario in fixtures:
        scenario_id = scenario["id"]
        text = scenario["text"]
        expected = scenario["expected_label"]
        
        try:
            res = detector.check_response(text)
            distance = res["cosine_distance"]
            is_drifting = res["is_drifting"]
            predicted = "drifted" if is_drifting else "clean"
            
            status = "PASS" if predicted == expected else "FAIL"
            if status == "FAIL":
                all_passed = False
                
            snippet = text[:37] + "..." if len(text) > 40 else text
            print(f"{scenario_id:20} | {snippet:40} | {expected:8} | {distance:.4f} | {detector.threshold:.4f} | {predicted:9} | {status}")
        except Exception as e:
            print(f"{scenario_id:20} | ERROR during check: {e}")
            all_passed = False

    print("-" * 110)
    
    if all_passed:
        print("\n🎉 All local model regression tests passed successfully!")
    else:
        print("\n❌ Some local model regression tests failed. Please review the output above.")
        
    return all_passed

if __name__ == "__main__":
    success = run_local_model_tests()
    sys.exit(0 if success else 1)
