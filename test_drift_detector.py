#!/usr/bin/env python3
import sys
import os

# Add src to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from drift_detector import BaselineStore, DriftDetector

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    baseline_path = os.path.join(os.path.dirname(__file__), "baselines", "default.json")
    print(f"Loading baseline from: {baseline_path}")
    
    store = BaselineStore(baseline_path)
    detector = DriftDetector(store, api_key, threshold=0.25)
    print("✅ Centroid calculated successfully.\n")

    # Test 1: On-topic response
    on_topic_text = (
        "This project implements a drift detector that measures the cosine distance "
        "of LLM output embeddings from a baseline centroid. It aims to flag degradation "
        "as context fills, ensuring low overhead and inline warnings in CLI sessions."
    )
    print("--- Test 1: On-Topic Response (Expect no drift) ---")
    print(f"Response: {on_topic_text}")
    res1 = detector.check_response(on_topic_text)
    print(f"Cosine Distance: {res1['cosine_distance']:.4f}")
    print(f"Is Drifting: {res1['is_drifting']}")
    print("-" * 50 + "\n")

    # Test 2: Off-topic response
    off_topic_text = (
        "To make chocolate chip cookies, preheat your oven to 350 degrees. "
        "Mix creamed butter and sugar, then add eggs and vanilla. Slowly stir in "
        "flour, baking soda, and chocolate chips before scooping onto a baking sheet."
    )
    print("--- Test 2: Off-Topic Response (Expect drift) ---")
    print(f"Response: {off_topic_text}")
    res2 = detector.check_response(off_topic_text)
    print(f"Cosine Distance: {res2['cosine_distance']:.4f}")
    print(f"Is Drifting: {res2['is_drifting']}")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
