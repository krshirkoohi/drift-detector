#!/usr/bin/env python3
"""Interactive drift test CLI."""
from __future__ import annotations

import json
import os
import sys

# Ensure src is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from drift_detector.core import DriftDetector
from drift_detector.embedding import DeterministicProvider


def main():
    baseline_path = os.path.join(os.path.dirname(__file__), "baselines", "default.json")
    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])

    provider = DeterministicProvider(dim=256)
    detector = DriftDetector.from_examples(samples, provider=provider, use_trend=True)

    print("\n--- Drift Detector Interactive Session ---")
    print(f"Baseline: {len(samples)} examples (Domain: Financial & Operational reporting)")
    print("Type any text below to score it against the baseline.")
    print("Type 'exit' or press Ctrl+C to quit.\n")

    while True:
        try:
            user_input = input("Input > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break

        result = detector.score(user_input)
        print(f"        └─ status: {result.badge}  (cos: {result.cosine_distance:.4f} | euc: {result.euclidean_distance:.4f})\n")

    print("\nSession summary:")
    print(json.dumps(detector.summary(), indent=2))


if __name__ == "__main__":
    main()
