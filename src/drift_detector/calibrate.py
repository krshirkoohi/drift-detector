#!/usr/bin/env python3
"""Calibrate Drift Detector offline using a local embedding provider."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from .baseline import BaselineStore
from .detector import DriftDetector
from .embedding import LocalTransformerProvider, l2_normalise


def run_calibration(baseline_path: str, output_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
    print(f"Loading baseline from: {baseline_path}")
    print(f"Loading local model: {model_name}...")
    
    if not os.path.exists(baseline_path):
        print(f"Error: Baseline file '{baseline_path}' not found.", file=sys.stderr)
        sys.exit(1)
        
    provider = LocalTransformerProvider(model_name=model_name)
    store = BaselineStore(provider)
    baseline = store.build_from_file(baseline_path)
    
    print("Computing distances against centroid...")
    vecs = l2_normalise(provider.embed(baseline.samples))
    distances = [float(1.0 - v @ baseline.centroid) for v in vecs]
    
    threshold = float(np.percentile(distances, 95))
    mu = float(np.mean(distances))
    std = float(np.std(distances))
    delta = std
    lambd = 3 * std
    
    config = {
        "model_name": model_name,
        "centroid": baseline.centroid.tolist(),
        "threshold": threshold,
        "mu": mu,
        "delta": delta,
        "lambda": lambd,
        "n_samples": len(baseline.samples),
    }
    
    print(f"Saving calibration config to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        
    print(f"Calibration complete. Calibrated Threshold: {threshold:.4f}")
    
    print("\nRunning validation pass...")
    detector = DriftDetector(
        baseline=baseline,
        provider=provider,
        metric="cosine",
        use_trend=True,
    )
    
    fpr_count = 0
    for text in baseline.samples:
        res = detector.score(text)
        if res.threshold_breach:
            fpr_count += 1
            
    fpr = fpr_count / len(baseline.samples) if baseline.samples else 0.0
    print(f"Validation FPR: {fpr*100:.1f}%")
    
    if fpr <= 0.05:
        print("✅ Validation passed: FPR is 5% or under.")
    else:
        print("❌ Validation warning: FPR exceeds 5%.")


def main():
    parser = argparse.ArgumentParser(description="Calibrate Drift Detector offline using a local model.")
    parser.add_argument(
        "--baseline",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "default.json"),
        help="Path to baseline JSON file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calibration_config.json",
        help="Path to output configuration JSON file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Local embedding model name."
    )
    
    args = parser.parse_args()
    run_calibration(args.baseline, args.output, model_name=args.model)


if __name__ == "__main__":
    main()
