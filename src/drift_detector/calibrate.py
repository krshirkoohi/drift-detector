#!/usr/bin/env python3
import argparse
import json
import os
import sys
import numpy as np

# Add src directory to python path if not installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drift_detector import BaselineStore, DriftDetector, LocalEmbeddingAdapter

def run_calibration(baseline_path: str, output_path: str, model_name: str = "roberta-base") -> None:
    print(f"Loading baseline from: {baseline_path}")
    print(f"Loading local model: {model_name}...")
    
    if not os.path.exists(baseline_path):
        print(f"Error: Baseline file '{baseline_path}' not found.", file=sys.stderr)
        sys.exit(1)
        
    store = BaselineStore(baseline_path)
    adapter = LocalEmbeddingAdapter(model_name)
    store.compute_centroid(adapter=adapter)
    
    print("Computing distances...")
    distances = []
    for text in store.examples:
        emb = adapter.embed(text)
        norm_emb = np.linalg.norm(emb)
        norm_centroid = np.linalg.norm(store.centroid)
        if norm_emb == 0 or norm_centroid == 0:
            dist = 1.0
        else:
            dist = 1.0 - np.dot(emb, store.centroid) / (norm_emb * norm_centroid)
        distances.append(dist)
        
    threshold = np.percentile(distances, 95)
    mu = np.mean(distances)
    std = np.std(distances)
    delta = std
    lambd = 3 * std
    
    config = {
        "centroid": store.centroid.tolist(),
        "threshold": float(threshold),
        "mu": float(mu),
        "delta": float(delta),
        "lambda": float(lambd)
    }
    
    print(f"Saving calibration config to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Calibration complete. Calibrated Threshold: {threshold:.4f}")
    
    print("\nRunning validation pass...")
    detector = DriftDetector(
        baseline_store=store,
        threshold=threshold,
        metric="cosine",
        embedding_adapter=adapter
    )
    
    fpr_count = 0
    for text in store.examples:
        res = detector.score(text)
        if res.threshold_breach:
            fpr_count += 1
            
    fpr = fpr_count / len(store.examples)
    print(f"Validation FPR: {fpr*100:.1f}%")
    
    if fpr <= 0.05:
        print("✅ Validation passed: FPR is 5% or under.")
    else:
        print("❌ Validation failed: FPR is over 5%.")
        sys.exit(1)

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
        default="roberta-base",
        help="Local embedding model name."
    )
    
    args = parser.parse_args()
    run_calibration(args.baseline, args.output, args.model)

if __name__ == "__main__":
    main()
