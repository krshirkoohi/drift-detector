import os
import json
import numpy as np
import pytest
from drift_detector import BaselineStore, DriftDetector, LocalEmbeddingAdapter

def test_roberta_calibration_suite(tmp_path):
    baseline_path = os.path.join(os.path.dirname(__file__), "..", "baselines", "default.json")
    output_path = tmp_path / "calibration_config.json"
    
    # 1. Load data
    store = BaselineStore(baseline_path)
    
    # 2. Embed using local RoBERTa model
    adapter = LocalEmbeddingAdapter("roberta-base")
    store.compute_centroid(adapter=adapter)
    
    # 4. Compute Distances
    distances = []
    for text in store.examples:
        emb = adapter.embed(text)
        # Using cosine distance
        norm_emb = np.linalg.norm(emb)
        norm_centroid = np.linalg.norm(store.centroid)
        if norm_emb == 0 or norm_centroid == 0:
            dist = 1.0
        else:
            dist = 1 - np.dot(emb, store.centroid) / (norm_emb * norm_centroid)
        distances.append(dist)
        
    # 5. Calculate Parameters
    threshold = np.percentile(distances, 95)
    mu = np.mean(distances)
    std = np.std(distances)
    delta = std
    lambd = 3 * std
    
    # 6. Save Config
    config = {
        "centroid": store.centroid.tolist(),
        "threshold": float(threshold),
        "mu": float(mu),
        "delta": float(delta),
        "lambda": float(lambd)
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    assert output_path.exists()
    
    # 7. Validation Pass
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
    assert fpr <= 0.05, f"False Positive Rate {fpr*100:.1f}% is not 5% or under!"
