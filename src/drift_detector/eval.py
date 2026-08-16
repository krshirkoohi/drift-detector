"""eval.py — Performance evaluation and benchmark module for drift-detector.

Measures detector latency, distance calculation overhead, memory footprint,
and detection stats across test turns.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from typing import Any, Dict

import numpy as np

from .baseline import BaselineStore
from .detector import DriftDetector
from .embedding import get_provider


def _get_memory_mb() -> float:
    """Return max RSS memory footprint in megabytes."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return rusage.ru_maxrss / (1024 * 1024)
    return rusage.ru_maxrss / 1024


def evaluate_performance(
    baseline_path: str = "baselines/default.json",
    provider_name: str = "deterministic",
    num_runs: int = 50,
) -> Dict[str, Any]:
    """Run a performance benchmark measuring throughput, latency, and memory footprint."""
    mem_before = _get_memory_mb()
    provider = get_provider(provider_name)
    store = BaselineStore(provider)
    baseline = store.build_from_file(baseline_path)
    
    init_start = time.perf_counter()
    detector = DriftDetector(
        baseline=baseline,
        provider=provider,
        metric="cosine",
        use_trend=True,
    )
    init_time_ms = (time.perf_counter() - init_start) * 1000
    
    sample_turns = [
        "We reconciled the accounts and the ledger balances match the bank statements.",
        "The capital allocation plan funds the billing system upgrade this fiscal year.",
        "Next quarter budget keeps operating expenses flat while revenue grows modestly.",
        "The best banana bread recipe uses overripe bananas, cinnamon, and brown sugar.",
        "Sharks have cartilage skeletons and inhabit warm ocean waters.",
    ]
    
    latencies = []
    
    for i in range(num_runs):
        text = sample_turns[i % len(sample_turns)]
        t0 = time.perf_counter()
        detector.score(text)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        
    mem_after = _get_memory_mb()
    mem_delta_mb = mem_after - mem_before
    latencies_np = np.array(latencies)
    
    report = {
        "provider": provider_name,
        "baseline_samples_count": len(baseline.samples),
        "initialization_time_ms": round(init_time_ms, 2),
        "total_benchmark_turns": num_runs,
        "turn_latency_ms": {
            "mean": round(float(np.mean(latencies_np)), 2),
            "median": round(float(np.median(latencies_np)), 2),
            "p95": round(float(np.percentile(latencies_np, 95)), 2),
            "min": round(float(np.min(latencies_np)), 2),
            "max": round(float(np.max(latencies_np)), 2),
        },
        "memory_footprint_mb": {
            "rss_total": round(mem_after, 2),
            "rss_delta": round(mem_delta_mb, 2),
        },
        "drift_detection_stats": detector.summary()
    }
    
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate drift-detector performance metrics")
    parser.add_argument("--baseline", default="baselines/default.json", help="Path to baseline file")
    parser.add_argument("--provider", default="deterministic", help="Embedding provider name")
    parser.add_argument("--runs", type=int, default=50, help="Number of benchmark iterations")
    
    args = parser.parse_args()
    report = evaluate_performance(baseline_path=args.baseline, provider_name=args.provider, num_runs=args.runs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
