"""
eval.py — Performance evaluation and benchmark module for drift-detector.

Measures embedding latency, distance calculation overhead, memory footprint,
and detection accuracy across test turns.
"""

import time
import os
import sys
import json
import resource
from typing import Dict, Any, List
import numpy as np

def _get_memory_mb() -> float:
    """Return max RSS memory footprint in megabytes."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in bytes on macOS, in kilobytes on Linux
    if sys.platform == "darwin":
        return rusage.ru_maxrss / (1024 * 1024)
    return rusage.ru_maxrss / 1024


def evaluate_performance(
    baseline_path: str = "baselines/default.json",
    provider: str = "deterministic",
    num_runs: int = 50,
) -> Dict[str, Any]:
    """Run a synthetic performance benchmark measuring throughput, latency, and memory footprint."""
    from .session import DriftSession
    from .baseline import BaselineStore
    from .embeddings import GeminiEmbeddingAdapter, DeterministicEmbeddingAdapter, LocalEmbeddingAdapter

    mem_before = _get_memory_mb()
    store = BaselineStore(baseline_path)
    examples = store.examples
    
    if provider == "hosted":
        api_key = os.environ.get("GEMINI_API_KEY")
        adapter = GeminiEmbeddingAdapter(api_key=api_key) if api_key else DeterministicEmbeddingAdapter()
    elif provider == "local":
        adapter = LocalEmbeddingAdapter()
    else:
        adapter = DeterministicEmbeddingAdapter()
        
    init_start = time.perf_counter()
    session = DriftSession.initialise(
        known_good_responses=examples,
        embedding_adapter=adapter,
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
    verdicts = []
    
    for i in range(num_runs):
        text = sample_turns[i % len(sample_turns)]
        t0 = time.perf_counter()
        v = session.observe(text)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        verdicts.append(v)
        
    mem_after = _get_memory_mb()
    mem_delta_mb = mem_after - mem_before
    
    latencies_np = np.array(latencies)
    
    report = {
        "provider": provider,
        "baseline": store.name,
        "baseline_examples_count": len(examples),
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
        "drift_detection_stats": session.summary()
    }
    
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate drift-detector performance metrics")
    parser.add_argument("--baseline", default="baselines/default.json", help="Path to baseline file")
    parser.add_argument("--provider", choices=["deterministic", "local", "hosted"], default="deterministic", help="Embedding provider")
    parser.add_argument("--runs", type=int, default=50, help="Number of benchmark iterations")
    
    args = parser.parse_args()
    report = evaluate_performance(baseline_path=args.baseline, provider=args.provider, num_runs=args.runs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
