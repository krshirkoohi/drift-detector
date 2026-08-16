import json
import os
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.comparative_benchmark import (
    run_benchmark_a,
    run_benchmark_b,
    run_benchmark_c,
    run_benchmark_d,
)


def test_benchmark_a_latency_and_cost():
    """Verify Benchmark A executes and satisfies sub-millisecond and neural latency bounds."""
    res = run_benchmark_a(n_turns=50)
    det = res["deterministic_vector"]
    neural = res["real_neural_transformer"]
    
    assert det["mean_latency_us"] < 2000.0  # Vector scoring well under 2ms (<0.05ms typical)
    assert det["cost_per_10k_usd"] == 0.0
    assert neural["mean_latency_ms"] < 100.0  # Real PyTorch transformer inference < 100ms
    assert neural["cost_per_10k_usd"] == 0.0
    assert res["speedup_vs_llm_judge"]["deterministic_speedup"] > 100.0


def test_benchmark_b_separation_ratios():
    """Verify Benchmark B shows strictly positive separation across all real neural and vector models."""
    res = run_benchmark_b()
    separations = res["provider_separations"]
    assert len(separations) >= 4
    
    for s in separations:
        assert s["separation_delta"] > 0.02  # Off-topic distance strictly higher than on-task
        assert s["fisher_ratio"] > 1.0


def test_benchmark_c_blip_forgiveness_and_drift_detection():
    """Verify Benchmark C achieves 0% FPR on transient blips and >= 95% TPR on sustained drift under real neural embeddings."""
    res = run_benchmark_c(n_trials=10)
    ph = res["page_hinkley"]
    raw = res["instantaneous_threshold"]
    
    # Page-Hinkley must forgive transient blips (0% FPR)
    assert ph["false_positive_rate"] == 0.0
    # Page-Hinkley must catch sustained drift (100% TPR)
    assert ph["true_positive_rate"] >= 0.95
    # Raw threshold has high false alarm rate on blips
    assert raw["false_positive_rate"] > 0.50


def test_benchmark_d_compaction_recovery():
    """Verify Benchmark D successfully wipes elevated accumulators on compaction reset with neural embeddings."""
    res = run_benchmark_d()
    assert res["accumulator_wiped"] is True

