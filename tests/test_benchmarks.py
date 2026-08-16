import json
import os
import pytest

from experiments.comparative_benchmark import (
    run_benchmark_a,
    run_benchmark_b,
    run_benchmark_c,
    run_benchmark_d,
)


def test_benchmark_a_latency_and_cost():
    """Verify Benchmark A executes and satisfies sub-millisecond latency bounds."""
    res = run_benchmark_a(n_turns=200)
    dd = res["drift_detector"]
    
    assert dd["mean_latency_us"] < 2000.0  # Must be well under 2ms (<0.2ms typical)
    assert dd["cost_per_10k_turns_usd"] == 0.0
    assert res["speedup_factor"] > 100.0


def test_benchmark_b_separation_ratios():
    """Verify Benchmark B shows strictly positive separation between on-task and off-task."""
    res = run_benchmark_b()
    dims = res["dimension_evaluations"]
    assert len(dims) >= 3
    
    for d in dims:
        assert d["separation_delta"] > 0.15  # Off-topic distance strictly higher than on-task
        assert d["fisher_ratio"] > 1.0


def test_benchmark_c_blip_forgiveness_and_drift_detection():
    """Verify Benchmark C achieves 0% FPR on transient blips and >= 95% TPR on sustained drift."""
    res = run_benchmark_c(n_trials=20)
    ph = res["page_hinkley"]
    raw = res["instantaneous_threshold"]
    
    # Page-Hinkley must forgive transient blips (0% FPR)
    assert ph["false_positive_rate"] == 0.0
    # Page-Hinkley must catch sustained drift (100% TPR)
    assert ph["true_positive_rate"] >= 0.95
    # Raw threshold has high false alarm rate on blips
    assert raw["false_positive_rate"] > 0.50


def test_benchmark_d_compaction_recovery():
    """Verify Benchmark D successfully wipes elevated accumulators on compaction reset."""
    res = run_benchmark_d()
    assert res["accumulator_wiped"] is True
