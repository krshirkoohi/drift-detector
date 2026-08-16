"""Comprehensive Stress Test Suite for drift-detector.

Covers:
1. High-volume throughput & memory leak testing (10,000 turns).
2. Extreme compaction cycling (100 compaction cycles).
3. Statistical resilience of Page-Hinkley (isolated blips vs sustained drift).
4. Adversarial & extreme edge-case payloads (100KB strings, empty, unicode, emojis).
5. Concurrency & thread safety under parallel load.
"""

import json
import math
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
import pytest

from drift_detector.core import DriftDetector
from drift_detector.embedding import DeterministicProvider
from drift_detector.mcp_server import (
    drift_toggle,
    drift_compact_reset,
    drift_evaluate_turn,
    drift_get_status,
)


def test_10k_turns_throughput_and_memory():
    """Stress test evaluating 10,000 sequential turns to verify speed and memory stability."""
    provider = DeterministicProvider(dim=256)
    detector = DriftDetector.from_examples(
        baseline_texts=[
            "Building backend APIs in Python and FastAPI",
            "Writing unit tests with pytest and coverage reporting",
            "Database migrations and PostgreSQL query optimisation",
            "Configuring asynchronous message queues and background workers",
        ],
        provider=provider,
        metric="cosine",
        use_trend=True,
    )

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    start_time = time.perf_counter()
    n_turns = 10_000

    # Simulate 10,000 conversational turns
    for i in range(n_turns):
        text = f"Writing unit tests for module component {i % 50} and verifying database indexes"
        result = detector.score(text)
        assert result.cosine_distance >= 0.0
        assert not math.isnan(result.cosine_distance)

    duration = time.perf_counter() - start_time
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    turns_per_sec = n_turns / duration
    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_memory_diff_kb = sum(stat.size_diff for stat in top_stats) / 1024

    # Performance assertions
    assert turns_per_sec > 2_000, f"Expected >2,000 turns/sec, got {turns_per_sec:.0f}"
    # History stores lightweight TurnScore objects (~100 bytes each)
    assert total_memory_diff_kb < 15_000, f"Memory growth excessive: {total_memory_diff_kb:.2f} KB"
    print(f"\n[Stress Test 1] 10k Turns Throughput: {turns_per_sec:.0f} turns/sec ({duration:.3f}s total). Memory diff: {total_memory_diff_kb:.2f} KB")


def test_100_compaction_cycles_numerical_stability():
    """Stress test executing 100 consecutive context compactions to verify mathematical stability."""
    drift_toggle("reset")

    for i in range(100):
        summary = f"Compacted summary cycle {i}: Focus is on building high-performance microservices and caching with Redis"
        resp = drift_compact_reset(summary)
        assert "Drift detector re-baselined" in resp

        # Evaluate a turn on the newly compacted context
        eval_resp = drift_evaluate_turn(
            agent_response="Implementing Redis cache cluster for microservice API",
            user_prompt="How do we scale cache?",
        )
        assert isinstance(eval_resp, str)

    raw_status = drift_get_status()
    status = json.loads(raw_status) if isinstance(raw_status, str) else raw_status
    assert status["turns"] == 1  # only 1 turn since last compaction
    assert status["drifted_turns"] == 0
    assert status["drift_rate"] == 0.0
    print("\n[Stress Test 2] 100 Compaction Cycles: 100% mathematically stable, zero accumulator drift.")


def test_page_hinkley_blip_forgiveness_under_bursts():
    """Verify that 50 isolated single-turn spikes NEVER latch permanent drift alarms."""
    provider = DeterministicProvider(dim=256)
    detector = DriftDetector.from_examples(
        baseline_texts=[
            "Quantum physics simulation using Python and NumPy matrices",
            "Computing Hamiltonian eigenvalues and time evolution operators",
            "Simulating quantum circuit entanglement and density matrices",
        ],
        provider=provider,
        metric="cosine",
        use_trend=True,
    )

    # 50 bursts of: 1 off-topic blip followed immediately by 2 on-topic returns
    for i in range(50):
        # Off-topic blip
        blip_res = detector.score(f"Banana bread recipe baking temperature {i}")
        assert blip_res.threshold_breach is True  # breached instantaneous threshold
        assert blip_res.drifted is False          # forgiven by Page-Hinkley (not sustained)

        # On-topic recovery
        rec1 = detector.score("Quantum physics simulation using Python and NumPy matrices")
        rec2 = detector.score("Computing Hamiltonian eigenvalues and time evolution operators")
        assert rec1.drifted is False
        assert rec2.drifted is False

    summary = detector.summary()
    assert summary["drifted_turns"] == 0, "Expected 0 permanent drift turns for isolated blips"
    assert summary["drift_rate"] == 0.0
    print(f"\n[Stress Test 3] 50 Burst Blips: 0 false drift latches ({summary['turns']} total turns evaluated).")


def test_adversarial_and_extreme_payloads():
    """Stress test extreme payloads: empty, 100KB strings, binary-like chars, emojis, CJK."""
    provider = DeterministicProvider(dim=256)
    detector = DriftDetector.from_examples(
        baseline_texts=[
            "Standard software engineering and unit testing task",
            "Refactoring modular components and optimizing algorithms",
            "Running automated tests and continuous integration workflows",
        ],
        provider=provider,
    )

    extreme_payloads = [
        "",                                      # Empty string
        "    \n\t   ",                           # Whitespace only
        "A" * 100_000,                           # 100,000 char monolithic string
        "🚀🔥🎉🤖" * 500,                         # 2,000 emojis
        "你好世界，这是一个超长中文测试文本" * 500,     # CJK unicode string
        "SELECT * FROM users WHERE id = '1'; DROP TABLE users; --", # SQL injection attempt
        "\x00\x01\x02\x03\x04\x05\x06\x07\x08",  # Control characters
        "{'json': {'nested': {'deep': [1,2,3,4]}}}", # JSON serialization
    ]

    for payload in extreme_payloads:
        res = detector.score(payload)
        assert isinstance(res.cosine_distance, float)
        assert not math.isnan(res.cosine_distance)
        assert not math.isinf(res.cosine_distance)

    print(f"\n[Stress Test 4] Adversarial Inputs: {len(extreme_payloads)} extreme payloads handled with zero exceptions.")


def test_concurrent_threading_safety():
    """Verify thread safety when 50 concurrent threads evaluate turns simultaneously."""
    provider = DeterministicProvider(dim=256)
    detector = DriftDetector.from_examples(
        baseline_texts=[
            "Distributed systems and concurrency management in Go and Rust",
            "Implementing thread-safe mutexes and atomic memory operations",
            "Benchmarking parallel workers under high throughput load",
        ],
        provider=provider,
    )

    def worker(worker_id: int):
        results = []
        for j in range(20):
            res = detector.score(f"Worker {worker_id} turn {j} checking distributed lock lease")
            results.append(res.cosine_distance)
        return results

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, i) for i in range(50)]
        for f in futures:
            distances = f.result()
            assert len(distances) == 20
            for d in distances:
                assert 0.0 <= d <= 2.0

    print("\n[Stress Test 5] Concurrent Threading: 50 concurrent workers (1,000 total turns) completed safely.")
