"""
tests/test_harsh_edge_cases.py — Harsher, in-depth edge-case testing for the refactored DriftSession.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import numpy as np

from drift_detector.embeddings import EmbeddingAdapter, DeterministicEmbeddingAdapter
from drift_detector.session import DriftSession
from drift_detector.models import DriftVerdict

class ZeroEmbeddingAdapter(EmbeddingAdapter):
    """Returns a vector of all zeros to simulate extreme adapter failures."""
    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        return [0.0] * self.dim

class TestHarshEdgeCases(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_empty_baseline_raises_value_error(self):
        adapter = DeterministicEmbeddingAdapter()
        with self.assertRaises(ValueError):
            DriftSession.initialise(
                known_good_responses=[],
                embedding_adapter=adapter
            )

    def test_zero_vector_embeddings_handling(self):
        # Ensure zero-magnitude vectors do not trigger division-by-zero crashes
        zero_adapter = ZeroEmbeddingAdapter()
        
        # We must mock some responses. The initializer should handle zero norms.
        session = DriftSession.initialise(
            known_good_responses=["test response 1", "test response 2", "test response 3"],
            embedding_adapter=zero_adapter,
            metric="cosine"
        )
        
        # Centroid and threshold calculations should complete without crashing
        self.assertEqual(session.threshold, 1.0)
        
        # Test observing a turn that also yields a zero vector
        verdict = session.observe("some text")
        self.assertEqual(verdict.cosine_distance, 1.0)
        self.assertFalse(verdict.drift_detected)

    def test_zero_variance_in_baseline(self):
        # All baseline responses are identical -> distance variance is 0.0
        adapter = DeterministicEmbeddingAdapter()
        session = DriftSession.initialise(
            known_good_responses=["identical", "identical", "identical"],
            embedding_adapter=adapter,
            metric="cosine"
        )
        
        # Confirm fallback values are active to prevent zero std dev
        self.assertEqual(session.ph_delta, 0.001)  # 0.1 * 0.01 std_dev fallback
        self.assertEqual(session.ph_threshold, 0.05)  # 5.0 * 0.01 std_dev fallback
        
        # Observing identical text should be clean
        v1 = session.observe("identical")
        self.assertFalse(v1.drift_detected)

    def test_euclidean_metric_calibration(self):
        # Test Euclidean distance metric calculations
        adapter = DeterministicEmbeddingAdapter()
        session = DriftSession.initialise(
            known_good_responses=["finance revenue projection", "finance cost balance sheet", "finance ledger close"],
            embedding_adapter=adapter,
            metric="euclidean"
        )
        
        self.assertEqual(session.metric, "euclidean")
        
        # On-topic (similar words) should be close
        v1 = session.observe("finance projection")
        # Off-topic should be far
        v2 = session.observe("the dragon flew away")
        
        self.assertGreater(v2.euclidean_distance, v1.euclidean_distance)

    def test_session_logger_directory_creation(self):
        # Test logger with a nested, non-existent logging folder path
        nested_log_dir = os.path.join(self.test_dir, "nested", "metrics", "logs")
        adapter = DeterministicEmbeddingAdapter()
        
        session = DriftSession.initialise(
            known_good_responses=["on-task one", "on-task two", "on-task three"],
            embedding_adapter=adapter,
            log_dir=nested_log_dir
        )
        
        verdict = session.observe("any reply")
        
        # Check that path was created and turn was logged successfully
        self.assertTrue(os.path.exists(nested_log_dir))
        log_file = os.path.join(nested_log_dir, f"drift_metrics_{session.name}.jsonl")
        self.assertTrue(os.path.exists(log_file))
        
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("cosine_distance", content)
