"""
tests/test_embeddings.py — Unit tests for embedding adapters.
"""
from __future__ import annotations

import unittest
from drift_detector.embeddings import (
    get_adapter,
    DeterministicEmbeddingAdapter,
    EmbeddingAdapter
)

class TestEmbeddingAdapters(unittest.TestCase):
    def test_get_adapter(self):
        adapter = get_adapter("local")
        self.assertIsInstance(adapter, DeterministicEmbeddingAdapter)
        
        # Test case insensitivity
        adapter_caps = get_adapter("LOCAL")
        self.assertIsInstance(adapter_caps, DeterministicEmbeddingAdapter)

    def test_deterministic_embedding_adapter(self):
        adapter = DeterministicEmbeddingAdapter(dim=128)
        
        # Checking vector dimensions
        v1 = adapter.embed("hello world")
        self.assertEqual(len(v1), 128)
        
        # Verify determinism
        v2 = adapter.embed("hello world")
        self.assertEqual(v1, v2)
        
        # Verify different inputs yield different vectors
        v3 = adapter.embed("different content")
        self.assertNotEqual(v1, v3)
