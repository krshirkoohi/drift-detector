"""
tests/test_session_e2e.py — End-to-end unit tests for DriftSession.
"""
from __future__ import annotations

import unittest
from drift_detector.embeddings import DeterministicEmbeddingAdapter
from drift_detector.session import DriftSession

BASELINE = [
    "The quarterly budget shows revenue growth across all product lines this year.",
    "Operating expenses were reduced by renegotiating supplier contracts in the second quarter.",
    "Cash flow projections indicate we can fund the expansion without additional borrowing.",
    "The finance team reconciled the ledger and closed the monthly accounts on schedule.",
    "Gross margin improved after we adjusted pricing on the subscription tiers.",
    "The audit confirmed that our financial statements comply with reporting standards.",
    "Investment in the new billing system will pay back within eighteen months.",
    "Headcount costs remain the largest line item in the operating budget.",
    "The board approved the capital allocation plan for the next fiscal year.",
    "Currency fluctuations had a minor impact on consolidated revenue this quarter.",
]

ON_TOPIC = "The revenue forecast for next quarter assumes stable subscription pricing and controlled expenses."
OFF_TOPIC = "The dragon soared over the misty mountains while the wizard chanted ancient spells at dawn."

class TestSessionE2E(unittest.TestCase):
    def test_initialise_and_observe_on_topic(self):
        adapter = DeterministicEmbeddingAdapter(dim=256)
        session = DriftSession.initialise(
            known_good_responses=BASELINE,
            embedding_adapter=adapter,
            name="test-session",
            metric="cosine",
            threshold=0.9
        )
        
        self.assertEqual(len(session.examples), 10)
        self.assertEqual(session.threshold, 0.9)
        
        # Test on-topic
        verdict = session.observe(ON_TOPIC)
        self.assertFalse(verdict.drift_detected)
        self.assertFalse(verdict.recommend_fresh_chat)
        self.assertEqual(verdict.turn_index, 1)

    def test_observe_off_topic_drift(self):
        adapter = DeterministicEmbeddingAdapter(dim=256)
        session = DriftSession.initialise(
            known_good_responses=BASELINE,
            embedding_adapter=adapter,
            name="test-session",
            metric="cosine",
            use_trend=False
        )
        
        verdict = session.observe(OFF_TOPIC)
        self.assertTrue(verdict.drift_detected)
        self.assertTrue(verdict.recommend_fresh_chat)
        self.assertTrue(session.has_drifted)
        
        summary = session.summary()
        self.assertEqual(summary["turns"], 1)
        self.assertEqual(summary["drifted_turns"], 1)
        self.assertEqual(summary["drift_rate"], 1.0)
        self.assertTrue(summary["has_drifted"])
