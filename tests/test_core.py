"""End-to-end unit tests for the drift_detector package (offline, deterministic)."""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import unittest

from drift_detector import BaselineStore, DeterministicEmbeddingAdapter, DriftDetector

PROVIDER = DeterministicEmbeddingAdapter(dim=256)

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


class TestCoreDriftDetector(unittest.TestCase):

    def test_baseline_and_thresholds(self):
        b = BaselineStore.from_examples(BASELINE)
        det = DriftDetector(b, metric="cosine", embedding_adapter=PROVIDER)
        self.assertEqual(len(b.examples), 10)
        self.assertTrue(np.linalg.norm(b.centroid) > 0)
        self.assertTrue(0 < det.threshold < 1)

    def test_on_topic_vs_off_topic_separation(self):
        b = BaselineStore.from_examples(BASELINE)
        det = DriftDetector(b, use_trend=False, embedding_adapter=PROVIDER)
        on = det.score(ON_TOPIC)
        off = det.score(OFF_TOPIC)
        self.assertTrue(off.cosine_distance > on.cosine_distance, "off-topic must be further from centroid")
        self.assertTrue(off.threshold_breach, "off-topic should breach the static threshold")

    def test_detector_end_to_end_session(self):
        b = BaselineStore.from_examples(BASELINE)
        det = DriftDetector(b, use_trend=True, embedding_adapter=PROVIDER)
        session = [
            "Next quarter's budget keeps operating expenses flat while revenue grows modestly.",
            "We reconciled the accounts and the ledger balances match the bank statements.",
            "Margin improvements come mostly from the renegotiated supplier contracts.",
            OFF_TOPIC,
            "The wizard's spells grew wilder as the dragon circled the burning castle towers.",
            "Elves and goblins clashed in the enchanted forest under the blood moon.",
            "The ancient prophecy foretold the return of the shadow king to the realm.",
        ]
        verdicts = [det.score(t).drifted for t in session]
        self.assertFalse(any(verdicts[:3]), "clean opening turns must not alarm")
        self.assertTrue(any(verdicts[3:]), "sustained off-topic run must eventually alarm")
        s = det.summary()
        self.assertEqual(s["turns"], 7)


if __name__ == "__main__":
    unittest.main()
