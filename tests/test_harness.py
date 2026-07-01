"""
tests/test_harness.py — Regression tests for the AgentHarness data-flow module.

These tests use the LocalEmbeddingAdapter (roberta-base, cached offline) so
that no API keys are required.  They validate:

  1. Basic turn capture and TurnRecord structure
  2. Session lifecycle (start → process_turn → end_session → summary)
  3. JSONL log output is valid and complete
  4. Summary statistics are accurate
  5. Drift status propagates correctly from the detector
  6. Multiple sequential sessions on the same harness instance
  7. Error cases: process_turn without start_session, double start_session

Run with:
    python -m pytest tests/test_harness.py -v
or:
    python tests/test_harness.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

BASELINES_DIR = os.path.join(PROJECT_ROOT, "baselines")
BASELINE_FILE = os.path.join(BASELINES_DIR, "default.json")

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# Responses that should score as CLEAN (similar to a Python-assistant baseline)
CLEAN_RESPONSES = [
    "Here is a simple Python function to add two numbers together: def add(a, b): return a + b",
    "You can sort a list in Python using the built-in sorted() function or list.sort() method.",
    "Python's dictionary comprehension syntax is: {k: v for k, v in items.items()}",
]

# Responses that should score as DRIFTED (off-topic / semantically remote)
DRIFTED_RESPONSES = [
    "The Renaissance was a cultural movement in Europe between the 14th and 17th centuries.",
    "To bake sourdough bread you need flour, water, salt, and a live starter culture.",
    "The Amazon rainforest produces approximately 20% of the world's oxygen supply.",
]


def _build_detector(
    threshold: float = 0.05,
    use_trend: bool = False,
) -> "DriftDetector":
    """Build a DriftDetector backed by the local roberta-base adapter."""
    from drift_detector.baseline import BaselineStore
    from drift_detector.detector import DriftDetector
    from drift_detector.embeddings import LocalEmbeddingAdapter

    adapter = LocalEmbeddingAdapter("roberta-base")
    store = BaselineStore(BASELINE_FILE)
    detector = DriftDetector(
        baseline_store=store,
        threshold=threshold,
        metric="cosine",
        use_trend=use_trend,
        embedding_adapter=adapter,
    )
    return detector


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestHarnessSessionLifecycle(unittest.TestCase):
    """Validates session start/end lifecycle and basic invariants."""

    def setUp(self):
        from drift_detector.harness import AgentHarness
        self.tmp_dir = tempfile.mkdtemp()
        self.detector = _build_detector()
        self.harness = AgentHarness(
            detector=self.detector,
            log_dir=self.tmp_dir,
            verbose=False,
        )

    def test_start_session_returns_session_id(self):
        session_id = self.harness.start_session()
        self.assertIsInstance(session_id, str)
        self.assertTrue(len(session_id) > 0)
        self.harness.end_session()

    def test_custom_session_id_preserved(self):
        custom_id = "test-session-xyz"
        returned_id = self.harness.start_session(session_id=custom_id)
        self.assertEqual(returned_id, custom_id)
        summary = self.harness.end_session()
        self.assertEqual(summary.session_id, custom_id)

    def test_double_start_raises(self):
        self.harness.start_session(session_id="s1")
        with self.assertRaises(RuntimeError):
            self.harness.start_session(session_id="s2")
        self.harness.end_session()

    def test_process_turn_without_session_raises(self):
        with self.assertRaises(RuntimeError):
            self.harness.process_turn("hello", "world")

    def test_end_session_without_start_raises(self):
        with self.assertRaises(RuntimeError):
            self.harness.end_session()


class TestHarnessTurnCapture(unittest.TestCase):
    """Validates per-turn TurnRecord structure and data flow."""

    def setUp(self):
        from drift_detector.harness import AgentHarness
        self.tmp_dir = tempfile.mkdtemp()
        self.detector = _build_detector()
        self.harness = AgentHarness(
            detector=self.detector,
            log_dir=self.tmp_dir,
            verbose=False,
        )

    def test_turn_record_has_required_fields(self):
        self.harness.start_session(session_id="field-check")
        record = self.harness.process_turn(
            user_prompt="What is Python?",
            agent_response=CLEAN_RESPONSES[0],
        )
        self.harness.end_session()

        # Required numeric fields
        self.assertIsInstance(record.cosine_distance, float)
        self.assertIsInstance(record.euclidean_distance, float)
        self.assertIsInstance(record.threshold, float)
        self.assertIsInstance(record.latency_ms, float)
        self.assertIsInstance(record.is_drifting, bool)

        # Metadata fields
        self.assertEqual(record.turn_index, 1)
        self.assertEqual(record.session_id, "field-check")
        self.assertIn("T", record.timestamp_utc)

    def test_turn_index_increments(self):
        self.harness.start_session(session_id="index-check")
        for i, response in enumerate(CLEAN_RESPONSES, start=1):
            record = self.harness.process_turn("prompt", response)
            self.assertEqual(record.turn_index, i)
        self.harness.end_session()

    def test_response_snippet_truncation(self):
        long_response = "A" * 500
        self.harness.start_session(session_id="snippet-check")
        record = self.harness.process_turn("prompt", long_response)
        self.harness.end_session()
        # Snippet should be capped and end with "..."
        self.assertLessEqual(len(record.agent_response_snippet), 203)
        self.assertTrue(record.agent_response_snippet.endswith("..."))

    def test_full_response_stored_untruncated(self):
        long_response = "A" * 500
        self.harness.start_session(session_id="full-response-check")
        record = self.harness.process_turn("prompt", long_response)
        self.harness.end_session()
        self.assertEqual(len(record.agent_response_full), 500)


class TestHarnessLogging(unittest.TestCase):
    """Validates JSONL turn log and JSON summary file output."""

    def setUp(self):
        from drift_detector.harness import AgentHarness
        self.tmp_dir = tempfile.mkdtemp()
        self.detector = _build_detector()
        self.harness = AgentHarness(
            detector=self.detector,
            log_dir=self.tmp_dir,
            verbose=False,
        )

    def test_jsonl_log_created(self):
        session_id = "log-test"
        self.harness.start_session(session_id=session_id)
        for resp in CLEAN_RESPONSES:
            self.harness.process_turn("prompt", resp)
        self.harness.end_session()

        log_path = os.path.join(self.tmp_dir, f"{session_id}.jsonl")
        self.assertTrue(os.path.exists(log_path), "JSONL log file was not created")

    def test_jsonl_log_has_correct_line_count(self):
        session_id = "line-count-test"
        n_turns = len(CLEAN_RESPONSES)
        self.harness.start_session(session_id=session_id)
        for resp in CLEAN_RESPONSES:
            self.harness.process_turn("prompt", resp)
        self.harness.end_session()

        log_path = os.path.join(self.tmp_dir, f"{session_id}.jsonl")
        with open(log_path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        self.assertEqual(len(lines), n_turns)

    def test_jsonl_lines_are_valid_json(self):
        session_id = "json-valid-test"
        self.harness.start_session(session_id=session_id)
        for resp in CLEAN_RESPONSES:
            self.harness.process_turn("prompt", resp)
        self.harness.end_session()

        log_path = os.path.join(self.tmp_dir, f"{session_id}.jsonl")
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)  # Should not raise
                    self.assertIn("turn_index", obj)
                    self.assertIn("cosine_distance", obj)
                    self.assertIn("is_drifting", obj)

    def test_summary_json_created(self):
        session_id = "summary-test"
        self.harness.start_session(session_id=session_id)
        for resp in CLEAN_RESPONSES:
            self.harness.process_turn("prompt", resp)
        self.harness.end_session()

        summary_path = os.path.join(self.tmp_dir, f"{session_id}_summary.json")
        self.assertTrue(os.path.exists(summary_path), "Session summary file was not created")

    def test_summary_json_is_valid(self):
        session_id = "summary-valid-test"
        self.harness.start_session(session_id=session_id)
        for resp in CLEAN_RESPONSES:
            self.harness.process_turn("prompt", resp)
        self.harness.end_session()

        summary_path = os.path.join(self.tmp_dir, f"{session_id}_summary.json")
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["session_id"], session_id)
        self.assertEqual(data["total_turns"], len(CLEAN_RESPONSES))
        self.assertIn("drift_rate", data)
        self.assertIn("turns", data)
        self.assertEqual(len(data["turns"]), len(CLEAN_RESPONSES))


class TestHarnessSummaryStats(unittest.TestCase):
    """Validates SessionSummary aggregate statistics."""

    def setUp(self):
        from drift_detector.harness import AgentHarness
        self.tmp_dir = tempfile.mkdtemp()
        # Use a low threshold so drifted responses are reliably flagged
        self.detector = _build_detector(threshold=0.05)
        self.harness = AgentHarness(
            detector=self.detector,
            log_dir=self.tmp_dir,
            verbose=False,
        )

    def test_summary_total_turns(self):
        self.harness.start_session(session_id="stat-total")
        all_responses = CLEAN_RESPONSES + DRIFTED_RESPONSES
        for resp in all_responses:
            self.harness.process_turn("prompt", resp)
        summary = self.harness.end_session()
        self.assertEqual(summary.total_turns, len(all_responses))

    def test_summary_drift_rate_in_range(self):
        self.harness.start_session(session_id="stat-rate")
        all_responses = CLEAN_RESPONSES + DRIFTED_RESPONSES
        for resp in all_responses:
            self.harness.process_turn("prompt", resp)
        summary = self.harness.end_session()
        self.assertGreaterEqual(summary.drift_rate, 0.0)
        self.assertLessEqual(summary.drift_rate, 1.0)

    def test_summary_mean_distances_are_positive(self):
        self.harness.start_session(session_id="stat-dists")
        for resp in CLEAN_RESPONSES:
            self.harness.process_turn("prompt", resp)
        summary = self.harness.end_session()
        self.assertGreater(summary.mean_cosine_distance, 0.0)
        self.assertGreater(summary.mean_euclidean_distance, 0.0)

    def test_peak_distances_gte_mean(self):
        self.harness.start_session(session_id="stat-peak")
        for resp in CLEAN_RESPONSES:
            self.harness.process_turn("prompt", resp)
        summary = self.harness.end_session()
        self.assertGreaterEqual(summary.peak_cosine_distance, summary.mean_cosine_distance)
        self.assertGreaterEqual(summary.peak_euclidean_distance, summary.mean_euclidean_distance)


class TestHarnessMultipleSessions(unittest.TestCase):
    """Validates that sequential sessions on the same harness are isolated."""

    def setUp(self):
        from drift_detector.harness import AgentHarness
        self.tmp_dir = tempfile.mkdtemp()
        self.detector = _build_detector()
        self.harness = AgentHarness(
            detector=self.detector,
            log_dir=self.tmp_dir,
            verbose=False,
        )

    def test_sequential_sessions_isolated(self):
        # Session 1: 2 turns
        self.harness.start_session(session_id="session-1")
        for resp in CLEAN_RESPONSES[:2]:
            self.harness.process_turn("prompt", resp)
        s1 = self.harness.end_session()

        # Session 2: 3 turns
        self.harness.start_session(session_id="session-2")
        for resp in CLEAN_RESPONSES:
            self.harness.process_turn("prompt", resp)
        s2 = self.harness.end_session()

        self.assertEqual(s1.total_turns, 2)
        self.assertEqual(s2.total_turns, 3)
        self.assertEqual(s1.session_id, "session-1")
        self.assertEqual(s2.session_id, "session-2")

    def test_ph_state_reset_between_sessions(self):
        """Page-Hinkley running state must reset on each new session."""
        detector = _build_detector(use_trend=True)
        from drift_detector.harness import AgentHarness
        harness = AgentHarness(detector=detector, log_dir=self.tmp_dir, verbose=False)

        harness.start_session(session_id="ph-1")
        for resp in DRIFTED_RESPONSES:
            harness.process_turn("prompt", resp)
        harness.end_session()

        # After session 1, the PH state should be reset
        self.assertEqual(detector.ph_n, 0)
        self.assertEqual(detector.ph_running_mean, 0.0)
        self.assertEqual(detector.ph_running_sum, 0.0)
        self.assertEqual(detector.ph_min_sum, 0.0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  DRIFT DETECTOR — AGENT HARNESS REGRESSION TESTS")
    print("=" * 65)
    print(f"  Baseline : {BASELINE_FILE}")
    print(f"  Provider : local (roberta-base)\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestHarnessSessionLifecycle,
        TestHarnessTurnCapture,
        TestHarnessLogging,
        TestHarnessSummaryStats,
        TestHarnessMultipleSessions,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n✅  All harness regression tests passed.")
        sys.exit(0)
    else:
        print(f"\n❌  {len(result.failures)} failure(s), {len(result.errors)} error(s).")
        sys.exit(1)
