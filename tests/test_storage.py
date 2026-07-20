"""
tests/test_storage.py — Unit tests for baseline storage and session logging.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from drift_detector.storage import BaselineStorage, SessionLogger

class TestStorage(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_baseline_storage_save_and_load(self):
        filepath = os.path.join(self.test_dir, "test_baseline.json")
        examples = ["example response one", "example response two"]
        name = "test-baseline-name"
        description = "Test description"
        
        # Save baseline
        BaselineStorage.save(
            path=filepath,
            name=name,
            examples=examples,
            description=description
        )
        
        self.assertTrue(os.path.exists(filepath))
        
        # Load baseline back
        data = BaselineStorage.load(filepath)
        self.assertEqual(data["name"], name)
        self.assertEqual(data["description"], description)
        self.assertEqual(data["examples"], examples)

    def test_session_logger(self):
        logger = SessionLogger(self.test_dir)
        turn_data = {
            "turn_index": 1,
            "distance": 0.1234,
            "drift_detected": False
        }
        
        log_file = logger.log_turn("test-session", turn_data)
        self.assertTrue(os.path.exists(log_file))
        
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            import json
            logged_data = json.loads(lines[0].strip())
            self.assertEqual(logged_data["turn_index"], 1)
            self.assertEqual(logged_data["distance"], 0.1234)
