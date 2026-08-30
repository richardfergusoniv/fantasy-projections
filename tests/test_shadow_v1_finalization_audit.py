"""Finalization / ladder / corrections audit closeout tests."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.projection.shadow.finalization_audit import (
    audit_depth_ladder,
    audit_corrections_joblib,
)


class FinalizationAuditTests(unittest.TestCase):
    def test_depth_ladder_not_on_live_paths(self):
        result = audit_depth_ladder()
        self.assertIsNone(result["defect"])
        self.assertFalse(result["live_ladder_application_detected"])

    def test_corrections_te_only_on_disk(self):
        # Uses real models/corrections.joblib when present.
        with tempfile.TemporaryDirectory() as tmp:
            shadow = Path(tmp)
            (shadow / "stages").mkdir()
            result = audit_corrections_joblib(shadow)
        self.assertFalse(result["rb_wr_params_present"])
        self.assertIsNone(result["defect"])
        if result["exists"]:
            self.assertIn("TE", result["positions"])
            self.assertNotIn("RB", result["positions"])
            self.assertNotIn("WR", result["positions"])


if __name__ == "__main__":
    unittest.main()
