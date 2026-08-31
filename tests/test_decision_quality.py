"""Tests for E1 decision-quality evaluation contracts."""
import unittest

import numpy as np
import pandas as pd

from src.projection.evaluation import decision_quality as dq
from src.projection.evaluation import decision_quality_gate as dq_gate


class DecisionQualityTest(unittest.TestCase):
    def test_top_n_precision_recall_is_tie_aware(self):
        frame = pd.DataFrame({
            "preseason_position": ["RB", "RB", "RB", "RB"],
            "actual_points": [100.0, 100.0, 90.0, 10.0],
            "pure_model_points": [95.0, 95.0, 80.0, 5.0],
            "forecast_covered": [True, True, True, True],
            "depth_tier": [1.0, 1.0, 2.0, 5.0],
            "actual_games_played": [17, 17, 17, 0],
        })
        rows = dq.top_n_precision_recall_rows(
            frame,
            source_season=2024,
            target_season=2025,
            forecast_family="pure_model",
            points_col="pure_model_points",
        )
        rb_n12 = next(r for r in rows if r["position"] == "RB" and r["top_n"] == 12)
        self.assertEqual(rb_n12["predicted_top_n"], 4)
        self.assertGreaterEqual(rb_n12["precision"], 0.0)

    def test_adp_choice_regret_zero_when_optimal(self):
        matched = pd.DataFrame({
            "player_id": ["a", "b", "c"],
            "position": ["RB", "RB", "RB"],
            "mkt_raw": [1.0, 2.0, 3.0],
            "pure_model_points": [120.0, 110.0, 80.0],
            "market_informed_points": [120.0, 110.0, 80.0],
            "actual_points": [200.0, 150.0, 50.0],
            "adp": [1.0, 2.0, 3.0],
        })
        rows = dq.adp_choice_regret_rows(
            matched, source_season=2024, target_season=2025, window=3
        )
        pure = [r for r in rows if r["strategy"] == "pure_model" and r["pick_index"] == 1]
        self.assertTrue(pure)
        self.assertTrue(pure[0]["zero_regret"])

    def test_gate_fails_closed_without_baseline(self):
        gate = dq_gate.build_decision_quality_gate(
            evidence_manifest={"folds": [{"target_season": 2025}], "sha256": {"manifest.json": "x"}},
            evaluation_payload={"segment_metrics": pd.DataFrame()},
            frozen_baseline_manifest=None,
            required_folds=(2023, 2024, 2025),
        )
        self.assertFalse(gate["passes"])
        self.assertIn("missing_frozen_baseline_reference", gate["reasons"])

    def test_contract_hashes_are_stable(self):
        first = dq.vorp_tier_contract_hashes()
        second = dq.vorp_tier_contract_hashes()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
