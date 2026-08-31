"""Tests for availability-only Gate-A blend shadow repair."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.projection.shadow.availability_repair import (
    CANDIDATE_ID,
    blend_games,
    fit_nested_alphas,
    select_alpha_for_position,
    shadow_prediction,
    verify_rates_untouched,
)


class AvailabilityRepairUnitTests(unittest.TestCase):
    def test_blend_endpoints(self):
        gate = pd.Series([10.0, 14.0, 17.0])
        np.testing.assert_allclose(blend_games(gate, alpha=0.0), [17.0, 17.0, 17.0])
        np.testing.assert_allclose(blend_games(gate, alpha=1.0), [10.0, 14.0, 17.0])
        mid = blend_games(gate, alpha=0.5)
        np.testing.assert_allclose(mid, [13.5, 15.5, 17.0])

    def test_prediction_preserves_rates_and_finalization(self):
        frame = pd.DataFrame({
            "position": ["RB", "WR"],
            "gate_a_games": [12.0, 15.0],
            "composed_rate_ppg": [10.0, 8.0],
            "finalization_remainder": [-2.0, 1.0],
            "raw_rate_ppg": [9.0, 7.5],
            "draft_relevant_top120": [True, True],
            "actual_points": [100.0, 90.0],
        })
        pred = shadow_prediction(frame, alpha_by_position={"RB": 1.0, "WR": 0.0})
        # RB: 10*12 + (-2) = 118; WR: 8*17 + 1 = 137
        np.testing.assert_allclose(pred.to_numpy(), [118.0, 137.0])

    def test_nested_fit_uses_only_prior_folds(self):
        rows = []
        for fold, season in (("2023->2024", 2024), ("2024->2025", 2025)):
            for j in range(8):
                rows.append({
                    "fold": fold,
                    "season": season,
                    "position": "RB" if j < 4 else "WR",
                    "player_id": f"{season}-{j}",
                    "draft_relevant_top120": True,
                    "gate_a_games": 12.0,
                    "projected_games": 17.0,
                    "composed_rate_ppg": 10.0,
                    "raw_rate_ppg": 10.0,
                    "finalization_remainder": 0.0,
                    # Actual near rate * 12 so alpha=1 wins on train.
                    "actual_points": 120.0,
                    "v1_pred": 170.0,
                    "availability_effect": 50.0,
                    "raw_rate_error": 0.0,
                    "composition_rate_effect": 0.0,
                })
        players = pd.DataFrame(rows)
        scored, fit_log = fit_nested_alphas(players)
        self.assertEqual(fit_log[0]["positions"]["RB"]["source"], "cold_start")
        self.assertEqual(fit_log[0]["alpha_by_position"]["RB"], 1.0)
        # Second fold sees prior where alpha=1 matches actuals.
        self.assertEqual(fit_log[1]["positions"]["RB"]["source"], "nested_prior_folds")
        self.assertEqual(fit_log[1]["alpha_by_position"]["RB"], 1.0)
        self.assertIn("pred_availability_repair", scored.columns)
        guard = verify_rates_untouched(scored)
        self.assertTrue(guard["raw_rate_unchanged"])
        self.assertTrue(guard["composed_rate_unchanged"])

    def test_select_alpha_prefers_production_on_tie(self):
        # Constant actuals make every alpha equally bad/good on MAE of a
        # degenerate cell; ensure smaller alpha wins ties.
        train = pd.DataFrame({
            "position": ["RB"] * 4,
            "draft_relevant_top120": [True] * 4,
            "gate_a_games": [17.0] * 4,
            "composed_rate_ppg": [1.0] * 4,
            "finalization_remainder": [0.0] * 4,
            "actual_points": [17.0] * 4,
        })
        sel = select_alpha_for_position(train, position="RB")
        self.assertEqual(sel["alpha"], 0.0)

    def test_candidate_id_stable(self):
        self.assertEqual(CANDIDATE_ID, "shadow_availability_gate_a_blend_v1")


if __name__ == "__main__":
    unittest.main()
