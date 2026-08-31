"""Step-6 decision tables and freeze-gate tests."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.projection.shadow.step6_decision import (
    apply_counterfactual_predictions,
    build_component_table,
    build_counterfactual_table,
    evaluate_candidate_gates,
    identify_codominant_components,
)


def _toy_players() -> pd.DataFrame:
    rows = []
    for fold, season in (("2023->2024", 2024), ("2024->2025", 2025)):
        for i, position in enumerate(("RB", "WR")):
            for j in range(10):
                # Raw and availability nearly cancel.
                raw = 20.0 if j < 7 else -5.0
                avail = -18.0 if j < 7 else 4.0
                comp = -1.5
                fin = -2.0
                actual = 100.0
                total = raw + avail + comp + fin
                v1 = actual + total
                rows.append({
                    "player_id": f"{season}-{position}-{j}",
                    "fold": fold,
                    "season": season,
                    "position": position,
                    "draft_relevant_top120": j < 6,
                    "actual_points": actual,
                    "v1_pred": v1,
                    "total_error": total,
                    "raw_rate_error": raw,
                    "composition_rate_effect": comp,
                    "availability_effect": avail,
                    "finalization_remainder": fin,
                })
    return pd.DataFrame(rows)


class Step6DecisionTests(unittest.TestCase):
    def test_codominant_flags_near_ties(self):
        codom = identify_codominant_components({
            "raw_rate_error": -26.0,
            "availability_effect": 25.5,
            "composition_rate_effect": -1.6,
            "finalization_remainder": -4.0,
        })
        self.assertIn("raw_rate_error", codom)
        self.assertIn("availability_effect", codom)
        self.assertNotIn("composition_rate_effect", codom)

    def test_component_table_has_populations(self):
        table = build_component_table(_toy_players())
        self.assertTrue(set(table["population"]) >= {"all_eligible", "top120"})
        self.assertIn("raw_rate_availability_covariance", table.columns)
        top = table[table["population"].eq("top120") & table["position"].eq("RB")]
        self.assertFalse(top.empty)
        self.assertTrue(np.isfinite(top["raw_rate_error_mean"]).all())

    def test_counterfactuals_include_four_arms(self):
        cf = build_counterfactual_table(_toy_players())
        self.assertEqual(set(cf["candidate"]), set({
            "v1_control",
            "availability_only",
            "raw_rate_only",
            "joint_diagnostic",
        }))
        scored = apply_counterfactual_predictions(_toy_players())
        # Joint removes both components from the prediction.
        expected = (
            scored["v1_pred"]
            - scored["raw_rate_error"]
            - scored["availability_effect"]
        )
        np.testing.assert_allclose(
            scored["pred_joint_diagnostic"], expected, atol=1e-9
        )

    def test_joint_never_eligible_for_freeze(self):
        players = _toy_players()
        cf = build_counterfactual_table(players)
        result = evaluate_candidate_gates(
            cf, players, candidate="joint_diagnostic", population="top120"
        )
        self.assertFalse(result["eligible_for_freeze"])
        self.assertEqual(result.get("verdict"), "hold_v1_structural_role")
        self.assertIn("joint_repair_is_diagnostic_only", result.get("reason", ""))


if __name__ == "__main__":
    unittest.main()
