"""Tests for the leakage-safe fantasy evaluation.

Written as unittest.TestCase rather than bare pytest functions on purpose:
the project's documented verification step is
`python -m unittest discover -s tests`, which collects only TestCase
subclasses. As bare functions these six tests silently did not run under
the documented command - and they are the only coverage of the module the
freeze uses as its acceptance criterion. Both runners collect this form.
"""
import sqlite3
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src.projection import fantasy_evaluation as fe


def _roster_conn(rows):
    conn = sqlite3.connect(":memory:")
    pd.DataFrame(rows).to_sql("weekly_rosters", conn, index=False)
    return conn


class FantasyEvaluationTest(unittest.TestCase):
    def _patch(self, name, value):
        """monkeypatch.setattr equivalent, undone at test teardown."""
        patcher = mock.patch.object(fe, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_week1_population_is_frozen_and_retains_zero_game_rookie(self):
        conn = _roster_conn([
            {"season": 2025, "week": 1, "game_type": "REG", "player_id": "v1", "player_name": "Vet", "team": "A", "position": "WR", "status": "ACT", "years_exp": 2, "draft_number": 10, "pfr_id": "V"},
            {"season": 2025, "week": 1, "game_type": "REG", "player_id": "r1", "player_name": "Rook", "team": "A", "position": "TE", "status": "DEV", "years_exp": 0, "draft_number": np.nan, "pfr_id": "R"},
            {"season": 2025, "week": 1, "game_type": "REG", "player_id": "cut", "player_name": "Cut", "team": "A", "position": "RB", "status": "CUT", "years_exp": 0, "draft_number": np.nan, "pfr_id": "C"},
            # Later-season additions must not enter the frozen universe.
            {"season": 2025, "week": 9, "game_type": "REG", "player_id": "late", "player_name": "Late", "team": "A", "position": "QB", "status": "ACT", "years_exp": 5, "draft_number": 1, "pfr_id": "L"},
        ])
        rookies = pd.DataFrame({"season": [2025], "player_id": ["r1"]})
        out = fe.build_preseason_population(conn, 2025, rookies)
        self.assertEqual(set(out.player_id), {"v1", "r1"})
        rookie = out.set_index("player_id").loc["r1"]
        self.assertTrue(bool(rookie.is_rookie))
        self.assertEqual(rookie.preseason_position, "TE")

    def test_target_outcomes_never_reach_forecast_transform(self):
        feat = pd.DataFrame({
            "season": [2024, 2025],
            "player_id": ["v", "v"],
            "passing_yards": [100.0, 9999.0],
            "games_played": [1.0, 17.0],
        })
        rookie = pd.DataFrame({
            "season": [2024, 2025],
            "player_id": ["old", "new"],
            "games_played": [10.0, 17.0],
            "opportunity_games": [10.0, 17.0],
            "passing_yards_pg": [10.0, 500.0],
        })
        population = pd.DataFrame({
            "player_id": ["v"], "preseason_position": ["QB"],
            "preseason_team": ["A"], "is_rookie": [False],
        })
        seen = {}

        self._patch("build_preseason_rookie_cohort", lambda *a, **k: rookie.copy())
        self._patch("build_preseason_population", lambda *a, **k: population.copy())

        def fake_forecast(conn, history, population_arg, rookie_arg, source, target):
            seen["history_max"] = history.season.max()
            seen["rookie_target"] = rookie_arg[rookie_arg.season.eq(target)].copy()
            return pd.DataFrame({
                "player_id": ["v"], "model_forecast_points": [1.0],
                "projected_games": [1.0], "projected_volume_games": [1.0],
                "forecast_covered": [True],
            })

        self._patch("_forecast_from_history", fake_forecast)
        out, _ = fe.build_leakage_safe_forecasts(None, feat, 2024, 2025)
        self.assertEqual(seen["history_max"], 2024)
        held_out = seen["rookie_target"].iloc[0]
        self.assertTrue(pd.isna(held_out.games_played))
        self.assertTrue(pd.isna(held_out.opportunity_games))
        self.assertTrue(pd.isna(held_out.passing_yards_pg))
        self.assertEqual(out.loc[0, "model_points_end_to_end"], 1.0)

    def test_actuals_group_across_position_rows_and_missing_players_are_zero(self):
        forecasts = pd.DataFrame({
            "player_id": ["switch", "blank", "zero_games_stats", "never"]
        })
        feat = pd.DataFrame({
            "season": [2025, 2025, 2025, 2025],
            "player_id": ["switch", "switch", "blank", "zero_games_stats"],
            "receiving_yards": [40.0, 60.0, 0.0, -1.0],
            "receptions": [2.0, 3.0, 0.0, 0.0],
            "games_played": [4.0, 5.0, 1.0, 0.0],
        })
        out = fe.attach_actual_outcomes(forecasts, feat, 2025).set_index("player_id")
        self.assertEqual(out.loc["switch", "receiving_yards"], 100.0)
        self.assertEqual(out.loc["switch", "actual_points"], 12.5)
        self.assertEqual(out.loc["never", "actual_points"], 0.0)
        self.assertTrue(bool(out.loc["never", "actual_zero_game_outcome"]))
        self.assertEqual(out.loc["blank", "actual_points"], 0.0)
        self.assertFalse(bool(out.loc["blank", "actual_zero_game_outcome"]))
        self.assertEqual(out.loc["zero_games_stats", "actual_points"], -0.1)
        self.assertTrue(bool(out.loc["zero_games_stats", "actual_zero_game_outcome"]))

    def test_top_tier_includes_cutoff_ties_and_average_ranks(self):
        frame = pd.DataFrame({
            "player_id": list("abcd"),
            "preseason_position": ["QB"] * 4,
            "forecast_covered": [True] * 4,
            "actual_zero_game_outcome": [False] * 4,
            "actual_points": [10.0, 9.0, 9.0, 0.0],
            "model_points_end_to_end": [10.0, 8.0, 8.0, 0.0],
            "carry_forward_points": [10.0, 8.0, 8.0, 0.0],
            "availability_adjusted_points": [10.0, 8.0, 8.0, 0.0],
        })
        ranked, summary = fe.evaluate_forecasts(
            frame,
            tier_ranks={"QB": 2, "RB": 1, "WR": 1, "TE": 1},
            replacement_ranks={"QB": 3, "RB": 1, "WR": 1, "TE": 1},
        )
        self.assertEqual(ranked.set_index("player_id").loc["b", "actual_position_finish"], 2.5)
        row = summary.query(
            "position == 'QB' and scope == 'all_eligible' and method == 'model'").iloc[0]
        self.assertEqual(row.predicted_top_n, 3)
        self.assertEqual(row.actual_top_n, 3)
        self.assertEqual(row.tier_hits, 3)

    def test_vorp_uses_configured_kth_replacement_score(self):
        frame = pd.DataFrame({
            "player_id": list("abc"),
            "preseason_position": ["QB"] * 3,
            "forecast_covered": [True] * 3,
            "actual_zero_game_outcome": [False] * 3,
            "actual_points": [30.0, 20.0, 10.0],
            "model_points_end_to_end": [25.0, 20.0, 5.0],
            "carry_forward_points": [25.0, 20.0, 5.0],
            "availability_adjusted_points": [25.0, 20.0, 5.0],
        })
        _, summary = fe.evaluate_forecasts(
            frame,
            tier_ranks={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
            replacement_ranks={"QB": 2, "RB": 1, "WR": 1, "TE": 1},
        )
        row = summary.query(
            "position == 'QB' and scope == 'all_eligible' and method == 'model'").iloc[0]
        self.assertEqual(row.actual_replacement_points, 20.0)
        self.assertEqual(row.predicted_replacement_points, 20.0)
        self.assertTrue(np.isclose(row.vorp_mae, 10.0 / 3.0))

    def test_missing_forecast_is_retained_as_end_to_end_zero(self):
        feat = pd.DataFrame({"season": [2024], "player_id": ["covered"], "games_played": [1.0]})
        population = pd.DataFrame({
            "player_id": ["covered", "missing"],
            "preseason_position": ["QB", "QB"],
            "preseason_team": ["A", "A"],
            "is_rookie": [False, False],
        })
        self._patch("build_preseason_rookie_cohort",
                    lambda *a, **k: pd.DataFrame({"season": [], "player_id": []}))
        self._patch("build_preseason_population", lambda *a, **k: population.copy())
        self._patch("_forecast_from_history", lambda *a, **k: pd.DataFrame({
            "player_id": ["covered"], "model_forecast_points": [3.0],
            "projected_games": [1.0], "projected_volume_games": [1.0],
            "forecast_covered": [True],
        }))
        self._patch("_actual_player_totals", lambda *a, **k: pd.DataFrame({
            "player_id": ["covered"], "actual_points": [2.0]
        }))
        out, _ = fe.build_leakage_safe_forecasts(None, feat, 2024, 2025)
        missing = out.set_index("player_id").loc["missing"]
        self.assertFalse(bool(missing.forecast_covered))
        self.assertEqual(missing.model_points_end_to_end, 0.0)


if __name__ == "__main__":
    unittest.main()
