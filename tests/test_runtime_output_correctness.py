import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.comparison import sleeper_compare
from src.projection.fantasy_points import DESCRIPTIVE_COLS, compute_fantasy_points
from src.projection.predict import (
    _ensure_output_parent,
    add_projected_season_totals,
    add_team_pass_catch_coherence_flag,
    normalize_team_passing_volume,
    reconcile_qb_projected_volume_games,
    reconcile_stat_constraints,
)
from src.projection.transitions import receiving_share_scale


class RuntimeOutputCorrectnessTests(unittest.TestCase):
    def _projection_row(self, stat, point, low, high):
        row = {
            "player_id": "p1",
            "position": "QB",
            "season": 2026,
            "stat": stat,
            "pred_pg": point,
            "pred_pg_low": low,
            "pred_pg_high": high,
            "interval_low_n_flag": False,
        }
        row.update({c: None for c in DESCRIPTIVE_COLS})
        row.update({
            "display_name": "Test Quarterback",
            "team": "TST",
            "source": "veteran_model",
            "low_confidence": False,
            "projected_games": 10.0,
        })
        return row

    def test_negative_scoring_stat_uses_opposite_interval_endpoint(self):
        rows = [
            self._projection_row("passing_yards", 200.0, 100.0, 300.0),
            self._projection_row("interceptions", 1.0, 0.0, 2.0),
        ]
        out = compute_fantasy_points(pd.DataFrame(rows)).iloc[0]
        self.assertAlmostEqual(out["fantasy_pts"], 6.0)
        self.assertAlmostEqual(out["fantasy_pts_low"], 0.0)
        self.assertAlmostEqual(out["fantasy_pts_high"], 12.0)

    def test_final_stat_constraints_cap_children_at_parent_stats(self):
        rows = []
        for stat, values in {
            "attempts": (4.0, 2.0, 6.0),
            "completions": (5.0, 3.0, 7.0),
            "targets": (3.0, 1.0, 4.0),
            "receptions": (4.0, 2.0, 5.0),
        }.items():
            rows.append({
                "player_id": "p1", "position": "QB", "season": 2026,
                "stat": stat, "pred_pg": values[0],
                "pred_pg_low": values[1], "pred_pg_high": values[2],
            })
        out = reconcile_stat_constraints(pd.DataFrame(rows)).set_index("stat")
        for child, parent in (("completions", "attempts"), ("receptions", "targets")):
            for col in ("pred_pg", "pred_pg_low", "pred_pg_high"):
                self.assertLessEqual(out.loc[child, col], out.loc[parent, col])
            self.assertTrue(out.loc[child, "stat_constraint_applied"])

    def test_coherence_uses_participation_weighted_all_qb_volume(self):
        df = pd.DataFrame([
            {"player_id": "q1", "team": "TST", "position": "QB", "stat": "passing_yards", "pred_pg": 200.0, "projected_games": 10.0},
            {"player_id": "q2", "team": "TST", "position": "QB", "stat": "passing_yards", "pred_pg": 100.0, "projected_games": 7.0},
            {"player_id": "w1", "team": "TST", "position": "WR", "stat": "receiving_yards", "pred_pg": 150.0, "projected_games": 17.0},
        ])
        out = add_team_pass_catch_coherence_flag(reconcile_qb_projected_volume_games(df))
        # Expected passing/team-game = (200*10 + 100*7)/17.
        expected_ratio = 150.0 / (2700.0 / 17.0)
        self.assertAlmostEqual(out["team_pass_catch_ratio"].iloc[0], expected_ratio)
        self.assertFalse(bool(out["team_pass_catch_coherence_flag"].iloc[0]))

    def test_qb_volume_reconciliation_handles_10_17_and_25_marginal_games(self):
        rows = []
        for team, games in (("TEN", [6.0, 4.0]), ("SEV", [10.0, 7.0]), ("TWF", [15.0, 10.0])):
            for i, value in enumerate(games):
                rows.append({"player_id": f"{team}{i}", "team": team, "position": "QB",
                             "projected_games": value, "role": None,
                             "depth_chart_status": None})
        out = reconcile_qb_projected_volume_games(pd.DataFrame(rows))
        totals = out.groupby("team")["projected_volume_games"].sum()
        self.assertAlmostEqual(totals["TEN"], 17.0)
        self.assertAlmostEqual(totals["SEV"], 17.0)
        self.assertAlmostEqual(totals["TWF"], 17.0)
        self.assertEqual(out.groupby("team")["projected_games"].sum()["TWF"], 25.0)

    def test_qb_volume_reconciliation_preserves_singular_starter_first(self):
        df = pd.DataFrame([
            {"player_id": "s", "team": "TST", "position": "QB", "projected_games": 14.0,
             "role": "starter", "depth_chart_status": "curated"},
            {"player_id": "b1", "team": "TST", "position": "QB", "projected_games": 8.0,
             "role": "backup", "depth_chart_status": "curated"},
            {"player_id": "b2", "team": "TST", "position": "QB", "projected_games": 4.0,
             "role": "deep_bench", "depth_chart_status": "deep_bench_discounted"},
        ])
        out = reconcile_qb_projected_volume_games(df).set_index("player_id")
        self.assertAlmostEqual(out.loc["s", "projected_volume_games"], 14.0)
        self.assertAlmostEqual(out.loc["b1", "projected_volume_games"], 2.0)
        self.assertAlmostEqual(out.loc["b2", "projected_volume_games"], 1.0)

    def test_team_attempt_anchor_scales_starved_qb_room(self):
        rows = []
        for stat, rate in (("attempts", 10.0), ("completions", 6.0),
                           ("passing_yards", 80.0)):
            rows.append({
                "player_id": "q", "team": "CLE", "position": "QB", "stat": stat,
                "pred_pg": rate, "pred_pg_low": rate * .5, "pred_pg_high": rate * 1.5,
                "projected_volume_games": 17.0,
                "team_pass_attempts_pg_pred": 34.0,
                "team_passing_yards_pg_pred": 220.0,
            })
        out = normalize_team_passing_volume(pd.DataFrame(rows))
        attempts = out[out["stat"] == "attempts"].iloc[0]
        yards = out[out["stat"] == "passing_yards"].iloc[0]
        self.assertAlmostEqual(attempts["pred_pg"], 34.0)
        self.assertAlmostEqual(yards["pred_pg"], 220.0)

    def test_team_yard_anchor_enforces_identity_and_preserves_pre_ratio(self):
        rows = [
            {"player_id": "q", "team": "TST", "position": "QB", "stat": "passing_yards",
             "pred_pg": 200.0, "pred_pg_low": 150.0, "pred_pg_high": 250.0,
             "projected_games": 17.0, "projected_volume_games": 17.0,
             "team_pass_attempts_pg_pred": 34.0, "team_passing_yards_pg_pred": 240.0},
            {"player_id": "w", "team": "TST", "position": "WR", "stat": "receiving_yards",
             "pred_pg": 100.0, "pred_pg_low": 80.0, "pred_pg_high": 120.0,
             "projected_games": 17.0, "projected_volume_games": 17.0,
             "team_pass_attempts_pg_pred": 34.0, "team_passing_yards_pg_pred": 240.0},
        ]
        out = normalize_team_passing_volume(pd.DataFrame(rows))
        qb = out[out["position"] == "QB"].iloc[0]
        wr = out[out["position"] == "WR"].iloc[0]
        self.assertAlmostEqual(qb["pred_pg"], 240.0)
        self.assertAlmostEqual(wr["pred_pg"], 240.0)
        self.assertAlmostEqual(wr["team_pass_catch_ratio_pre_normalization"], 100 / 240)
        self.assertTrue(wr["team_pass_catch_pre_normalization_flag"])

    def test_receiving_share_cap_uses_exposure_weights(self):
        shares = pd.DataFrame([
            {"team": "LOW", "share": .4, "weight": .5},
            {"team": "LOW", "share": .4, "weight": .5},
            {"team": "HIGH", "share": .8, "weight": 1.0},
            {"team": "HIGH", "share": .8, "weight": 1.0},
        ])
        scale, above = receiving_share_scale(shares)
        normalized = shares["share"] * shares["weight"] * scale
        self.assertAlmostEqual(normalized[shares.team == "LOW"].sum(), .4)
        self.assertAlmostEqual(normalized[shares.team == "HIGH"].sum(), 1.2)
        self.assertFalse(bool(above[shares.team == "LOW"].iloc[0]))
        self.assertTrue(bool(above[shares.team == "HIGH"].iloc[0]))

    def test_canonical_season_total_uses_reconciled_volume_exposure(self):
        row = pd.DataFrame([{
            "pred_pg": 30.0, "pred_pg_low": 20.0, "pred_pg_high": 40.0,
            "projected_games": 10.0, "projected_volume_games": 4.0,
        }])
        out = add_projected_season_totals(row).iloc[0]
        self.assertAlmostEqual(out["pred_season"], 120.0)
        self.assertAlmostEqual(out["pred_season_low"], 80.0)
        self.assertAlmostEqual(out["pred_season_high"], 160.0)

    def test_negative_fantasy_low_is_floored_and_audited(self):
        rows = [
            self._projection_row("passing_yards", 10.0, 0.0, 20.0),
            self._projection_row("interceptions", 1.0, 0.0, 2.0),
        ]
        out = compute_fantasy_points(pd.DataFrame(rows)).iloc[0]
        self.assertLess(out["fantasy_pts_low_raw"], 0)
        self.assertEqual(out["fantasy_pts_low"], 0)
        self.assertTrue(out["fantasy_low_floor_applied"])
        raw = compute_fantasy_points(pd.DataFrame(rows), floor_low_at_zero=False).iloc[0]
        self.assertEqual(raw["fantasy_pts_low"], raw["fantasy_pts_low_raw"])

    def test_sleeper_comparison_is_season_total_and_invalidates_fake_rate(self):
        ours = pd.DataFrame([{
            "player_id": "p1", "position": "QB", "display_name": "Player One",
            "fantasy_pts": 10.0, "fantasy_pts_season": 100.0,
            "projected_games": 10.0, "projected_volume_games": 8.0,
            "pg_passing_yards": 200.0,
        }])
        sleeper_row = {
            "sleeper_id": "s1", "player_id": "p1", "position": "QB",
            "team": "TST", "name_key": "player one", "reported_gp": 18,
            "rate_denominator_valid": False, "pts_half_ppr_season": 90.0,
            "pts_half_ppr_pg": float("nan"),
        }
        for stat in sleeper_compare.STAT_MAP.values():
            sleeper_row[f"{stat}_season"] = 1800.0 if stat == "passing_yards" else 0.0
            sleeper_row[stat] = float("nan")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fantasy.csv")
            ours.to_csv(path, index=False)
            with patch.object(sleeper_compare, "build_sleeper_comparison_table", return_value=pd.DataFrame([sleeper_row])):
                out = sleeper_compare.compare(path, 2026).iloc[0]
        self.assertAlmostEqual(out["fantasy_pts_season_delta"], 10.0)
        self.assertAlmostEqual(out["our_passing_yards_season"], 1600.0)
        self.assertAlmostEqual(out["passing_yards_season_delta"], -200.0)
        self.assertTrue(pd.isna(out["sleeper_fantasy_pts"]))
        self.assertTrue(pd.isna(out["fantasy_pts_delta"]))

    def test_bare_output_filename_has_a_real_parent(self):
        _ensure_output_parent("projections.csv")


if __name__ == "__main__":
    unittest.main()
