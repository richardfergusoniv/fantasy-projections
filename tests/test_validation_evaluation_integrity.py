import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.comparison import sleeper_compare
from src.projection import backtest, corrections
from src.projection.fantasy_points import compute_fantasy_points
from src.projection.transitions import AVAILABILITY_FEATURES


class _AvailabilityModel:
    def predict(self, frame):
        return pd.to_numeric(frame["games_played"], errors="coerce").to_numpy()


class _Response:
    def __init__(self, payload):
        self.payload = payload
        self.status_checked = False

    def raise_for_status(self):
        self.status_checked = True

    def json(self):
        return self.payload


class ValidationEvaluationIntegrityTests(unittest.TestCase):
    def test_projected_participation_weight_does_not_read_held_out_games(self):
        base = {feature: 0.0 for feature in AVAILABILITY_FEATURES[:-1]}
        base.update(player_id="p1", games_played=8.0, games_played_to=1.0)
        test_a = pd.DataFrame([base])
        test_b = test_a.copy()
        test_b["games_played_to"] = 17.0

        def attach(frame, season):
            out = frame.copy()
            out["target_depth_rank"] = 1.0
            return out

        with patch.object(corrections, "fit_availability", return_value=(_AvailabilityModel(), 10)), \
             patch.object(corrections, "attach_availability_depth_rank", side_effect=attach):
            a = corrections.projected_participation_weight(
                pd.DataFrame(), test_a, "WR", [(2021, 2022)], (2022, 2023))
            b = corrections.projected_participation_weight(
                pd.DataFrame(), test_b, "WR", [(2021, 2022)], (2022, 2023))
        np.testing.assert_allclose(a, b)
        self.assertAlmostEqual(float(a[0]), 8 / 17)

    def test_availability_reports_all_returners_and_attrition_separately(self):
        features = {feature: [0.0, 0.0] for feature in AVAILABILITY_FEATURES}
        features.update({
            "player_id": ["returner", "gone"],
            "games_played_to": [10.0, 0.0],
            "naive_pred": [12.0, 12.0],
            "played_again": [True, False],
        })
        test = pd.DataFrame(features)

        class Model:
            def fit(self, x, y):
                return self

            def predict(self, x):
                return np.array([9.0, 3.0])[:len(x)]

        def pairs(_feat, _position, season_pairs):
            return test.copy()

        with patch.object(backtest, "build_availability_pairs", side_effect=pairs), \
             patch.object(backtest, "LGBMRegressor", return_value=Model()):
            out = backtest.backtest_availability(
                pd.DataFrame(), conn=None, test_pairs=[(2022, 2023)])
        scopes = set(out["scope"])
        self.assertIn("all_source_players", scopes)
        self.assertIn("returning_players_outcome_stratum", scopes)
        self.assertIn("attrition_outcome_stratum", scopes)
        all_rows = out[out["scope"] == "all_source_players"]
        self.assertTrue((all_rows["n_test"] == 2).all())

    def test_sleeper_name_collision_requires_unique_team_resolution(self):
        ours = pd.DataFrame([
            {"player_id": "ours-a", "display_name": "Same Name", "team": "A",
             "position": "WR", "season": 2026, "fantasy_pts": 10.0,
             "fantasy_pts_season": 100.0, "projected_games": 10.0},
            {"player_id": "ours-c", "display_name": "Same Name", "team": "C",
             "position": "WR", "season": 2026, "fantasy_pts": 10.0,
             "fantasy_pts_season": 100.0, "projected_games": 10.0},
        ])
        sleeper = pd.DataFrame([
            {"sleeper_id": "s-a", "player_id": None, "sleeper_name": "Same Name",
             "name_key": "same name", "sleeper_team": "A", "position": "WR",
             "pts_half_ppr_season": 90.0, "pts_half_ppr_pg": np.nan,
             "reported_gp": 18, "rate_denominator_valid": False},
            {"sleeper_id": "s-b", "player_id": None, "sleeper_name": "Same Name",
             "name_key": "same name", "sleeper_team": "B", "position": "WR",
             "pts_half_ppr_season": 80.0, "pts_half_ppr_pg": np.nan,
             "reported_gp": 18, "rate_denominator_valid": False},
        ])
        for stat in sleeper_compare.STAT_MAP.values():
            sleeper[f"{stat}_season"] = 0.0
            sleeper[stat] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ours.csv")
            ours.to_csv(path, index=False)
            with patch.object(
                    sleeper_compare, "build_sleeper_comparison_table",
                    return_value=sleeper):
                out = sleeper_compare.compare(path, 2026).set_index("player_id")
        self.assertEqual(out.loc["ours-a", "sleeper_id"], "s-a")
        self.assertEqual(out.loc["ours-a", "match_method"], "name")
        self.assertTrue(bool(out.loc["ours-a", "name_team_disambiguated"]))
        self.assertFalse(bool(out.loc["ours-c", "matched_sleeper"]))
        self.assertTrue(bool(out.loc["ours-c", "match_collision"]))

    def test_sleeper_snapshot_has_endpoint_timestamp_and_hash(self):
        response = _Response({"s1": {"gp": 18}, "bad": "not-a-dict"})
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(sleeper_compare.requests, "get", return_value=response):
            out = sleeper_compare.fetch_sleeper_season_projections(
                2026, snapshot_dir=tmp)
            metadata_files = [name for name in os.listdir(tmp)
                              if name.endswith(".metadata.json")]
            self.assertEqual(len(metadata_files), 1)
            with open(os.path.join(tmp, metadata_files[0]), encoding="utf-8") as fh:
                metadata = json.load(fh)
        self.assertTrue(response.status_checked)
        self.assertEqual(len(out), 1)
        self.assertIn("fetched_at", metadata)
        self.assertIn("endpoint", metadata)
        self.assertEqual(len(metadata["sha256"]), 64)

    def test_fantasy_output_preserves_normalization_provenance(self):
        rows = []
        for stat, point, low, high, scale in [
            ("passing_yards", 250.0, 200.0, 300.0, 1.2),
            ("passing_tds", 2.0, 1.0, 3.0, np.nan),
            ("interceptions", 1.0, 0.0, 2.0, np.nan),
        ]:
            rows.append({
                "player_id": "q", "position": "QB", "season": 2026,
                "display_name": "Quarterback", "team": "TST", "stat": stat,
                "pred_pg": point, "pred_pg_low": low, "pred_pg_high": high,
                "interval_low_n_flag": False, "stat_constraint_applied": False,
                "projected_games": 10.0, "projected_volume_games": 10.0,
                "projected_games_raw": 8.0,
                "team_qb_raw_appearance_games": 8.0,
                "team_qb_volume_allocation_direction": "upward",
                "team_qb_roster_resolved": True,
                "qb_volume_games_scale": 1.25,
                "qb_volume_allocation_adjusted": True,
                "team_qb_attempt_anchor_fully_allocated": True,
                "team_passing_volume_scale": scale,
                "team_pass_attempts_pg_pred": 34.0,
                "team_passing_yards_pg_pred": 250.0,
                "team_carries_pg_pred": 26.0,
                "team_rushing_yards_pg_pred": 115.0,
                "team_anchor_source_season": 2025,
                "team_anchor_lag_team": "TST",
                "team_anchor_provenance": "canonical_source_team_frame",
                "team_unmodeled_qb_attempts_season": 17.0,
                "team_unmodeled_receiving_yards_season": 125.0,
                "receiving_share_capped": False,
                "receiving_share_normalized": False,
            })
        out = compute_fantasy_points(pd.DataFrame(rows)).iloc[0]
        self.assertAlmostEqual(out["normalization_scale_passing_yards"], 1.2)
        self.assertAlmostEqual(out["team_pass_attempts_pg_pred"], 34.0)
        self.assertAlmostEqual(out["team_passing_yards_pg_pred"], 250.0)
        self.assertAlmostEqual(out["team_carries_pg_pred"], 26.0)
        self.assertAlmostEqual(out["team_rushing_yards_pg_pred"], 115.0)
        self.assertEqual(out["team_anchor_provenance"], "canonical_source_team_frame")
        self.assertAlmostEqual(out["team_unmodeled_qb_attempts_season"], 17.0)
        self.assertAlmostEqual(out["team_unmodeled_receiving_yards_season"], 125.0)
        self.assertAlmostEqual(out["projected_games_raw"], 8.0)
        self.assertEqual(out["team_qb_volume_allocation_direction"], "upward")
        self.assertAlmostEqual(out["qb_volume_games_scale"], 1.25)
        self.assertTrue(bool(out["qb_volume_allocation_adjusted"]))

    def test_forward_interval_coverage_uses_only_earlier_test_seasons(self):
        residuals = pd.DataFrame([
            {"position": "WR", "stat": "targets", "test_season": 2023,
             "pred": 5.0, "actual": 4.0, "resid": -1.0},
            {"position": "WR", "stat": "targets", "test_season": 2023,
             "pred": 5.0, "actual": 6.0, "resid": 1.0},
            {"position": "WR", "stat": "targets", "test_season": 2024,
             "pred": 5.0, "actual": 5.0, "resid": 0.0},
            {"position": "WR", "stat": "targets", "test_season": 2025,
             "pred": 5.0, "actual": 5.0, "resid": 0.0},
        ])
        with patch.object(backtest, "rolling_residual_rows", return_value=residuals):
            out = backtest.forward_interval_coverage(pd.DataFrame())
        self.assertEqual(list(out["test_season"]), [2024, 2025])
        self.assertEqual(out.loc[out["test_season"] == 2024,
                                 "n_calibration"].iloc[0], 2)
        self.assertEqual(out.loc[out["test_season"] == 2025,
                                 "n_calibration"].iloc[0], 3)


if __name__ == "__main__":
    unittest.main()
