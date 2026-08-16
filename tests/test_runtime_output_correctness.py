import contextlib
import io
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.comparison import sleeper_compare
from src.projection.fantasy_points import DESCRIPTIVE_COLS, compute_fantasy_points
from src.projection.predict import (
    RUSH_ATTEMPTS_PER_APPEARANCE_MAX,
    _ensure_output_parent,
    _warn_board_level_allocation,
    apply_curated_availability_override,
    apply_full_season_games_baseline,
    apply_status_overrides,
    add_projected_season_totals,
    enforce_availability_chart_review,
    reconcile_stat_constraints,
)
from src.projection.transitions import SEASON_GAMES, receiving_share_scale


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

    def test_status_override_zero_preserves_raw_games(self):
        df = pd.DataFrame([
            {"player_id": "pear", "projected_games": 13.2},
            {"player_id": "ok", "projected_games": 15.0},
        ])
        overrides = pd.DataFrame([{
            "gsis_id": "pear", "mode": "zero", "projected_games": None,
        }])
        out = apply_status_overrides(df, overrides).set_index("player_id")
        self.assertEqual(out.loc["pear", "projected_games"], 0.0)
        self.assertAlmostEqual(out.loc["pear", "projected_games_raw"], 13.2)
        self.assertAlmostEqual(out.loc["ok", "projected_games"], 15.0)
        self.assertTrue(out.loc["pear", "status_override_applied"])
        self.assertFalse(out.loc["ok", "status_override_applied"])

    def test_status_override_cap(self):
        df = pd.DataFrame([{"player_id": "x", "projected_games": 14.0}])
        overrides = pd.DataFrame([{
            "gsis_id": "x", "mode": "cap", "projected_games": 6.0,
        }])
        out = apply_status_overrides(df, overrides).iloc[0]
        self.assertAlmostEqual(out["projected_games"], 6.0)
        self.assertAlmostEqual(out["projected_games_raw"], 14.0)

    def test_full_season_baseline_keeps_gate_a_in_raw(self):
        df = pd.DataFrame([
            {"player_id": "a", "projected_games": 13.0},
            {"player_id": "b", "projected_games": 15.5},
        ])
        out = apply_full_season_games_baseline(df).set_index("player_id")
        self.assertEqual(out.loc["a", "projected_games"], SEASON_GAMES)
        self.assertEqual(out.loc["b", "projected_games"], SEASON_GAMES)
        self.assertAlmostEqual(out.loc["a", "projected_games_raw"], 13.0)
        self.assertAlmostEqual(out.loc["b", "projected_games_raw"], 15.5)

    def test_curated_availability_override_forces_off_chart(self):
        base = pd.DataFrame([
            {"player_id": "on", "position": "WR", "target_depth_rank": 2.0},
            {"player_id": "off", "position": "WR", "target_depth_rank": 1.0},
            {"player_id": "miss", "position": "WR", "target_depth_rank": float("nan")},
        ])
        chart = pd.DataFrame([
            {"gsis_id": "on", "position": "WR", "depth_rank": 1},
            {"gsis_id": "miss", "position": "WR", "depth_rank": 2},
        ])
        out = apply_curated_availability_override(base, chart).set_index("player_id")
        self.assertAlmostEqual(out.loc["on", "target_depth_rank"], 2.0)
        self.assertTrue(pd.isna(out.loc["off", "target_depth_rank"]))
        self.assertAlmostEqual(out.loc["miss", "target_depth_rank"], 2.0)

    def test_reverse_chart_conflict_warns_without_status_override(self):
        base = pd.DataFrame([{"player_id": "pear", "position": "WR"}])
        chart = pd.DataFrame([
            {"gsis_id": "other", "position": "WR", "depth_rank": 1,
             "player_name": "Other", "role": "starter"},
        ])
        nfl = pd.DataFrame([{
            "player_id": "pear", "position": "WR", "depth_rank": 2.0,
            "availability_rank": 1.0, "full_name": "Ricky Pearsall",
            "team": "SF", "source": "test",
        }])
        err = io.StringIO()
        with patch("src.projection.depth_gating.load_preseason_depth_chart", return_value=nfl):
            with contextlib.redirect_stderr(err):
                enforce_availability_chart_review(
                    base, chart, overrides=pd.DataFrame(), target_season=2026)
        self.assertIn("nflverse-on/curated-off", err.getvalue())

    def test_reverse_chart_conflict_does_not_require_status_override(self):
        base = pd.DataFrame([{"player_id": "pear", "position": "WR"}])
        chart = pd.DataFrame([
            {"gsis_id": "other", "position": "WR", "depth_rank": 1,
             "player_name": "Other", "role": "starter"},
        ])
        nfl = pd.DataFrame([{
            "player_id": "pear", "position": "WR", "depth_rank": 2.0,
            "availability_rank": 1.0, "full_name": "Ricky Pearsall",
            "team": "SF", "source": "test",
        }])
        with patch("src.projection.depth_gating.load_preseason_depth_chart", return_value=nfl):
            enforce_availability_chart_review(
                base, chart, overrides=pd.DataFrame(), target_season=2026)

    def test_wr_usage_share_defaults_are_unreviewed(self):
        """Most WR slot priors stay unreviewed; only curated rooms flip True."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "src", "depth_chart", "starters_2026.csv",
        )
        dc = pd.read_csv(path)
        wr = dc[dc["position"] == "WR"]
        reviewed = wr["usage_share_reviewed"].astype(str).str.strip().str.lower().isin(
            ("true", "1", "yes")
        )
        # Phase C5: a few high-leverage rooms are researched; the rest stay defaults.
        self.assertGreaterEqual(int((~reviewed).sum()), 80)
        self.assertLessEqual(int(reviewed.sum()), 12)
        curated_teams = set(wr.loc[reviewed, "team"])
        self.assertIn("DAL", curated_teams)
        dal = wr[wr["team"] == "DAL"].sort_values("depth_rank")
        self.assertGreater(
            float(dal.iloc[0]["usage_share_prior"]),
            float(dal.iloc[1]["usage_share_prior"]),
        )
        self.assertEqual(wr["team"].nunique(), 32)
        # Formation-order defaults: higher depth_rank → lower slot prior.
        for _, room in wr.groupby("team"):
            ordered = room.sort_values("depth_rank")
            priors = ordered["usage_share_prior"].astype(float).tolist()
            self.assertEqual(priors, sorted(priors, reverse=True))
        # Ourlads columns: depth_rank 1/2/3 ↔ LWR/RWR/SWR on every WR.
        self.assertTrue((wr.loc[wr["depth_rank"] == 1, "formation_role"] == "LWR").all())
        self.assertTrue((wr.loc[wr["depth_rank"] == 2, "formation_role"] == "RWR").all())
        self.assertTrue((wr.loc[wr["depth_rank"] == 3, "formation_role"] == "SWR").all())
        non_wr = dc.loc[dc["position"] != "WR", "formation_role"]
        blank = non_wr.isna() | non_wr.astype(str).str.strip().isin(("", "nan", "None"))
        self.assertTrue(blank.all())

    def _tripwire_conn(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("create table players (gsis_id text, display_name text)")
        conn.executemany("insert into players values (?, ?)", [
            ("vet", "Established Starter"), ("new", "Shiny Newcomer"),
            ("bell", "Bell Cow"), ("ghost", "Absent Backup"),
        ])
        conn.commit()
        return conn

    def _tripwire_row(self, player_id, position, stat, pred_pg, **kw):
        row = {
            "player_id": player_id, "team": "TST", "position": position, "stat": stat,
            "pred_pg": pred_pg, "projected_games": 17.0, "projected_volume_games": 17.0,
            "team_carries_pg_pred": 27.0, "source": "veteran_model", "team_changed": False,
        }
        row.update(kw)
        return row

    def _run_tripwire(self, rows, chart):
        conn = self._tripwire_conn()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            _warn_board_level_allocation(conn, pd.DataFrame(rows), chart)
        return err.getvalue()

    def test_tripwire_flags_a_rate_pinned_to_its_support_ceiling(self):
        rows = [self._tripwire_row(
            "bell", "RB", "carries", RUSH_ATTEMPTS_PER_APPEARANCE_MAX["RB"])]
        out = self._run_tripwire(rows, pd.DataFrame())
        self.assertIn("CAPPED", out)
        self.assertIn("Bell Cow", out)

    def test_tripwire_flags_a_curated_player_with_no_row(self):
        rows = [self._tripwire_row("vet", "WR", "targets", 5.0)]
        chart = pd.DataFrame([dict(gsis_id="ghost", position="QB", team="TST",
                                   player_name="Absent Backup", role="backup")])
        out = self._run_tripwire(rows, chart)
        self.assertIn("MISSING", out)
        self.assertIn("Absent Backup", out)

    def test_tripwire_flags_a_newcomer_passing_the_charted_starter(self):
        # The Makai Lemon shape: real vacancy, real eligibility, wrong
        # conclusion - invisible to any check that looks at one player.
        rows = [
            self._tripwire_row("vet", "WR", "targets", 5.0),
            self._tripwire_row("new", "WR", "targets", 9.0, source="rookie_rule"),
        ]
        chart = pd.DataFrame([dict(gsis_id="vet", position="WR", team="TST",
                                   player_name="Established Starter", role="starter")])
        out = self._run_tripwire(rows, chart)
        self.assertIn("NEWCOMER", out)
        self.assertIn("Shiny Newcomer", out)

    def test_tripwire_is_silent_on_a_healthy_board(self):
        rows = [
            self._tripwire_row("vet", "RB", "carries", 12.0),
            self._tripwire_row("new", "RB", "carries", 6.0),
        ]
        chart = pd.DataFrame([dict(gsis_id="vet", position="RB", team="TST",
                                   player_name="Established Starter", role="starter")])
        self.assertEqual(self._run_tripwire(rows, chart), "")

    def test_tripwire_never_changes_a_projection(self):
        rows = [self._tripwire_row(
            "bell", "RB", "carries", RUSH_ATTEMPTS_PER_APPEARANCE_MAX["RB"])]
        df = pd.DataFrame(rows)
        before = df.copy()
        conn = self._tripwire_conn()
        with contextlib.redirect_stderr(io.StringIO()):
            _warn_board_level_allocation(conn, df, pd.DataFrame())
        pd.testing.assert_frame_equal(df, before)

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

    def test_canonical_season_total_uses_projected_volume_games(self):
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
