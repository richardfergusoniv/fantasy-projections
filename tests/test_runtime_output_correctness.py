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
    NAMED_RUSH_COVERAGE,
    QB_ATTEMPTS_PER_VOLUME_GAME_MAX,
    REPLACEMENT_POSITIONS,
    RUSH_ATTEMPTS_PER_APPEARANCE_MAX,
    USAGE_SHARE_BLEND_W,
    USAGE_SHARE_CURATED_W,
    _ensure_output_parent,
    _warn_board_level_allocation,
    apply_usage_share_prior,
    add_projected_season_totals,
    add_team_pass_catch_coherence_flag,
    normalize_team_passing_volume,
    normalize_team_rushing_volume,
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

    def test_coherence_includes_explicit_residuals_without_losing_series_names(self):
        df = pd.DataFrame([
            {"player_id": "q1", "team": "TST", "position": "QB", "stat": "passing_yards",
             "pred_pg": 100.0, "projected_games": 10.0, "projected_volume_games": 10.0,
             "team_unmodeled_qb_passing_yards_season": 700.0,
             "team_unmodeled_receiving_yards_season": 850.0},
            {"player_id": "w1", "team": "TST", "position": "WR", "stat": "receiving_yards",
             "pred_pg": 50.0, "projected_games": 17.0, "projected_volume_games": 17.0,
             "team_unmodeled_qb_passing_yards_season": 700.0,
             "team_unmodeled_receiving_yards_season": 850.0},
        ])
        out = add_team_pass_catch_coherence_flag(df)
        self.assertAlmostEqual(out["team_pass_catch_ratio"].iloc[0], 1.0)
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
        residual = out.drop_duplicates("team").set_index("team")["team_unmodeled_qb_volume_games"]
        self.assertAlmostEqual(residual["TEN"], 0.0)
        self.assertAlmostEqual(residual["SEV"], 0.0)
        self.assertAlmostEqual(residual["TWF"], 0.0)
        self.assertEqual(
            out.drop_duplicates("team").set_index("team")[
                "team_qb_volume_allocation_direction"
            ].to_dict(),
            {"TEN": "upward", "SEV": "exact", "TWF": "downward"},
        )
        self.assertTrue((out["projected_games_raw"] == out["projected_games"]).all())
        self.assertTrue(out[out.team == "TEN"]["qb_volume_allocation_adjusted"].all())
        self.assertFalse(out[out.team == "SEV"]["qb_volume_allocation_adjusted"].any())
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
        self.assertAlmostEqual(out["projected_volume_games"].sum(), 17.0)
        self.assertGreater(out.loc["b1", "projected_volume_games"],
                           out.loc["b2", "projected_volume_games"])
        self.assertEqual(out.loc["s", "projected_games_raw"], 14.0)
        self.assertEqual(out.loc["b1", "projected_games_raw"], 8.0)
        self.assertEqual(out.loc["b2", "projected_games_raw"], 4.0)

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
        rows.append({
            "player_id": "w", "team": "CLE", "position": "WR", "stat": "receiving_yards",
            "pred_pg": 180.0, "pred_pg_low": 100.0, "pred_pg_high": 240.0,
            "projected_games": 17.0, "projected_volume_games": 17.0,
            "team_pass_attempts_pg_pred": 34.0, "team_passing_yards_pg_pred": 220.0,
        })
        out = normalize_team_passing_volume(pd.DataFrame(rows))
        attempts = out[out["stat"] == "attempts"].iloc[0]
        yards = out[out["stat"] == "passing_yards"].iloc[0]
        self.assertAlmostEqual(attempts["pred_pg"], 34.0)
        self.assertAlmostEqual(yards["pred_pg"], 220.0)

    def test_team_yard_anchor_enforces_identity_and_preserves_pre_ratio(self):
        rows = [
            {"player_id": "q", "team": "TST", "position": "QB", "stat": "attempts",
             "pred_pg": 34.0, "pred_pg_low": 25.0, "pred_pg_high": 40.0,
             "projected_games": 17.0, "projected_volume_games": 17.0,
             "team_pass_attempts_pg_pred": 34.0, "team_passing_yards_pg_pred": 240.0},
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
        qb_yards = out[(out["position"] == "QB") & (out["stat"] == "passing_yards")].iloc[0]
        self.assertAlmostEqual(qb_yards["pred_pg"], 240.0)
        self.assertAlmostEqual(wr["pred_pg"], 100.0)
        self.assertAlmostEqual(wr["team_unmodeled_receiving_yards_season"], 140.0 * 17.0)
        self.assertAlmostEqual(wr["team_pass_catch_ratio_pre_normalization"], 100 / 240)
        self.assertTrue(wr["team_pass_catch_pre_normalization_flag"])

    def test_named_qb_attempt_rate_is_bounded_and_anchor_is_fully_allocated(self):
        rows = []
        for stat, rate in (("attempts", 20.0), ("completions", 12.0),
                           ("passing_yards", 140.0)):
            rows.append({
                "player_id": "q", "team": "TST", "position": "QB", "stat": stat,
                "pred_pg": rate, "pred_pg_low": rate * .5, "pred_pg_high": rate * 1.5,
                "projected_games": 10.0,
                "team_pass_attempts_pg_pred": 42.0, "team_passing_yards_pg_pred": 300.0,
            })
        rows.append({
            "player_id": "w", "team": "TST", "position": "WR", "stat": "receiving_yards",
            "pred_pg": 200.0, "pred_pg_low": 100.0, "pred_pg_high": 250.0,
            "projected_games": 17.0,
            "team_pass_attempts_pg_pred": 42.0, "team_passing_yards_pg_pred": 300.0,
        })
        reconciled = reconcile_qb_projected_volume_games(pd.DataFrame(rows))
        out = normalize_team_passing_volume(reconciled)
        attempts = out[out["stat"] == "attempts"].iloc[0]
        self.assertLessEqual(attempts["pred_pg"], QB_ATTEMPTS_PER_VOLUME_GAME_MAX)
        self.assertAlmostEqual(attempts["projected_volume_games"], 17.0)
        self.assertAlmostEqual(attempts["pred_pg"], 42.0)
        self.assertAlmostEqual(attempts["team_unmodeled_qb_attempts_season"], 0.0)
        self.assertTrue(attempts["team_qb_attempt_anchor_fully_allocated"])
        self.assertAlmostEqual(
            attempts["pred_pg"] * attempts["projected_volume_games"], 42.0 * 17.0
        )

    def test_impossible_resolved_attempt_anchor_fails_loudly(self):
        rows = [
            {"player_id": "q", "team": "TST", "position": "QB", "stat": "attempts",
             "pred_pg": 20.0, "pred_pg_low": 10.0, "pred_pg_high": 30.0,
             "projected_games": 17.0, "projected_volume_games": 17.0,
             "team_qb_roster_resolved": True,
             "team_pass_attempts_pg_pred": 43.0, "team_passing_yards_pg_pred": 300.0},
            {"player_id": "q", "team": "TST", "position": "QB", "stat": "passing_yards",
             "pred_pg": 200.0, "pred_pg_low": 100.0, "pred_pg_high": 300.0,
             "projected_games": 17.0, "projected_volume_games": 17.0,
             "team_qb_roster_resolved": True,
             "team_pass_attempts_pg_pred": 43.0, "team_passing_yards_pg_pred": 300.0},
            {"player_id": "w", "team": "TST", "position": "WR", "stat": "receiving_yards",
             "pred_pg": 200.0, "pred_pg_low": 100.0, "pred_pg_high": 250.0,
             "projected_games": 17.0, "projected_volume_games": 17.0,
             "team_qb_roster_resolved": True,
             "team_pass_attempts_pg_pred": 43.0, "team_passing_yards_pg_pred": 300.0},
        ]
        with self.assertRaisesRegex(ValueError, "could not meet the team attempt anchor"):
            normalize_team_passing_volume(pd.DataFrame(rows))

    def test_receiver_shortfall_does_not_inflate_named_players(self):
        rows = [
            {"player_id": "q", "team": "TST", "position": "QB", "stat": "attempts",
             "pred_pg": 34.0, "pred_pg_low": 30.0, "pred_pg_high": 38.0,
             "projected_games": 17.0, "projected_volume_games": 17.0,
             "team_pass_attempts_pg_pred": 34.0, "team_passing_yards_pg_pred": 240.0},
            {"player_id": "q", "team": "TST", "position": "QB", "stat": "passing_yards",
             "pred_pg": 240.0, "pred_pg_low": 200.0, "pred_pg_high": 280.0,
             "projected_games": 17.0, "projected_volume_games": 17.0,
             "team_pass_attempts_pg_pred": 34.0, "team_passing_yards_pg_pred": 240.0},
            {"player_id": "w", "team": "TST", "position": "WR", "stat": "receiving_yards",
             "pred_pg": 80.0, "pred_pg_low": 60.0, "pred_pg_high": 100.0,
             "projected_games": 17.0, "projected_volume_games": 17.0,
             "team_pass_attempts_pg_pred": 34.0, "team_passing_yards_pg_pred": 240.0},
        ]
        out = normalize_team_passing_volume(pd.DataFrame(rows))
        wr = out[out["position"] == "WR"].iloc[0]
        self.assertAlmostEqual(wr["pred_pg"], 80.0)
        self.assertAlmostEqual(wr["team_unmodeled_receiving_yards_season"], 160.0 * 17.0)

    def _carry_row(self, player_id, team, position, pred_pg, anchor_carries,
                   anchor_yards, games=17.0):
        return {
            "player_id": player_id, "team": team, "position": position,
            "stat": "carries", "pred_pg": pred_pg,
            "pred_pg_low": pred_pg * .5, "pred_pg_high": pred_pg * 1.5,
            "projected_games": games, "projected_volume_games": games,
            "team_carries_pg_pred": anchor_carries,
            "team_rushing_yards_pg_pred": anchor_yards,
        }

    def test_rushing_shortfall_fills_only_to_measured_coverage(self):
        # Anchor is 27 carries/game but the only modeled back is at 15. He
        # may be filled up to the historically supported share of the
        # anchor - not to the whole thing, which is what pinned a lead back
        # to his position ceiling when his backfield was under-represented.
        rows = [self._carry_row("rb1", "TST", "RB", 15.0, 27.0, 120.0)]
        out = normalize_team_rushing_volume(pd.DataFrame(rows))
        rb = out[out["position"] == "RB"].iloc[0]
        self.assertAlmostEqual(rb["pred_pg"], 27.0 * NAMED_RUSH_COVERAGE)
        self.assertLess(rb["pred_pg"], 27.0)
        self.assertAlmostEqual(
            rb["team_unmodeled_carries_season"],
            27.0 * 17.0 * (1 - NAMED_RUSH_COVERAGE),
        )

    def test_rushing_room_above_coverage_is_left_alone(self):
        # Coverage is a floor to fill up to, never a ceiling to cut down
        # to: a room already projecting above it keeps what it projects.
        rows = [
            self._carry_row("rb1", "TST", "RB", 16.0, 24.0, 110.0),
            self._carry_row("rb2", "TST", "RB", 6.0, 24.0, 110.0),
        ]
        out = normalize_team_rushing_volume(pd.DataFrame(rows))
        carries = out[out["stat"] == "carries"].set_index("player_id")
        self.assertGreater(22.0, 24.0 * NAMED_RUSH_COVERAGE)  # premise
        self.assertAlmostEqual(carries.loc["rb1", "pred_pg"], 16.0)
        self.assertAlmostEqual(carries.loc["rb2", "pred_pg"], 6.0)

    def test_rushing_overflow_still_scales_named_players_down(self):
        # The downward half is the half that was always correct: two backs
        # projected past the anchor get scaled back into it.
        rows = [
            self._carry_row("rb1", "TST", "RB", 20.0, 24.0, 110.0),
            self._carry_row("rb2", "TST", "RB", 16.0, 24.0, 110.0),
        ]
        out = normalize_team_rushing_volume(pd.DataFrame(rows))
        carries = out[out["stat"] == "carries"]
        allocated = (carries["pred_pg"] * carries["projected_volume_games"]).sum()
        self.assertAlmostEqual(allocated, 24.0 * 17.0)
        self.assertAlmostEqual(carries["team_unmodeled_carries_season"].iloc[0], 0.0)
        self.assertTrue((carries["team_rushing_volume_scale"] < 1.0).all())
        # Pecking order survives the scale-down.
        by_player = carries.set_index("player_id")["pred_pg"]
        self.assertGreater(by_player["rb1"], by_player["rb2"])

    def test_rushing_shortfall_leaves_capacity_ceiling_unreached(self):
        # The regression that motivated this: a lone lead back pinned to
        # exactly RUSH_ATTEMPTS_PER_APPEARANCE_MAX because his committee
        # partner had no row at all. Filling to measured coverage rather
        # than to the full anchor keeps him off the ceiling.
        rows = [self._carry_row("rb1", "TST", "RB", 14.5, 28.0, 125.0)]
        out = normalize_team_rushing_volume(pd.DataFrame(rows))
        rb = out[out["position"] == "RB"].iloc[0]
        self.assertLess(rb["pred_pg"], RUSH_ATTEMPTS_PER_APPEARANCE_MAX["RB"])
        self.assertAlmostEqual(rb["pred_pg"], 28.0 * NAMED_RUSH_COVERAGE)

    def test_replacement_prior_holds_a_missing_backs_share_open(self):
        # A charted committee back with no row at all does not merely go
        # unreported - the reconciler hands his share to whoever is present.
        # With a floor row he holds it himself.
        without = [self._carry_row("rb1", "TST", "RB", 15.0, 27.0, 120.0)]
        alone = normalize_team_rushing_volume(pd.DataFrame(without))
        lead_alone = alone[alone.player_id == "rb1"].iloc[0]["pred_pg"]

        with_floor = without + [self._carry_row("rb2", "TST", "RB", 7.0, 27.0, 120.0)]
        both = normalize_team_rushing_volume(pd.DataFrame(with_floor))
        lead_both = both[both.player_id == "rb1"].iloc[0]["pred_pg"]
        floor = both[both.player_id == "rb2"].iloc[0]["pred_pg"]

        self.assertLess(lead_both, lead_alone)
        self.assertGreater(floor, 0.0)

    def test_replacement_positions_exclude_qb(self):
        # A QB row is a claim against the room's fixed 17-game budget, not an
        # extra claimant on a volume pool - filling one from a band mean
        # takes exposure straight off the starter.
        self.assertNotIn("QB", REPLACEMENT_POSITIONS)
        self.assertEqual(set(REPLACEMENT_POSITIONS), {"RB", "WR", "TE"})

    def _wr_room(self):
        rows = []
        for pid, rate, rank in (("a", 8.0, 2.0), ("b", 6.0, 1.0), ("c", 2.0, 3.0)):
            rows.append({
                "player_id": pid, "team": "TST", "position": "WR", "stat": "targets",
                "pred_pg": rate, "pred_pg_low": rate * .5, "pred_pg_high": rate * 1.5,
                "projected_games": 17.0, "projected_volume_games": 17.0,
                "nfl_depth_rank": rank,
            })
        priors = pd.DataFrame(
            {"target_share": [0.15, 0.06, 0.03], "carry_share": [0.0, 0.0, 0.0]},
            index=pd.MultiIndex.from_tuples(
                [("WR", 1), ("WR", 2), ("WR", 3)], names=["position", "rank"]),
        )
        return pd.DataFrame(rows), priors

    def test_unreviewed_usage_prior_never_moves_a_projection(self):
        # An unreviewed default is a starting point for research, not a claim.
        df, priors = self._wr_room()
        chart = pd.DataFrame([
            dict(gsis_id="a", position="WR", usage_share_prior=0.06,
                 usage_share_reviewed=False),
            dict(gsis_id="b", position="WR", usage_share_prior=0.15,
                 usage_share_reviewed=False),
        ])
        out = apply_usage_share_prior(df, priors, chart, weight=0.0)
        pd.testing.assert_series_equal(out["pred_pg"], df["pred_pg"])

    def test_reviewed_usage_prior_reorders_the_room(self):
        # The Dallas case: the models rank 'a' first on prior-season usage,
        # research says 'b' is the real alpha.
        df, priors = self._wr_room()
        chart = pd.DataFrame([
            dict(gsis_id="a", position="WR", usage_share_prior=0.06,
                 usage_share_reviewed=True),
            dict(gsis_id="b", position="WR", usage_share_prior=0.15,
                 usage_share_reviewed=True),
        ])
        out = apply_usage_share_prior(df, priors, chart, weight=0.0).set_index("player_id")
        self.assertGreater(out.loc["b", "pred_pg"], out.loc["a", "pred_pg"])
        # Unranked/unreviewed teammate is untouched.
        self.assertAlmostEqual(out.loc["c", "pred_pg"], 2.0)

    def test_usage_prior_preserves_group_volume(self):
        # Reordering redistributes between teammates; it never invents or
        # destroys team volume, which the anchors downstream depend on.
        df, priors = self._wr_room()
        chart = pd.DataFrame([
            dict(gsis_id="a", position="WR", usage_share_prior=0.06,
                 usage_share_reviewed=True),
            dict(gsis_id="b", position="WR", usage_share_prior=0.15,
                 usage_share_reviewed=True),
        ])
        out = apply_usage_share_prior(df, priors, chart, weight=0.0)
        moved = out[out["usage_share_blend_factor"].notna()]
        before = df.set_index("player_id").loc[moved["player_id"], "pred_pg"].sum()
        self.assertAlmostEqual(moved["pred_pg"].sum(), before)

    def test_fitted_rank_prior_ships_disabled(self):
        # Measured against real 2025 outcomes and it lost; kept at 0 until
        # evidence says otherwise.
        self.assertEqual(USAGE_SHARE_BLEND_W, 0.0)
        self.assertGreater(USAGE_SHARE_CURATED_W, 0.0)

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
