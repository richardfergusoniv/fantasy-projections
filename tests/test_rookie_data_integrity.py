import sqlite3
import unittest

import numpy as np
import pandas as pd

from src.projection.data_prep import STAT_COLS, season_aggregate
from src.projection.rookies import (
    _effective_pick,
    combine_athletic_scores_by_pfr_id,
    fit_rookie_baselines,
    identify_target_season_rookie_class,
    load_combine_athletic_tier,
    predict_rookies,
    rookie_interval_ratios,
)
from src.projection.predict import (
    _apply_rookie_depth_rate_gating,
    _attach_rookie_depth_tier,
    _attach_rookie_intervals,
)
from src.projection.roster_moves import load_target_roster_map


class RookieDataIntegrityTests(unittest.TestCase):
    def test_refreshed_arizona_roster_code_normalizes_to_feature_code(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        pd.DataFrame([
            dict(player_id="cardinal", season=2026, team="AZ", status="ACT")
        ]).to_sql("seasonal_rosters", conn, index=False)

        roster = load_target_roster_map(conn, 2026)

        self.assertEqual(roster.loc["cardinal", "team"], "ARI")

    def test_curated_rookie_qb1_materializes_reconcile_starter_tier(self):
        rows = pd.DataFrame([
            dict(player_id="rookie", position="QB", nfl_depth_rank=6.0),
            dict(player_id="other", position="QB", nfl_depth_rank=5.0),
        ])
        chart = pd.DataFrame([
            dict(gsis_id="rookie", position="QB", depth_rank=1.0, role="starter")
        ])

        out = _attach_rookie_depth_tier(rows, chart).set_index("player_id")

        self.assertEqual(out.loc["rookie", "depth_tier"], 1.0)
        self.assertEqual(out.loc["rookie", "depth_tier_source"], "curated")
        self.assertEqual(out.loc["other", "depth_tier"], 4.0)
        self.assertEqual(out.loc["other", "depth_tier_source"], "nflverse")

    def test_rookie_interval_zero_mass_and_missing_cell_both_have_zero_floor(self):
        cohort = pd.DataFrame([
            dict(season=2024, position="QB", round_bucket="round_4_7",
                 attempts_per_elig=value, role_rate_eligible=True)
            for value in (0.0, 0.0, 0.0, 10.0)
        ])
        idx = pd.MultiIndex.from_tuples(
            [("QB", "round_4_7")], names=["position", "round_bucket"])
        baselines = pd.DataFrame([dict(attempts_per_elig=2.5)], index=idx)

        ratios = rookie_interval_ratios(cohort, baselines, [2024])
        self.assertEqual(ratios.loc[0, "ratio_low"], 0.0)

        rows = pd.DataFrame([dict(
            player_id="rook", position="QB", round_bucket="missing",
            stat="attempts", pred_pg=10.0,
        )])
        attached = _attach_rookie_intervals(rows, ratios)
        self.assertEqual(attached.loc[0, "pred_pg_low"], 0.0)
        self.assertEqual(attached.loc[0, "pred_pg_high"], 30.0)
        self.assertTrue(attached.loc[0, "interval_low_n_flag"])

    def test_rookie_interval_always_contains_point_prediction(self):
        rows = pd.DataFrame([dict(
            player_id="rook", position="WR", round_bucket="round_4_7",
            stat="receiving_tds", pred_pg=0.01,
        )])
        ratios = pd.DataFrame([dict(
            position="WR", round_bucket="round_4_7", stat="receiving_tds",
            n=40, ratio_low=0.0, ratio_high=0.0,
            interval_low_n_flag=False,
        )])

        attached = _attach_rookie_intervals(rows, ratios)

        self.assertEqual(attached.loc[0, "pred_pg_low"], 0.0)
        self.assertEqual(attached.loc[0, "pred_pg_high"], 0.01)

    def test_log_pick_rate_separates_players_inside_the_same_round_cell(self):
        cohort = pd.DataFrame([
            dict(player_id="early", season=2024, team="A", position="WR",
                 round_bucket="round_2_3", pick=33, games_played=10,
                 targets_per_elig=8.0, role_rate_eligible=True,
                 vacated_carry_share=0.2, vacated_target_share=0.2,
                 vacated_attempts_share=0.2),
            dict(player_id="late", season=2024, team="B", position="WR",
                 round_bucket="round_2_3", pick=96, games_played=10,
                 targets_per_elig=2.0, role_rate_eligible=True,
                 vacated_carry_share=0.2, vacated_target_share=0.2,
                 vacated_attempts_share=0.2),
        ])
        baselines = fit_rookie_baselines(cohort, [2024])
        target = pd.DataFrame([
            dict(player_id="p40", season=2025, team="A", position="WR",
                 round_bucket="round_2_3", pick=40, rookie_tier="drafted",
                 vacated_carry_share=0.2, vacated_target_share=0.2,
                 vacated_attempts_share=0.2, target_depth_rank=np.nan),
            dict(player_id="p90", season=2025, team="B", position="WR",
                 round_bucket="round_2_3", pick=90, rookie_tier="drafted",
                 vacated_carry_share=0.2, vacated_target_share=0.2,
                 vacated_attempts_share=0.2, target_depth_rank=np.nan),
        ])

        pred = predict_rookies(target, baselines, [2025]).set_index("player_id")

        self.assertGreater(pred.loc["p40", "targets_pg"], pred.loc["p90", "targets_pg"])
        self.assertEqual(_effective_pick(np.nan), 270.0)

    def test_athletic_tier_is_metadata_not_a_volume_multiplier(self):
        idx = pd.MultiIndex.from_tuples(
            [("WR", "round_1")], names=["position", "round_bucket"])
        baselines = pd.DataFrame([dict(
            targets_per_elig=6.0,
            vacated_carry_share=0.2,
            vacated_target_share=0.2,
            vacated_attempts_share=0.2,
            mean_games_played=10.0,
            n_train_rookies=20,
        )], index=idx)
        common = dict(
            season=2026, team="A", position="WR", round_bucket="round_1",
            pick=10, rookie_tier="drafted", vacated_carry_share=0.2,
            vacated_target_share=0.2, vacated_attempts_share=0.2,
            target_depth_rank=np.nan,
        )
        target = pd.DataFrame([
            dict(common, player_id="fast", athletic_tier="above_median"),
            dict(common, player_id="slow", athletic_tier="below_median"),
        ])

        pred = predict_rookies(target, baselines, [2026]).set_index("player_id")

        self.assertEqual(pred.loc["fast", "targets_pg"], pred.loc["slow", "targets_pg"])
        self.assertEqual(pred.loc["fast", "athletic_tier"], "above_median")
        self.assertEqual(pred.loc["slow", "athletic_tier"], "below_median")

    def test_season_aggregate_separates_appearance_and_opportunity_games(self):
        rows = []
        for week, targets in [(1, 1), (2, 0)]:
            row = {c: 0.0 for c in STAT_COLS}
            row.update(
                player_id="p1", season=2025, week=week, position="WR", team="LA",
                targets=targets, receptions=targets, receiving_yards=10 * targets,
                _appeared=True,
            )
            rows.append(row)
        out = season_aggregate(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(out["games_played"], 2)
        self.assertEqual(out["opportunity_games"], 1)

    def test_rookie_availability_uses_full_cohort_and_depth_band(self):
        cohort = pd.DataFrame(
            [
                dict(player_id="a", season=2024, team="A", position="QB", round_bucket="round_4_7",
                     games_played=8, opportunity_games=4, attempts_pg=10.0,
                     attempts_per_elig=10.0, eligible_weeks=17, status='ACT',
                     role_rate_eligible=True, vacated_carry_share=0.2,
                     vacated_target_share=0.2, vacated_attempts_share=0.5,
                     target_depth_rank=1, nfl_depth_rank=8),
                dict(player_id="b", season=2024, team="B", position="QB", round_bucket="round_4_7",
                     games_played=0, opportunity_games=0, attempts_pg=np.nan,
                     attempts_per_elig=0.0, eligible_weeks=17, status='ACT',
                     role_rate_eligible=True, vacated_carry_share=0.2,
                     vacated_target_share=0.2, vacated_attempts_share=0.5,
                     target_depth_rank=np.nan, nfl_depth_rank=2),
            ]
        )
        baselines = fit_rookie_baselines(cohort, [2024])
        b = baselines.loc[("QB", "round_4_7")]
        self.assertEqual(b["mean_games_played"], 4)
        # 10.0 for the rookie who played, 0.0 for the one who did not: the
        # role rate is the mean over both, because "he was available and did
        # nothing" is evidence about the bucket. Averaging the conditional
        # rate would return 10 and over-project the bucket 2x.
        self.assertEqual(b["attempts_per_elig"], 5.0)
        self.assertEqual(b["n_train_rookies"], 2)
        # Both rookies now count toward the RATE, not just the one who
        # recorded an opportunity: n_rate_rookies tracks the role-eligible
        # cohort (rostered, available, zeros included), which is the
        # population the rate is fit on.
        self.assertEqual(b["n_rate_rookies"], 2)

        target = pd.DataFrame([dict(
            player_id="rook", season=2025, team="A", position="QB", round_bucket="round_4_7",
            pick=200, rookie_tier="drafted", vacated_attempts_share=0.5,
            vacated_carry_share=0.2, vacated_target_share=0.2, athletic_tier="no_data",
            target_depth_rank=1, nfl_depth_rank=8,
        )])
        pred = predict_rookies(target, baselines, [2025])
        self.assertEqual(pred.loc[0, "attempts_pg"], 5.0)
        # Rank-1 cell has n=1, so availability falls back to the full
        # draft-bucket mean instead of shipping the tiny cell's 8 games.
        self.assertEqual(pred.loc[0, "projected_games"], 4)
        self.assertEqual(pred.loc[0, "rookie_availability_cell_n"], 1)
        self.assertTrue(pred.loc[0, "rookie_availability_fallback_used"])
        self.assertEqual(pred.loc[0, "rookie_depth_band"], "rank_1")
        self.assertEqual(pred.loc[0, "target_depth_rank"], 1)
        self.assertEqual(pred.loc[0, "nfl_depth_rank"], 8)

    def test_rookie_upward_vacancy_scale_requires_starter_or_committee_role(self):
        idx = pd.MultiIndex.from_tuples(
            [("QB", "round_1")], names=["position", "round_bucket"])
        baselines = pd.DataFrame([dict(
            attempts_per_elig=10.0,
            vacated_carry_share=0.2,
            vacated_target_share=0.2,
            vacated_attempts_share=0.2,
            mean_games_rank_2=5.0,
            mean_games_played=3.0,
            n_train_rookies=20,
        )], index=idx)
        target = pd.DataFrame([
            dict(player_id="starter", season=2026, team="A", position="QB",
                 round_bucket="round_1", pick=1, rookie_tier="drafted",
                 vacated_carry_share=0.2, vacated_target_share=0.2,
                 vacated_attempts_share=0.5, athletic_tier="no_data",
                 target_depth_rank=2, nfl_depth_rank=2),
            dict(player_id="backup", season=2026, team="B", position="QB",
                 round_bucket="round_1", pick=2, rookie_tier="drafted",
                 vacated_carry_share=0.2, vacated_target_share=0.2,
                 vacated_attempts_share=0.5, athletic_tier="no_data",
                 target_depth_rank=2, nfl_depth_rank=2),
        ])
        chart = pd.DataFrame([
            dict(gsis_id="starter", position="QB", role="starter"),
            dict(gsis_id="backup", position="QB", role="backup"),
        ])

        pred = predict_rookies(target, baselines, [2026], depth_chart=chart).set_index("player_id")

        self.assertEqual(pred.loc["starter", "rookie_vacancy_scale"], 2.5)
        self.assertEqual(pred.loc["starter", "attempts_pg"], 25.0)
        self.assertEqual(pred.loc["backup", "rookie_vacancy_scale"], 1.0)
        self.assertEqual(pred.loc["backup", "attempts_pg"], 10.0)

    def _wr_vacancy_case(self, extra_rows=(), residual=1.0):
        idx = pd.MultiIndex.from_tuples(
            [("WR", "round_1")], names=["position", "round_bucket"])
        baselines = pd.DataFrame([dict(
            targets_per_elig=6.0,
            vacated_carry_share=0.2,
            vacated_target_share=0.2,
            vacated_attempts_share=0.2,
            mean_games_rank_2=12.0,
            mean_games_played=10.0,
            n_train_rookies=20,
        )], index=idx)
        base = dict(season=2026, team="A", position="WR", round_bucket="round_1",
                    rookie_tier="drafted", vacated_carry_share=0.2,
                    vacated_target_share=0.5, vacated_attempts_share=0.2,
                    athletic_tier="no_data", target_depth_rank=2, nfl_depth_rank=2,
                    rookie_residual_carry_fraction=1.0,
                    rookie_residual_target_fraction=residual)
        rows = [dict(base, player_id="r1", pick=1)]
        rows += [dict(base, **r) for r in extra_rows]
        chart = pd.DataFrame(
            [dict(gsis_id=r["player_id"], position="WR", role="starter") for r in rows])
        return pd.DataFrame(rows), baselines, chart

    def test_rookie_boost_is_netted_against_what_veterans_already_absorbed(self):
        # A team's opening cannot be spent twice: if the veteran paths were
        # credited with absorbing all of it, no boost is left for the rookie.
        target, baselines, chart = self._wr_vacancy_case(residual=1.0)
        full = predict_rookies(target, baselines, [2026], depth_chart=chart).iloc[0]
        target, baselines, chart = self._wr_vacancy_case(residual=0.0)
        none = predict_rookies(target, baselines, [2026], depth_chart=chart).iloc[0]

        self.assertGreater(full["rookie_vacancy_scale"], 1.0)
        self.assertEqual(none["rookie_vacancy_scale"], 1.0)
        self.assertEqual(none["targets_pg"], 6.0)
        self.assertLess(none["targets_pg"], full["targets_pg"])

    def test_two_rookies_in_one_room_split_the_opening(self):
        # Each of two co-eligible rookies claiming the whole residual is the
        # same double-count, one level down.
        lone, baselines, chart = self._wr_vacancy_case()
        alone = predict_rookies(lone, baselines, [2026], depth_chart=chart).iloc[0]

        pair, baselines, chart = self._wr_vacancy_case(
            extra_rows=[dict(player_id="r2", pick=2)])
        shared = predict_rookies(pair, baselines, [2026], depth_chart=chart)
        shared = shared.set_index("player_id")

        self.assertGreater(alone["rookie_vacancy_scale"], 1.0)
        for pid in ("r1", "r2"):
            self.assertLess(shared.loc[pid, "rookie_vacancy_scale"],
                            alone["rookie_vacancy_scale"])
        # Equal bucket weights, so the opening splits evenly between them.
        self.assertAlmostEqual(shared.loc["r1", "rookie_vacancy_scale"],
                               shared.loc["r2", "rookie_vacancy_scale"])
        excess = alone["rookie_vacancy_scale"] - 1.0
        self.assertAlmostEqual(
            sum(shared.loc[p, "rookie_vacancy_scale"] - 1.0 for p in ("r1", "r2")),
            excess)

    def test_downward_vacancy_scale_is_never_netted(self):
        # A below-average opening is a true fact about this rookie's
        # situation; nobody else's claim makes it less true.
        target, baselines, chart = self._wr_vacancy_case(residual=0.0)
        target["vacated_target_share"] = 0.05
        pred = predict_rookies(target, baselines, [2026], depth_chart=chart).iloc[0]
        self.assertLess(pred["rookie_vacancy_scale"], 1.0)

    def test_rookie_depth_rate_ladder_is_neutral_until_rookie_validated(self):
        rows = pd.DataFrame([dict(
            player_id="backup", position="QB", nfl_depth_rank=2,
            pred_pg=20.0, pred_pg_low=10.0, pred_pg_high=30.0,
            low_confidence=False,
        )])

        out = _apply_rookie_depth_rate_gating(rows).iloc[0]

        self.assertAlmostEqual(out["role_discount_factor"], 1.0)
        self.assertAlmostEqual(out["pred_pg"], 20.0)
        self.assertAlmostEqual(out["pred_pg_low"], 10.0)
        self.assertAlmostEqual(out["pred_pg_high"], 30.0)
        self.assertFalse(out["role_discount_applied"])
        self.assertTrue(out["low_confidence"])

    def test_target_drafted_rookie_placeholder_is_canonicalized_by_pfr_id(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        pd.DataFrame([dict(
            season=2026, gsis_id="placeholder", round=1, pick=5, position="WR", team="LAR",
            pfr_player_name="Rookie Receiver", pfr_player_id="ReceRo00",
        )]).to_sql("draft_picks", conn, index=False)
        pd.DataFrame([
            dict(player_id="00-0000001", team="LA", position="WR", years_exp=0, draft_number=5,
                 player_name="Rookie Receiver", pfr_id="ReceRo00", season=2026),
            dict(player_id="00-0000002", team="LA", position="RB", years_exp=0, draft_number=np.nan,
                 player_name="Camp Back", pfr_id="BackCa00", season=2026),
        ]).to_sql("seasonal_rosters", conn, index=False)
        pd.DataFrame([
            dict(gsis_id="00-0000001", pfr_id="ReceRo00"),
            dict(gsis_id="00-0000002", pfr_id="BackCa00"),
        ]).to_sql("players", conn, index=False)

        rookies = identify_target_season_rookie_class(conn, 2026)
        drafted = rookies[rookies["rookie_tier"] == "drafted"].iloc[0]
        self.assertEqual(drafted["player_id"], "00-0000001")
        self.assertEqual(drafted["team"], "LA")
        self.assertEqual(len(rookies[rookies["player_id"] == "00-0000001"]), 1)

    def test_null_draft_ids_are_conserved_and_name_pick_fallback_resolves(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        pd.DataFrame([
            dict(season=2026, gsis_id=None, round=2, pick=33, position="WR", team="SFO",
                 pfr_player_name="De'Zhaun Stribling", pfr_player_id="StriDe01"),
            dict(season=2026, gsis_id="LAW090280", round=5, pick=168, position="WR", team="DET",
                 pfr_player_name="Kendrick Law", pfr_player_id="LawxKe00"),
        ]).to_sql("draft_picks", conn, index=False)
        pd.DataFrame([
            dict(player_id="00-0041035", team="SF", position="WR", years_exp=0,
                 draft_number=33, player_name="De'Zhaun Stribling", pfr_id="StriDe01", season=2026),
            dict(player_id="00-0041446", team="DET", position="WR", years_exp=0,
                 draft_number=168, player_name="Kendrick Law", pfr_id=None, season=2026),
        ]).to_sql("seasonal_rosters", conn, index=False)
        pd.DataFrame([
            dict(gsis_id="00-0041035", pfr_id="StriDe01"),
        ]).to_sql("players", conn, index=False)

        rookies = identify_target_season_rookie_class(conn, 2026)
        drafted = rookies[rookies["rookie_tier"] == "drafted"]
        self.assertEqual(len(drafted), 2)
        self.assertEqual(set(drafted["player_id"]), {"00-0041035", "00-0041446"})
        self.assertFalse(drafted["rookie_id_unresolved"].any())

    def test_historical_combine_tier_is_scored_as_of_player_season(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        pd.DataFrame([
            dict(season=2016, pos="WR", pfr_id="A", forty=4.40, vertical=40.0),
            dict(season=2016, pos="WR", pfr_id="B", forty=4.80, vertical=30.0),
            dict(season=2017, pos="WR", pfr_id="E", forty=4.60, vertical=35.0),
            dict(season=2026, pos="WR", pfr_id="C", forty=4.20, vertical=45.0),
            dict(season=2026, pos="WR", pfr_id="D", forty=5.00, vertical=20.0),
        ]).to_sql("combine_data", conn, index=False)
        pd.DataFrame([
            dict(gsis_id="pA", pfr_id="A"),
            dict(gsis_id="pB", pfr_id="B"),
        ]).to_sql("players", conn, index=False)

        asof = combine_athletic_scores_by_pfr_id(conn, max_reference_season=2016).set_index("pfr_id")
        historical = load_combine_athletic_tier(conn).set_index("player_id")
        self.assertAlmostEqual(historical.loc["pA", "athletic_score"], asof.loc["A", "athletic_score"])
        self.assertAlmostEqual(historical.loc["pB", "athletic_score"], asof.loc["B", "athletic_score"])

        before = combine_athletic_scores_by_pfr_id(conn).set_index("pfr_id").loc["E", "athletic_score"]
        pd.DataFrame([
            dict(season=2017, pos="WR", pfr_id="F", forty=4.10, vertical=50.0),
        ]).to_sql("combine_data", conn, index=False, if_exists="append")
        after = combine_athletic_scores_by_pfr_id(conn).set_index("pfr_id").loc["E", "athletic_score"]
        self.assertAlmostEqual(before, after)
