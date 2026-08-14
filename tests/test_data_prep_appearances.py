import sqlite3
import unittest
from unittest.mock import patch

import pandas as pd

from src.projection.data_prep import (
    STAT_COLS,
    load_weekly_usage,
    player_active_rz_position_opportunity,
    player_active_team_opportunity,
    player_season_receiving_yards_share,
    season_aggregate,
    team_week_rz_position_totals,
)


def _weekly_row(player_id, week, position="WR", targets=0, attempts=0, carries=0):
    row = {c: 0.0 for c in STAT_COLS}
    row.update(
        player_id=player_id,
        season=2024,
        week=week,
        season_type="REG",
        recent_team="LA",
        position=position,
        targets=targets,
        attempts=attempts,
        carries=carries,
    )
    return row


def _put_base_tables(conn, weekly_rows, include_pfr_id=True):
    pd.DataFrame(weekly_rows).to_sql("weekly", conn, index=False)
    players = []
    for player_id, position in pd.DataFrame(weekly_rows)[["player_id", "position"]].drop_duplicates().itertuples(index=False):
        row = {"gsis_id": player_id, "position": position}
        if include_pfr_id:
            row["pfr_id"] = f"pfr-{player_id}"
        players.append(row)
    pd.DataFrame(players).to_sql("players", conn, index=False)


def _put_snap_table(conn, rows):
    pd.DataFrame(rows, columns=[
        "season", "week", "pfr_player_id", "team", "offense_pct", "game_type",
    ]).to_sql("snap_counts", conn, index=False)


class DataPrepAppearanceTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.addCleanup(self.conn.close)

    def test_missing_optional_snap_table_falls_back_to_opportunity(self):
        _put_base_tables(self.conn, [
            _weekly_row("used", 1, targets=1),
            _weekly_row("unused", 1, targets=0),
        ])

        usage = load_weekly_usage(self.conn).set_index("player_id")

        self.assertTrue(bool(usage.loc["used", "_appeared"]))
        self.assertFalse(bool(usage.loc["unused", "_appeared"]))

    def test_unexpected_crosswalk_schema_error_propagates(self):
        # The optional snap schema is valid, but the required player crosswalk
        # is malformed. This must not be converted into opportunity fallback.
        _put_base_tables(self.conn, [_weekly_row("p1", 1, targets=1)], include_pfr_id=False)
        _put_snap_table(self.conn, [(2024, 1, "pfr-p1", "LA", 0.5, "REG")])

        with self.assertRaises(pd.errors.DatabaseError):
            load_weekly_usage(self.conn)

    def test_snap_only_week_adds_zero_stat_appearance(self):
        _put_base_tables(self.conn, [_weekly_row("p1", 1, targets=1)])
        _put_snap_table(self.conn, [
            (2024, 1, "pfr-p1", "LA", 0.5, "REG"),
            (2024, 2, "pfr-p1", "LA", 0.4, "REG"),
        ])

        usage = load_weekly_usage(self.conn).sort_values("week")
        added = usage[usage["week"] == 2].iloc[0]
        self.assertTrue(bool(added["_appeared"]))
        self.assertEqual(float(added[STAT_COLS].sum()), 0.0)

        season = season_aggregate(usage).iloc[0]
        self.assertEqual(season["games_played"], 2)
        self.assertEqual(season["opportunity_games"], 1)

    def test_alias_rows_collapse_to_one_player_week_and_sum_stats(self):
        _put_base_tables(self.conn, [
            _weekly_row("p1", 1, targets=3),
            _weekly_row("p1", 1, targets=2),
        ])
        _put_snap_table(self.conn, [(2024, 1, "pfr-p1", "LA", 0.5, "REG")])

        usage = load_weekly_usage(self.conn)
        self.assertEqual(len(usage), 1)
        self.assertEqual(usage.iloc[0]["targets"], 5)
        season = season_aggregate(usage).iloc[0]
        self.assertEqual(season["games_played"], 1)
        self.assertEqual(season["opportunity_games"], 1)

    def test_opportunity_games_never_exceed_appearance_games(self):
        _put_base_tables(self.conn, [
            _weekly_row("wr", 1, targets=1),
            _weekly_row("qb", 1, position="QB", attempts=2),
        ])
        _put_snap_table(self.conn, [
            (2024, 1, "pfr-wr", "LA", 0.5, "REG"),
            (2024, 2, "pfr-wr", "LA", 0.2, "REG"),
            (2024, 1, "pfr-qb", "LA", 0.8, "REG"),
        ])

        season = season_aggregate(load_weekly_usage(self.conn))
        self.assertTrue((season["opportunity_games"] <= season["games_played"]).all())

    def test_receiving_share_denominator_includes_zero_target_appearances(self):
        usage = pd.DataFrame([
            dict(player_id="wr", season=2024, week=1, team="LA", position="WR",
                 targets=1, carries=0, attempts=0, receiving_yards=50, _appeared=True),
            dict(player_id="wr", season=2024, week=2, team="LA", position="WR",
                 targets=0, carries=0, attempts=0, receiving_yards=0, _appeared=True),
        ])
        team_weeks = pd.DataFrame([
            dict(season=2024, week=1, team="LA", team_passing_yards=100),
            dict(season=2024, week=2, team="LA", team_passing_yards=100),
        ])
        with patch("src.projection.data_prep.load_weekly_usage", return_value=usage), patch(
            "src.projection.data_prep.team_week_yardage_totals", return_value=team_weeks
        ):
            share = player_season_receiving_yards_share(self.conn, [2024]).iloc[0]

        self.assertAlmostEqual(share["receiving_yards_share"], 0.25)

    def test_receiving_share_sums_same_week_alias_rows(self):
        usage = pd.DataFrame([
            dict(player_id="wr", season=2024, week=1, team="LA", position="WR",
                 receiving_yards=30, _appeared=True),
            dict(player_id="wr", season=2024, week=1, team="LA", position="WR",
                 receiving_yards=20, _appeared=True),
        ])
        team_weeks = pd.DataFrame([
            dict(season=2024, week=1, team="LA", team_passing_yards=100),
        ])
        with patch("src.projection.data_prep.load_weekly_usage", return_value=usage), patch(
            "src.projection.data_prep.team_week_yardage_totals", return_value=team_weeks
        ):
            share = player_season_receiving_yards_share(self.conn, [2024]).iloc[0]
        self.assertAlmostEqual(share["receiving_yards_share"], 0.5)

    def test_rz_position_totals_use_season_position_not_master_position(self):
        pd.DataFrame([
            dict(season=2024, week=1, player_id="hybrid", position="QB"),
        ]).to_sql("weekly", self.conn, index=False)
        pd.DataFrame([
            dict(player_id="hybrid", season=2024, position="TE"),
        ]).to_sql("seasonal_rosters", self.conn, index=False)
        pd.DataFrame([
            dict(gsis_id="hybrid", position="TE"),
        ]).to_sql("players", self.conn, index=False)
        pd.DataFrame([
            dict(season=2024, week=1, posteam="NO", rush_attempt=1,
                 pass_attempt=0, rusher_player_id="hybrid", receiver_player_id=None,
                 yardline_100=5, season_type="REG"),
        ]).to_sql("pbp", self.conn, index=False)

        totals = team_week_rz_position_totals(self.conn, [2024])
        row = totals.iloc[0]
        self.assertEqual(row["position"], "QB")
        self.assertEqual(row["team_rz_carries_pos"], 1)

    def test_all_team_share_denominators_include_zero_opportunity_appearances(self):
        usage = pd.DataFrame([
            dict(player_id="wr", season=2024, week=1, team="LA", position="WR",
                 targets=1, carries=0, attempts=0, _appeared=True),
            dict(player_id="wr", season=2024, week=2, team="LA", position="WR",
                 targets=0, carries=0, attempts=0, _appeared=True),
        ])
        team_weeks = pd.DataFrame([
            dict(season=2024, week=1, team="LA", team_pass_attempts=10,
                 team_rush_attempts=5, team_rz_pass_attempts=2, team_rz_rush_attempts=1),
            dict(season=2024, week=2, team="LA", team_pass_attempts=20,
                 team_rush_attempts=8, team_rz_pass_attempts=3, team_rz_rush_attempts=2),
        ])
        air_weeks = pd.DataFrame([
            dict(season=2024, week=1, team="LA", team_air_yards=100),
            dict(season=2024, week=2, team="LA", team_air_yards=200),
        ])
        with patch("src.projection.data_prep.load_weekly_usage", return_value=usage), patch(
            "src.projection.data_prep.team_week_pbp_totals", return_value=team_weeks
        ), patch("src.projection.data_prep.team_week_air_yards", return_value=air_weeks):
            opp = player_active_team_opportunity(self.conn, [2024]).iloc[0]

        self.assertEqual(opp["team_pass_attempts_active"], 30)
        self.assertEqual(opp["team_rush_attempts_active"], 13)
        self.assertEqual(opp["team_rz_pass_attempts_active"], 5)
        self.assertEqual(opp["team_rz_rush_attempts_active"], 3)
        self.assertEqual(opp["team_air_yards_active"], 300)

    def test_rz_monopoly_denominator_includes_zero_opportunity_appearances(self):
        usage = pd.DataFrame([
            dict(player_id="wr", season=2024, week=1, team="LA", position="WR", _appeared=True),
            dict(player_id="wr", season=2024, week=2, team="LA", position="WR", _appeared=True),
        ])
        rz_weeks = pd.DataFrame([
            dict(season=2024, week=1, team="LA", position="WR",
                 team_rz_carries_pos=1, team_rz_targets_pos=2),
            dict(season=2024, week=2, team="LA", position="WR",
                 team_rz_carries_pos=3, team_rz_targets_pos=4),
        ])
        with patch("src.projection.data_prep.load_weekly_usage", return_value=usage), patch(
            "src.projection.data_prep.team_week_rz_position_totals", return_value=rz_weeks
        ):
            opp = player_active_rz_position_opportunity(self.conn, [2024]).iloc[0]

        self.assertEqual(opp["team_rz_carries_pos_active"], 4)
        self.assertEqual(opp["team_rz_targets_pos_active"], 6)


if __name__ == "__main__":
    unittest.main()
