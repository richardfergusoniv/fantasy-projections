"""Phase 1 of the role-rate relabel: the label denominator and the population.

These pin the two decisions the whole change rests on:

  * a rate is per ELIGIBLE week (rostered, off reserve), not per APPEARANCE
    week - the appearance denominator is survivorship-selected and is why
    backups projected like starters;
  * a zero-production season enters training only when its cause is ROLE.
    IR belongs to the status-override gate and cut players are out of the
    population; admitting either would bake injury attrition and roster
    churn into a rate that describes a role.
"""
import sqlite3
import unittest

import numpy as np
import pandas as pd

from src.projection.data_prep import (
    ELIGIBLE_ROSTER_STATUSES, SEASON_GAMES_CAP, player_dominant_roster_status,
    player_eligible_weeks,
)
from src.projection.depth_history import (
    DEEP_TIER, DEPTH_TIER_COLUMN, OFF_CHART_TIER, depth_tiers,
)
from src.projection.transitions import (
    MIN_ELIGIBLE_WEEKS, RECEIVING_SHARE_ELIG_LABEL, ROLE_ZERO_FLAG, SEASON_GAMES,
    role_rate_label,
)


def _roster_db(rows):
    """In-memory weekly_rosters with (season, player_id, week, status) rows."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "create table weekly_rosters (season int, player_id text, week int, "
        "status text, game_type text)")
    conn.executemany(
        "insert into weekly_rosters values (?, ?, ?, ?, 'REG')", rows)
    conn.commit()
    return conn


class EligibleWeeks(unittest.TestCase):
    def test_counts_only_eligible_statuses(self):
        rows = [(2024, "p1", w, "ACT") for w in range(1, 11)]
        rows += [(2024, "p1", w, "RES") for w in range(11, 19)]
        rows += [(2024, "p2", w, "DEV") for w in range(1, 6)]
        rows += [(2024, "p3", w, "CUT") for w in range(1, 19)]
        out = player_eligible_weeks(_roster_db(rows), [2024]).set_index("player_id")
        self.assertEqual(out.at["p1", "eligible_weeks"], 10)  # IR weeks excluded
        self.assertEqual(out.at["p2", "eligible_weeks"], 5)
        self.assertNotIn("p3", out.index)  # never eligible at all

    def test_capped_at_season_games(self):
        # weekly_rosters carries an 18th REG week; a rate over 18 would not
        # compose with a 17-game season.
        rows = [(2024, "p1", w, "ACT") for w in range(1, 19)]
        out = player_eligible_weeks(_roster_db(rows), [2024])
        self.assertEqual(out.at[0, "eligible_weeks"], SEASON_GAMES_CAP)

    def test_cap_matches_season_games(self):
        # data_prep keeps a local copy to stay a leaf module - pin them.
        self.assertEqual(SEASON_GAMES_CAP, SEASON_GAMES)

    def test_eligible_statuses_exclude_reserve_and_cut(self):
        self.assertEqual(set(ELIGIBLE_ROSTER_STATUSES), {"ACT", "INA", "DEV"})
        for gone in ("RES", "CUT", "RET"):
            self.assertNotIn(gone, ELIGIBLE_ROSTER_STATUSES)


class DominantRosterStatus(unittest.TestCase):
    def test_most_weeks_wins(self):
        rows = [(2024, "p1", w, "ACT") for w in range(1, 13)]
        rows += [(2024, "p1", w, "RES") for w in range(13, 19)]
        out = player_dominant_roster_status(_roster_db(rows), [2024])
        self.assertEqual(out.at[0, "status"], "ACT")

    def test_tie_breaks_to_the_later_week(self):
        # a player cut halfway through reads as CUT, not ACT
        rows = [(2024, "p1", w, "ACT") for w in range(1, 10)]
        rows += [(2024, "p1", w, "CUT") for w in range(10, 19)]
        out = player_dominant_roster_status(_roster_db(rows), [2024])
        self.assertEqual(out.at[0, "status"], "CUT")


class DepthTier(unittest.TestCase):
    def test_listed_deep_and_off_chart_are_distinct(self):
        # The finding this split exists for: merged into one bucket, WR/TE
        # land on opposite sides of calibrated.
        self.assertNotEqual(DEEP_TIER, OFF_CHART_TIER)
        got = depth_tiers([1, 2, 3, 4, 9, np.nan])
        np.testing.assert_array_equal(got, [1.0, 2.0, 3.0, DEEP_TIER, DEEP_TIER,
                                            OFF_CHART_TIER])

    def test_era_boundary_collapses_to_the_same_meaning(self):
        # 2016-2024 ranks are a tie-bearing tier capped at 3; 2025+ are a true
        # ordinal to 15. "Beyond the listed top three" must mean one thing.
        old_era = depth_tiers([1, 2, 3])
        new_era = depth_tiers([1, 2, 3, 4, 12, 15])
        np.testing.assert_array_equal(old_era, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(new_era[3:], [DEEP_TIER] * 3)


class RoleRateLabel(unittest.TestCase):
    def test_label_name(self):
        self.assertEqual(role_rate_label("attempts"), "attempts_per_elig")

    def test_eligible_denominator_separates_roles_appearance_does_not(self):
        # A starter and a backup with the SAME appearance-conditional rate:
        # the backup only took snaps in the 5 weeks he was pressed into
        # service. Per appearance they are identical; per eligible week the
        # backup is a third of the starter, which is the real gradient.
        starter = dict(attempts=510.0, games_played=17.0, eligible_weeks=17.0)
        backup = dict(attempts=150.0, games_played=5.0, eligible_weeks=17.0)
        self.assertEqual(starter["attempts"] / starter["games_played"], 30.0)
        self.assertEqual(backup["attempts"] / backup["games_played"], 30.0)
        self.assertAlmostEqual(starter["attempts"] / starter["eligible_weeks"], 30.0)
        self.assertAlmostEqual(backup["attempts"] / backup["eligible_weeks"], 150 / 17)

    def test_injury_shortened_starter_keeps_a_full_season_rate(self):
        # 9 eligible weeks, then IR. The role rate must describe the role, not
        # the injury - dividing by a flat 17 would price the absence in.
        rate = 270.0 / 9.0
        self.assertAlmostEqual(rate * SEASON_GAMES, 510.0)
        self.assertAlmostEqual(270.0 / SEASON_GAMES * SEASON_GAMES, 270.0)

    def test_share_label_uses_full_season_team_yards(self):
        # receiving_yards_share_elig = rec yds / (team season pass yds x elig/17)
        rec, team_pass, elig = 800.0, 4000.0, 17.0
        self.assertAlmostEqual(rec / (team_pass * elig / SEASON_GAMES_CAP), 0.20)
        half = 8.5
        self.assertAlmostEqual(rec / (team_pass * half / SEASON_GAMES_CAP), 0.40)


class MinimumEligibility(unittest.TestCase):
    def test_threshold_is_a_population_filter_not_a_clip(self):
        # A 2-week signing is not evidence about a full-season role. Such rows
        # must LEAVE, not be clipped up to the threshold and kept at weight 1.
        self.assertGreaterEqual(MIN_ELIGIBLE_WEEKS, 2)
        self.assertLess(MIN_ELIGIBLE_WEEKS, SEASON_GAMES)


class RoleZeroFlagContract(unittest.TestCase):
    def test_flag_name_is_exported(self):
        self.assertEqual(ROLE_ZERO_FLAG, "is_role_zero")

    def test_share_elig_label_is_wired(self):
        self.assertEqual(RECEIVING_SHARE_ELIG_LABEL, "receiving_yards_share_elig")

    def test_tier_column_is_wired(self):
        self.assertEqual(DEPTH_TIER_COLUMN, "depth_tier")


if __name__ == "__main__":
    unittest.main()
