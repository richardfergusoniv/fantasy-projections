"""Guards against scoring the team-grain model on a player-grain frame.

The bug this exists for: TEAM_MODEL_FEATURES used to end in `naive_pred`,
a name the PLAYER pair-builder also uses for its own carry-forward
baseline. Predicting the team model on a player frame therefore raised
nothing - it silently fed the model a player's prior receiving rate
(~30 yd/g) where it expected a team's prior passing volume (~230 yd/g).
Team totals came out ~40% low, and because the reframed receiving path is
`share x team_total`, every composed receiving prediction inherited that,
which biased the shipped interval residuals upward by ~10 yd/g.

Two tests: one on the mechanism (the name collision is gone), one on the
symptom (residual quantiles that straddle zero). The symptom test is the
cheap one that would have caught this in either direction.
"""
import unittest

import numpy as np
import pandas as pd

from src.projection.transitions import (
    TEAM_FEATURES, TEAM_MODEL_FEATURES, build_team_transition_pairs, team_model_inputs,
)


def _feat():
    """Two seasons x two teams, one player row each, with distinguishable
    team-grain and player-grain values."""
    rows = []
    for season, tt in ((2021, 200.0), (2022, 260.0)):
        for team, bump in (("AAA", 0.0), ("BBB", 30.0)):
            row = {c: 0.5 for c in TEAM_FEATURES}
            row.update(season=season, team=team, player_id=f"p_{team}",
                       team_passing_yards_pg=tt + bump)
            rows.append(row)
    return pd.DataFrame(rows)


class TeamModelGrainTest(unittest.TestCase):
    def test_team_model_features_do_not_collide_with_player_naive_pred(self):
        # The player pair-builder's carry-forward column must not be able to
        # masquerade as the team model's lag feature.
        self.assertNotIn("naive_pred", TEAM_MODEL_FEATURES)
        self.assertIn("team_naive_pred", TEAM_MODEL_FEATURES)

    def test_player_grain_frame_cannot_be_scored_as_a_team_frame(self):
        # A frame carrying only the player-grain name must raise, not
        # silently supply the wrong quantity.
        player_like = pd.DataFrame([{**{c: 0.5 for c in TEAM_FEATURES}, "naive_pred": 31.4}])
        with self.assertRaises(KeyError):
            player_like[TEAM_MODEL_FEATURES]

    def test_team_model_inputs_returns_team_grain_values(self):
        feat = _feat()
        pairs = [(2021, 2022)]
        # Two player rows on the same team must both receive that TEAM's
        # prior passing volume, not anything player-specific.
        got = team_model_inputs(feat, pairs, [2021, 2021, 2021],
                                ["AAA", "BBB", "AAA"])
        self.assertEqual(list(got.columns), list(TEAM_MODEL_FEATURES))
        self.assertEqual(got["team_naive_pred"].tolist(), [200.0, 230.0, 200.0])

    def test_team_model_inputs_marks_unknown_teams_rather_than_guessing(self):
        got = team_model_inputs(_feat(), [(2021, 2022)], [2021], ["ZZZ"])
        self.assertTrue(np.isnan(got["team_naive_pred"].iloc[0]))

    def test_build_team_transition_pairs_carries_both_names(self):
        pairs = build_team_transition_pairs(_feat(), [(2021, 2022)])
        # `naive_pred` stays for baseline scoring; `team_naive_pred` is the
        # model feature. They are the same number at team grain.
        self.assertTrue((pairs["naive_pred"] == pairs["team_naive_pred"]).all())


class IntervalResidualSymmetryTest(unittest.TestCase):
    """The symptom guard.

    A well-calibrated 10th/90th residual pair straddles zero. The reframed
    receiving stats shipped at 4.5x-6.1x asymmetry (WR -4.13/+25.34) for
    two review rounds while their non-reframed siblings sat near 1.0x -
    the tell was in the artifact the whole time and nothing read it.
    """

    MAX_ASYMMETRY = 3.0

    def test_reframed_residuals_are_not_wildly_one_sided(self):
        import os
        from src.projection.predict import INTERVAL_RESIDUALS_PATH

        if not os.path.exists(INTERVAL_RESIDUALS_PATH):
            self.skipTest("interval_residuals.csv not built (run backtest)")
        resid = pd.read_csv(INTERVAL_RESIDUALS_PATH)
        bad = []
        for r in resid.itertuples():
            lo, hi = float(r.resid_low), float(r.resid_high)
            if lo >= 0 or hi <= 0:
                bad.append(f"{r.position} {r.stat}: interval does not straddle 0 ({lo:.2f}, {hi:.2f})")
                continue
            ratio = max(abs(hi / lo), abs(lo / hi))
            if ratio > self.MAX_ASYMMETRY:
                bad.append(f"{r.position} {r.stat}: {ratio:.1f}x one-sided ({lo:.2f}, {hi:.2f})")
        self.assertFalse(bad, "systematically biased predictions: " + "; ".join(bad))


if __name__ == "__main__":
    unittest.main()
