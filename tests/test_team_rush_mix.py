"""Tests for hierarchical team RB/QB/OTHER rush mix (L2) and composition (L3)."""
import unittest

import numpy as np
import pandas as pd

from src.projection.team_rush_mix import (
    MIX_COLS,
    apply_hierarchical_rush_distribution,
    attach_team_rush_mix,
    _shares_to_logits,
    _softmax_shares,
)


class SoftmaxRoundTripTests(unittest.TestCase):
    def test_shares_round_trip(self):
        rb, qb, other = 0.78, 0.14, 0.08
        rb_l, qb_l = _shares_to_logits(rb, qb, other)
        rb2, qb2, ot2 = _softmax_shares(rb_l, qb_l)
        self.assertAlmostEqual(rb2, rb, places=5)
        self.assertAlmostEqual(qb2, qb, places=5)
        self.assertAlmostEqual(ot2, other, places=5)


class HierarchicalRushCompositionTests(unittest.TestCase):
    def test_rb_group_sums_to_l2_budget(self):
        # Team 27 carries/game * 17 = 459 season carries.
        # RB mix 0.80 -> 367.2 RB carries split 2:1 between A and B.
        rows = []
        for player_id, position, pred, games in [
            ("a", "RB", 14.0, 17.0),
            ("b", "RB", 7.0, 17.0),
            ("c", "QB", 4.0, 17.0),
            ("d", "WR", 1.0, 17.0),
        ]:
            rows.append({
                "player_id": player_id, "team": "GB", "position": position,
                "stat": "carries", "pred_pg": pred, "pred_pg_low": pred * 0.8,
                "pred_pg_high": pred * 1.2, "projected_volume_games": games,
                "team_carries_pg_pred": 27.0,
                "rb_carry_share": 0.80, "qb_carry_share": 0.15, "other_carry_share": 0.05,
            })
            if position in ("RB", "QB", "WR"):
                rows.append({
                    "player_id": player_id, "team": "GB", "position": position,
                    "stat": "rushing_yards", "pred_pg": pred * 4.5,
                    "pred_pg_low": pred * 4.5 * 0.8, "pred_pg_high": pred * 4.5 * 1.2,
                    "projected_volume_games": games,
                    "team_carries_pg_pred": 27.0,
                    "rb_carry_share": 0.80, "qb_carry_share": 0.15, "other_carry_share": 0.05,
                })
        df = pd.DataFrame(rows)
        out = apply_hierarchical_rush_distribution(df, season_games=17.0)
        rb = out[out.position.eq("RB") & out.stat.eq("carries")]
        rb_season = (rb["pred_pg"] * rb["projected_volume_games"]).sum()
        self.assertAlmostEqual(rb_season, 27.0 * 17.0 * 0.80, places=4)
        a = rb[rb.player_id.eq("a")].iloc[0]
        b = rb[rb.player_id.eq("b")].iloc[0]
        self.assertAlmostEqual(
            a.pred_pg * a.projected_volume_games,
            2 * b.pred_pg * b.projected_volume_games,
            places=4,
        )
        # Yards family scales with the same player factor.
        a_yards = out[(out.player_id.eq("a")) & (out.stat.eq("rushing_yards"))].iloc[0]
        self.assertAlmostEqual(
            a_yards.hierarchical_rush_scale, a.hierarchical_rush_scale, places=6,
        )

    def test_attach_mix_fills_missing_team(self):
        profiles = pd.DataFrame([
            {"season": 2026, "team": "GB", "rb_carry_share": 0.8,
             "qb_carry_share": 0.15, "other_carry_share": 0.05, "rush_mix_source": "scheme_model"},
        ])
        df = pd.DataFrame([
            {"team": "GB", "player_id": "a"},
            {"team": "XXX", "player_id": "b"},
        ])
        out = attach_team_rush_mix(df, profiles, 2026)
        self.assertTrue(out[MIX_COLS].notna().all(axis=None))
        self.assertAlmostEqual(out.loc[out.team.eq("GB"), "rb_carry_share"].iloc[0], 0.8)


if __name__ == "__main__":
    unittest.main()
