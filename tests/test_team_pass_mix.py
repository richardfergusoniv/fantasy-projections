"""Tests for hierarchical team WR/TE/RB pass mix (L2) and composition (L3)."""
import unittest

import numpy as np
import pandas as pd

from src.projection.team_pass_mix import (
    MIX_COLS,
    apply_hierarchical_pass_distribution,
    attach_team_pass_mix,
    _allocate_wr_by_formation_role,
    _shares_to_logits,
    _softmax_shares,
)


class SoftmaxRoundTripTests(unittest.TestCase):
    def test_shares_round_trip(self):
        wr, te, rb = 0.55, 0.25, 0.20
        wr_l, te_l = _shares_to_logits(wr, te, rb)
        wr2, te2, rb2 = _softmax_shares(wr_l, te_l)
        self.assertAlmostEqual(wr2, wr, places=5)
        self.assertAlmostEqual(te2, te, places=5)
        self.assertAlmostEqual(rb2, rb, places=5)


class HierarchicalCompositionTests(unittest.TestCase):
    def test_within_group_sums_to_l2_budget(self):
        # Team attempts 34/game * 17 = 578 season attempts.
        # WR mix 0.60 -> 346.8 WR targets to split 2:1 between A and B by model weights.
        rows = []
        for player_id, position, pred, games in [
            ("a", "WR", 10.0, 17.0),
            ("b", "WR", 5.0, 17.0),
            ("c", "TE", 4.0, 17.0),
            ("d", "RB", 2.0, 17.0),
        ]:
            rows.append({
                "player_id": player_id, "team": "JAX", "position": position,
                "stat": "targets", "pred_pg": pred, "pred_pg_low": pred * 0.8,
                "pred_pg_high": pred * 1.2, "projected_volume_games": games,
                "team_pass_attempts_pg_pred": 34.0,
                "wr_target_share": 0.60, "te_target_share": 0.25, "rb_target_share": 0.15,
            })
        df = pd.DataFrame(rows)
        out = apply_hierarchical_pass_distribution(df, season_games=17.0)
        wr = out[out.position.eq("WR")]
        wr_season = (wr["pred_pg"] * wr["projected_volume_games"]).sum()
        self.assertAlmostEqual(wr_season, 34.0 * 17.0 * 0.60, places=4)
        # A:B weights were 10:5 -> 2:1 of WR budget
        a = out[out.player_id.eq("a")].iloc[0]
        b = out[out.player_id.eq("b")].iloc[0]
        self.assertAlmostEqual(
            a.pred_pg * a.projected_volume_games,
            2 * b.pred_pg * b.projected_volume_games,
            places=4,
        )
        te_season = (
            out.loc[out.position.eq("TE"), "pred_pg"]
            * out.loc[out.position.eq("TE"), "projected_volume_games"]
        ).sum()
        self.assertAlmostEqual(te_season, 34.0 * 17.0 * 0.25, places=4)

    def test_formation_role_keeps_lwr_budget_when_model_favors_swr(self):
        """Pierce-class: chart LWR inherits LWR column even if SWR models hotter."""
        rows = []
        for player_id, role, pred in [
            ("pierce", "LWR", 4.0),   # weak model, chart LWR
            ("downs", "SWR", 10.0),   # hot model, chart SWR
            ("te", None, 3.0),
            ("rb", None, 2.0),
        ]:
            rows.append({
                "player_id": player_id, "team": "IND", "position": "WR" if role else ("TE" if player_id == "te" else "RB"),
                "stat": "targets", "pred_pg": pred, "pred_pg_low": pred * 0.8,
                "pred_pg_high": pred * 1.2, "projected_volume_games": 17.0,
                "team_pass_attempts_pg_pred": 34.0,
                "wr_target_share": 0.60, "te_target_share": 0.25, "rb_target_share": 0.15,
                "formation_role": role,
            })
        # Fix TE/RB rows
        for r in rows:
            if r["player_id"] == "te":
                r["position"] = "TE"
            elif r["player_id"] == "rb":
                r["position"] = "RB"
        df = pd.DataFrame(rows)
        # Pure role priors (blend=1): LWR:SWR = 0.1554:0.0386 among present
        out = apply_hierarchical_pass_distribution(
            df, season_games=17.0, formation_role_blend_w=1.0)
        wr = out[out.position.eq("WR")]
        pierce = wr[wr.player_id.eq("pierce")].iloc[0]
        downs = wr[wr.player_id.eq("downs")].iloc[0]
        pierce_season = pierce.pred_pg * pierce.projected_volume_games
        downs_season = downs.pred_pg * downs.projected_volume_games
        # LWR prior is 0.1554 / (0.1554+0.0386) of WR budget
        wr_budget = 34.0 * 17.0 * 0.60
        expected_lwr = wr_budget * (0.1554 / (0.1554 + 0.0386))
        expected_swr = wr_budget * (0.0386 / (0.1554 + 0.0386))
        self.assertAlmostEqual(pierce_season, expected_lwr, places=4)
        self.assertAlmostEqual(downs_season, expected_swr, places=4)
        self.assertGreater(pierce_season, downs_season)

    def test_formation_role_blend_zero_matches_fungible(self):
        rows = []
        for player_id, role, pred in [("a", "LWR", 10.0), ("b", "SWR", 5.0)]:
            rows.append({
                "player_id": player_id, "team": "JAX", "position": "WR",
                "stat": "targets", "pred_pg": pred, "pred_pg_low": pred * 0.8,
                "pred_pg_high": pred * 1.2, "projected_volume_games": 17.0,
                "team_pass_attempts_pg_pred": 34.0,
                "wr_target_share": 0.60, "te_target_share": 0.25, "rb_target_share": 0.15,
                "formation_role": role,
            })
        df = pd.DataFrame(rows)
        fungible = apply_hierarchical_pass_distribution(
            df, season_games=17.0, formation_role_blend_w=0.0)
        a = fungible[fungible.player_id.eq("a")].iloc[0]
        b = fungible[fungible.player_id.eq("b")].iloc[0]
        self.assertAlmostEqual(
            a.pred_pg * a.projected_volume_games,
            2 * b.pred_pg * b.projected_volume_games,
            places=4,
        )

    def test_allocate_helper_renormalizes_present_roles(self):
        alloc, wshare = _allocate_wr_by_formation_role(
            raw=[1.0, 1.0], roles=["LWR", "RWR"], budget=100.0, blend_w=1.0)
        self.assertAlmostEqual(alloc.sum(), 100.0, places=6)
        self.assertAlmostEqual(alloc[0] / alloc[1], 0.1554 / 0.0667, places=4)


class AttachMixTests(unittest.TestCase):
    def test_attach_mix_fills_missing_team(self):
        profiles = pd.DataFrame([
            {"season": 2026, "team": "JAX", "wr_target_share": 0.5,
             "te_target_share": 0.3, "rb_target_share": 0.2, "mix_source": "scheme_model"},
        ])
        df = pd.DataFrame([
            {"team": "JAX", "player_id": "a"},
            {"team": "XXX", "player_id": "b"},
        ])
        out = attach_team_pass_mix(df, profiles, 2026)
        self.assertTrue(out[MIX_COLS].notna().all(axis=None))
        self.assertAlmostEqual(out.loc[out.team.eq("JAX"), "wr_target_share"].iloc[0], 0.5)


if __name__ == "__main__":
    unittest.main()
