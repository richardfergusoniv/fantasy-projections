"""Upstream QB rush features + joint allocation tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.projection.composition import CompositionContext, compose_board
from src.projection.qb_joint_allocation import reconcile_qb_joint_room
from src.projection.qb_repair.allocation import AllocationParams
from src.projection.qb_repair.apply_board import non_qb_invariance_check
from src.projection.qb_rush_features import (
    apply_qb_rush_multi_season_pooling,
    compute_qb_rush_splits_from_pbp,
    patch_inference_row_with_rush_pool,
)


def test_designed_scramble_split_from_pbp():
    pbp = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 1, 1, 1],
            "rusher_player_id": ["qb1", "qb1", "qb1", "rb1"],
            "passer_player_id": ["qb1", "qb1", "qb1", None],
            "qb_scramble": [0, 1, 0, 0],
            "rush_attempt": [1, 1, 1, 1],
            "pass_attempt": [0, 0, 0, 0],
            "rushing_yards": [5.0, 12.0, 3.0, 8.0],
            "yardline_100": [15, 40, 3, 20],
            "sack": [0, 0, 0, 0],
        }
    )
    splits = compute_qb_rush_splits_from_pbp(pbp)
    qb = splits[splits.player_id == "qb1"].iloc[0]
    assert qb.designed_carries == 2
    assert qb.scramble_carries == 1
    assert qb.rz_designed_carries == 2  # yardline 15 and 3
    assert qb.gl_designed_carries == 1


def test_multi_season_pooling_prevents_one_short_season_erasure():
    feat = pd.DataFrame(
        [
            {"player_id": "m1", "season": 2022, "games_played": 16, "qb_designed_run_rate": 0.08, "qb_scramble_per_dropback": 0.10, "carries_pg": 9.0, "rushing_yards_pg": 55.0},
            {"player_id": "m1", "season": 2023, "games_played": 16, "qb_designed_run_rate": 0.07, "qb_scramble_per_dropback": 0.12, "carries_pg": 9.2, "rushing_yards_pg": 51.0},
            {"player_id": "m1", "season": 2024, "games_played": 17, "qb_designed_run_rate": 0.075, "qb_scramble_per_dropback": 0.09, "carries_pg": 8.2, "rushing_yards_pg": 54.0},
            {"player_id": "m1", "season": 2025, "games_played": 8, "qb_designed_run_rate": 0.03, "qb_scramble_per_dropback": 0.08, "carries_pg": 5.0, "rushing_yards_pg": 27.0},
        ]
    )
    out = apply_qb_rush_multi_season_pooling(feat)
    row = out[out.season == 2025].iloc[0]
    assert row.qb_rush_archetype_carries_pg > 6.5
    assert row.qb_designed_run_rate_pooled > row.qb_designed_run_rate


def test_patch_inference_uses_only_prior_seasons():
    history = pd.DataFrame(
        [
            {"player_id": "m1", "season": 2023, "games_played": 16, "carries_pg": 9.0, "rushing_yards_pg": 50.0, "qb_designed_run_rate": 0.07},
            {"player_id": "m1", "season": 2024, "games_played": 17, "carries_pg": 8.5, "rushing_yards_pg": 52.0, "qb_designed_run_rate": 0.08},
            {"player_id": "m1", "season": 2025, "games_played": 8, "carries_pg": 5.0, "rushing_yards_pg": 25.0, "qb_designed_run_rate": 0.03},
        ]
    )
    row = pd.Series(
        {
            "player_id": "m1",
            "games_played": 8.0,
            "prior_carries_pg": 5.0,
            "prior_rushing_yards_pg": 25.0,
            "qb_designed_run_rate": 0.03,
            "prior_role_rate": 5.0,
        }
    )
    patched, audit = patch_inference_row_with_rush_pool(row, history, target_season=2026)
    assert max(audit["patches"]["prior_carries_pg"]["input_seasons"]) < 2026
    # Games-weighted pool must not collapse to the short 2025 season alone.
    assert patched["prior_carries_pg"] > 7.0
    assert patched["prior_carries_pg"] < 9.0


def test_compose_default_keeps_joint_allocation_off():
    ctx = CompositionContext(
        target_season=2026,
        depth_chart=pd.DataFrame(),
        status_overrides=pd.DataFrame(),
        artifact_provenance="test",
    )
    assert ctx.qb_joint_room_allocation is False


def test_shipped_context_production_invariance_flag():
    """Sealed production path must not silently enable joint QB allocation."""
    from src.projection.composition import shipped_context

    ctx = shipped_context(conn=None, target_season=2026)
    assert ctx.qb_joint_room_allocation is False


def test_joint_allocation_starter_share_and_conservation():
    rows = []
    for pid, tier, att, raw_g in (("starter", 1.0, 34.0, 15.0), ("backup", 2.0, 30.0, 8.0)):
        for stat, val in (("attempts", att), ("completions", att * 0.65), ("passing_yards", att * 7)):
            rows.append(
                {
                    "player_id": pid,
                    "team": "ZZ",
                    "position": "QB",
                    "stat": stat,
                    "pred_pg": val,
                    "pred_pg_low": val * 0.8,
                    "pred_pg_high": val * 1.2,
                    "projected_games": 17.0,
                    "projected_games_raw": raw_g,
                    "projected_volume_games": 17.0,
                    "depth_tier": tier,
                    "team_pass_attempts_pg_pred": 36.0,
                    "team_passing_yards_pg_pred": 250.0,
                }
            )
    # Add an RB row that must stay untouched by joint QB allocation itself
    rows.append(
        {
            "player_id": "rb1",
            "team": "ZZ",
            "position": "RB",
            "stat": "carries",
            "pred_pg": 12.0,
            "pred_pg_low": 10.0,
            "pred_pg_high": 14.0,
            "projected_games": 17.0,
            "projected_volume_games": 17.0,
            "depth_tier": 1.0,
            "team_pass_attempts_pg_pred": 36.0,
            "team_passing_yards_pg_pred": 250.0,
            "team_carries_pg_pred": 25.0,
        }
    )
    frame = pd.DataFrame(rows)
    before_rb = float(frame[(frame.player_id == "rb1")].pred_pg.iloc[0])
    allocation = AllocationParams(0.90, 0.90, 10, (2023, 2024))
    out, report = reconcile_qb_joint_room(frame, allocation=allocation, alpha=1.0)
    starter = out[(out.player_id == "starter") & (out.stat == "attempts")].iloc[0]
    backup = out[(out.player_id == "backup") & (out.stat == "attempts")].iloc[0]
    target = 36.0 * 17.0 * 0.941
    assert float(starter.pred_pg) * 17 + float(backup.pred_pg) * 17 == pytest.approx(target, rel=1e-3)
    assert float(starter.pred_pg) * 17 >= target * 0.90 - 1e-6
    assert float(out[(out.player_id == "rb1")].pred_pg.iloc[0]) == pytest.approx(before_rb)
    assert report["conservation_violations"] == []
