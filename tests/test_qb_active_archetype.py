"""Tests for active-start / archetype QB experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.projection.composition import CompositionContext, shipped_context
from src.projection.qb_active_archetype.active_rates import (
    annotate_weekly_activity,
    build_active_season_rates,
    expected_availability,
    player_decomposition,
)
from src.projection.qb_active_archetype.allocation_v2 import reconcile_qb_joint_room_v2
from src.projection.qb_active_archetype.archetypes import classify_archetype, hierarchical_rush_priors
from src.projection.qb_active_archetype.thresholds import GATES, thresholds_dict
from src.projection.qb_repair.allocation import AllocationParams
from src.projection.qb_repair.apply_board import non_qb_invariance_check


def test_thresholds_predeclared_and_frozen_shape():
    d = thresholds_dict()
    assert d["gates"]["use_2026_for_selection"] is False
    assert GATES.overall_mae_non_inferiority_tol == 0.02
    assert GATES.holdout_bootstrap_ci_must_exclude_zero is True


def test_active_start_not_diluted_by_missed_games():
    weekly = pd.DataFrame(
        [
            {"player_id": "qb", "player_display_name": "Q", "recent_team": "ZZ", "season": 2024, "week": 1, "season_type": "REG",
             "attempts": 35, "completions": 22, "passing_yards": 280, "passing_tds": 2, "interceptions": 0, "carries": 3, "rushing_yards": 12, "rushing_tds": 0},
            {"player_id": "qb", "player_display_name": "Q", "recent_team": "ZZ", "season": 2024, "week": 2, "season_type": "REG",
             "attempts": 38, "completions": 24, "passing_yards": 300, "passing_tds": 3, "interceptions": 1, "carries": 2, "rushing_yards": 8, "rushing_tds": 0},
            # injured weeks absent — not present as zero-attempt rows
        ]
    )
    rates = build_active_season_rates(weekly)
    row = rates.iloc[0]
    assert row.active_starts == 2
    assert row.attempts_per_active == pytest.approx(36.5)
    # Conflated with only active weeks matches; missed games simply absent.


def test_partial_exit_flag_without_future_info():
    weekly = pd.DataFrame(
        [
            {"player_id": "qb", "attempts": 35, "carries": 2, "week": 1, "season": 2024, "season_type": "REG",
             "completions": 20, "passing_yards": 250, "passing_tds": 1, "interceptions": 0, "rushing_yards": 5, "rushing_tds": 0,
             "player_display_name": "Q", "recent_team": "ZZ"},
            {"player_id": "qb", "attempts": 5, "carries": 0, "week": 2, "season": 2024, "season_type": "REG",
             "completions": 3, "passing_yards": 40, "passing_tds": 0, "interceptions": 0, "rushing_yards": 0, "rushing_tds": 0,
             "player_display_name": "Q", "recent_team": "ZZ"},
        ]
    )
    ann = annotate_weekly_activity(weekly)
    assert bool(ann.loc[0, "active_start"]) is True
    assert bool(ann.loc[1, "partial_exit"]) is True


def test_archetype_uses_only_prior_seasons():
    hist = pd.DataFrame(
        [
            {"player_id": "m1", "season": 2022, "active_starts": 16, "designed_carries_per_active": 5.5, "scramble_per_dropback": 0.09, "carries_per_active": 9.0, "rushing_yards_per_active": 50.0, "rushing_tds_per_active": 0.3, "designed_ypc": 4.0, "scramble_ypa": 7.0},
            {"player_id": "m1", "season": 2023, "active_starts": 16, "designed_carries_per_active": 5.2, "scramble_per_dropback": 0.10, "carries_per_active": 9.1, "rushing_yards_per_active": 51.0, "rushing_tds_per_active": 0.3, "designed_ypc": 4.1, "scramble_ypa": 7.2},
            {"player_id": "m1", "season": 2024, "active_starts": 8, "designed_carries_per_active": 2.0, "scramble_per_dropback": 0.05, "carries_per_active": 4.0, "rushing_yards_per_active": 20.0, "rushing_tds_per_active": 0.1, "designed_ypc": 3.0, "scramble_ypa": 5.0},
            {"player_id": "p1", "season": 2022, "active_starts": 16, "designed_carries_per_active": 0.5, "scramble_per_dropback": 0.02, "carries_per_active": 1.5, "rushing_yards_per_active": 5.0, "rushing_tds_per_active": 0.0, "designed_ypc": 3.0, "scramble_ypa": 4.0},
            {"player_id": "p1", "season": 2023, "active_starts": 17, "designed_carries_per_active": 0.4, "scramble_per_dropback": 0.02, "carries_per_active": 1.2, "rushing_yards_per_active": 4.0, "rushing_tds_per_active": 0.0, "designed_ypc": 3.0, "scramble_ypa": 4.0},
        ]
    )
    meta = classify_archetype(hist, player_id="m1", target_season=2025)
    assert meta["archetype"] == "designed_runner"
    assert max(meta["input_seasons"]) < 2025
    priors = hierarchical_rush_priors(hist, player_id="m1", target_season=2025)
    # Short 2024 season cannot erase designed-runner prior.
    assert priors["priors"]["designed_carries_per_active"] > 3.5
    pocket = classify_archetype(hist, player_id="p1", target_season=2025)
    assert pocket["archetype"] == "pocket_passer"
    # Pocket must not inherit dual-threat carries prior
    p_priors = hierarchical_rush_priors(hist, player_id="p1", target_season=2025)
    assert p_priors["priors"]["carries_per_active"] < 3.0


def test_joint_v2_backup_cannot_reduce_starter_active_rate():
    rows = []
    for pid, tier, att in (("starter", 1.0, 36.0), ("backup", 2.0, 28.0)):
        for stat, val in (("attempts", att), ("passing_yards", att * 7)):
            rows.append(
                {
                    "player_id": pid,
                    "team": "ZZ",
                    "position": "QB",
                    "stat": stat,
                    "pred_pg": val,
                    "pred_pg_low": val * 0.9,
                    "pred_pg_high": val * 1.1,
                    "projected_games": 17.0,
                    "projected_games_raw": 15.0 if pid == "starter" else 4.0,
                    "projected_volume_games": 17.0,
                    "depth_tier": tier,
                    "team_pass_attempts_pg_pred": 36.0,
                    "team_passing_yards_pg_pred": 250.0,
                }
            )
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
    before_rb = float(frame[frame.player_id == "rb1"].pred_pg.iloc[0])
    out, report = reconcile_qb_joint_room_v2(
        frame,
        allocation=AllocationParams(0.90, 0.90, 10, (2023, 2024)),
        alpha=1.0,
        expected_active_starts={"starter": 15.0},
    )
    starter_att = float(out[(out.player_id == "starter") & (out.stat == "attempts")].pred_pg.iloc[0])
    # Season claim under exposure 17 should be ~ active 36 * 15 = 540 → pg = 540/17 ≈ 31.76
    # which is the board rate; active rate itself is preserved in the season identity.
    assert starter_att * 17 >= 36.0 * 15.0 - 1e-3 or starter_att >= 30.0
    assert float(out[out.player_id == "rb1"].pred_pg.iloc[0]) == pytest.approx(before_rb)
    # Team conservation
    qb_att = out[(out.team == "ZZ") & (out.stat == "attempts") & (out.position == "QB")]
    total = float((qb_att.pred_pg * 17).sum())
    target = 36.0 * 17 * 0.941
    assert total == pytest.approx(target, rel=1e-3)


def test_shipped_defaults_unchanged():
    ctx = shipped_context(conn=None, target_season=2026)
    assert ctx.qb_joint_room_allocation is False
    bare = CompositionContext(
        target_season=2026,
        depth_chart=pd.DataFrame(),
        status_overrides=pd.DataFrame(),
        artifact_provenance="test",
    )
    assert bare.qb_joint_room_allocation is False


def test_expected_availability_shrinks_short_seasons():
    hist = pd.DataFrame(
        [
            {"player_id": "b", "season": 2022, "active_starts": 16, "partial_exit_rate": 0.05},
            {"player_id": "b", "season": 2023, "active_starts": 10, "partial_exit_rate": 0.1},
            {"player_id": "b", "season": 2024, "active_starts": 17, "partial_exit_rate": 0.05},
            {"player_id": "b", "season": 2025, "active_starts": 8, "partial_exit_rate": 0.1},
        ]
    )
    avail = expected_availability(hist, player_id="b", target_season=2026)
    assert 12.0 <= avail["expected_active_starts"] <= 16.5
    assert max(avail["input_seasons"]) < 2026
