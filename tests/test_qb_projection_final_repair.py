"""Focused unit tests for the 2026 QB projection final repair."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.projection.qb_repair.allocation import (
    AllocationParams,
    estimate_starter_backup_shares,
    reconcile_qb_volume_with_allocation,
)
from src.projection.qb_repair.apply_board import non_qb_invariance_check
from src.projection.qb_repair.history import history_before, per_game_rates
from src.projection.qb_repair.rate_prior import (
    apply_qb_rate_prior,
    build_qb_rate_priors,
    classify_qb_archetype,
)


def _history_frame() -> pd.DataFrame:
    rows = []
    # Established mobile QB across 2022-2025
    for season, games, car, ry, att, py in [
        (2022, 16, 9.0, 55.0, 28.0, 220.0),
        (2023, 16, 9.2, 51.0, 28.5, 230.0),
        (2024, 17, 8.2, 54.0, 27.9, 245.0),
        (2025, 8, 5.0, 27.0, 26.0, 200.0),  # partial
    ]:
        rows.append(
            {
                "player_id": "mobile1",
                "season": season,
                "display_name": "Mobile One",
                "team": "AAA",
                "games": games,
                "attempts": att * games,
                "completions": 0.65 * att * games,
                "passing_yards": py * games,
                "passing_tds": 0.05 * att * games,
                "interceptions": 0.02 * att * games,
                "carries": car * games,
                "rushing_yards": ry * games,
                "rushing_tds": 0.03 * car * games,
                "designed_carries": 0.55 * car * games,
                "scramble_carries": 0.45 * car * games,
                "designed_rushing_yards": 0.5 * ry * games,
                "scramble_rushing_yards": 0.5 * ry * games,
            }
        )
    # Pocket QB
    for season, games in [(2022, 16), (2023, 16), (2024, 17), (2025, 17)]:
        rows.append(
            {
                "player_id": "pocket1",
                "season": season,
                "display_name": "Pocket One",
                "team": "BBB",
                "games": games,
                "attempts": 35 * games,
                "completions": 23 * games,
                "passing_yards": 260 * games,
                "passing_tds": 1.8 * games,
                "interceptions": 0.7 * games,
                "carries": 2.2 * games,
                "rushing_yards": 8.0 * games,
                "rushing_tds": 0.1 * games,
                "designed_carries": 1.0 * games,
                "scramble_carries": 1.2 * games,
                "designed_rushing_yards": 2.0 * games,
                "scramble_rushing_yards": 6.0 * games,
            }
        )
    # Backup volume on AAA in each season
    for season in (2022, 2023, 2024, 2025):
        rows.append(
            {
                "player_id": f"backup_{season}",
                "season": season,
                "display_name": "Backup",
                "team": "AAA",
                "games": 4,
                "attempts": 80,
                "completions": 40,
                "passing_yards": 500,
                "passing_tds": 2,
                "interceptions": 2,
                "carries": 5,
                "rushing_yards": 10,
                "rushing_tds": 0,
            }
        )
    return pd.DataFrame(rows)


def test_history_before_enforces_leakage_boundary():
    hist = _history_frame()
    prior = history_before(hist, 2025)
    assert prior["season"].max() == 2024
    assert (prior["season"] >= 2025).sum() == 0


def test_partial_season_weights_keep_per_game_info():
    hist = per_game_rates(_history_frame())
    priors = build_qb_rate_priors(
        target_season=2026,
        player_ids=["mobile1"],
        history=hist,
        established_only=True,
    )
    rec = priors["mobile1"]
    assert rec.applied
    assert 2025 in rec.input_seasons
    # Games-weighted prior should sit well above the partial 2025 5.0 carries/g.
    assert rec.components["carries_pg"] > 6.5
    assert rec.sample_games >= 16


def test_mobile_archetype_and_rush_prior_application():
    hist = _history_frame()
    assert classify_qb_archetype(hist, "mobile1", target_season=2026) == "mobile"
    assert classify_qb_archetype(hist, "pocket1", target_season=2026) == "pocket"
    priors = build_qb_rate_priors(
        target_season=2026, player_ids=["mobile1", "pocket1"], history=hist
    )
    long = pd.DataFrame(
        [
            {"player_id": "mobile1", "position": "QB", "stat": "carries", "pred_pg": 4.5, "depth_tier": 1.0},
            {"player_id": "mobile1", "position": "QB", "stat": "rushing_yards", "pred_pg": 22.0, "depth_tier": 1.0},
            {"player_id": "mobile1", "position": "QB", "stat": "attempts", "pred_pg": 24.0, "depth_tier": 1.0},
            {"player_id": "pocket1", "position": "QB", "stat": "carries", "pred_pg": 2.0, "depth_tier": 1.0},
        ]
    )
    out, audit = apply_qb_rate_prior(long, priors, mobile_rushing_only=True)
    mobile_car = float(out[(out.player_id == "mobile1") & (out.stat == "carries")].pred_pg.iloc[0])
    pocket_car = float(out[(out.player_id == "pocket1") & (out.stat == "carries")].pred_pg.iloc[0])
    assert mobile_car > 4.5
    assert pocket_car == pytest.approx(2.0)
    # Passing attempts untouched in mobile-rushing-only mode.
    att = float(out[(out.player_id == "mobile1") & (out.stat == "attempts")].pred_pg.iloc[0])
    assert att == pytest.approx(24.0)
    assert any(a["player_id"] == "mobile1" and a["applied"] for a in audit)


def test_qb1_backup_allocation_protects_starter_and_conserves():
    # Inflated backup raw forecast that would crush the starter under flat scaling.
    rows = []
    for pid, tier, att in (("starter", 1.0, 34.0), ("backup", 2.0, 28.0)):
        for stat, val in (("attempts", att), ("completions", att * 0.65), ("passing_yards", att * 7.0)):
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
                    "projected_volume_games": 17.0,
                    "depth_tier": tier,
                    "team_pass_attempts_pg_pred": 36.0,
                    "team_passing_yards_pg_pred": 250.0,
                }
            )
    frame = pd.DataFrame(rows)
    allocation = AllocationParams(
        starter_attempt_share=0.90,
        starter_yard_share=0.90,
        n_team_seasons=10,
        fit_seasons=(2023, 2024),
    )
    out, report = reconcile_qb_volume_with_allocation(frame, allocation=allocation, alpha=1.0)
    starter = out[(out.player_id == "starter") & (out.stat == "attempts")].iloc[0]
    backup = out[(out.player_id == "backup") & (out.stat == "attempts")].iloc[0]
    target = 36.0 * 17.0 * 0.941
    starter_season = float(starter.pred_pg) * 17.0
    backup_season = float(backup.pred_pg) * 17.0
    assert starter_season + backup_season == pytest.approx(target, rel=1e-3)
    assert starter_season >= target * 0.90 - 1e-6
    assert starter_season > backup_season
    assert report["starters_protected"] >= 1


def test_estimate_allocation_uses_only_prior_seasons():
    hist = _history_frame()
    params = estimate_starter_backup_shares(target_season=2025, history=hist)
    assert params.n_team_seasons > 0
    assert max(params.fit_seasons) < 2025
    assert 0.70 <= params.starter_attempt_share <= 0.985


def test_non_qb_invariance():
    base = pd.DataFrame(
        [
            {"player_id": "r1", "position": "RB", "stat": "carries", "pred_pg": 15.0},
            {"player_id": "w1", "position": "WR", "stat": "targets", "pred_pg": 8.0},
            {"player_id": "q1", "position": "QB", "stat": "attempts", "pred_pg": 30.0},
        ]
    )
    cand = base.copy()
    cand.loc[cand.player_id == "q1", "pred_pg"] = 33.0
    assert non_qb_invariance_check(baseline_long=base, candidate_long=cand)["pass"] is True
    cand2 = base.copy()
    cand2.loc[cand2.player_id == "r1", "pred_pg"] = 16.0
    assert non_qb_invariance_check(baseline_long=base, candidate_long=cand2)["pass"] is False
