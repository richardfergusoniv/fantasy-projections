"""Tests for TD rate clip, exposure blend, and rookie VORP haircut."""

import pandas as pd

from src.draft_assistant.vorp import add_vorp_columns
from src.projection.depth_gating import apply_full_season_games_baseline
from src.projection.team_reconcile import (
    reconcile_pass_td_t1_lite,
    reconcile_td_rate_constraints,
)


def _qb_long(player_id="qb1", attempts=28.0, pass_tds=1.75, carries=3.6, rush_tds=0.42):
    base = {
        "player_id": player_id,
        "position": "QB",
        "season": 2026,
        "team": "JAX",
        "projected_games": 17.0,
        "projected_volume_games": 17.0,
    }
    rows = []
    for stat, val in [
        ("attempts", attempts),
        ("passing_tds", pass_tds),
        ("carries", carries),
        ("rushing_tds", rush_tds),
    ]:
        row = base.copy()
        row["stat"] = stat
        row["pred_pg"] = val
        row["pred_pg_low"] = val * 0.6
        row["pred_pg_high"] = val * 1.4
        rows.append(row)
    return pd.DataFrame(rows)


def test_td_rate_clip_caps_high_pass_td_pct():
    df = _qb_long(pass_tds=1.75)  # 6.25% on 28 att
    out = reconcile_td_rate_constraints(df)
    tds = out[out.stat == "passing_tds"].iloc[0]
    att = out[out.stat == "attempts"].iloc[0]
    assert tds.pred_pg <= att.pred_pg * 0.060 + 1e-9
    assert bool(tds.td_rate_clip_applied)


def test_td_rate_clip_caps_high_rush_td_per_carry():
    df = _qb_long(rush_tds=0.42)  # 11.7% on 3.6 car
    out = reconcile_td_rate_constraints(df)
    tds = out[out.stat == "rushing_tds"].iloc[0]
    car = out[out.stat == "carries"].iloc[0]
    assert tds.pred_pg <= car.pred_pg * 0.100 + 1e-9
    assert bool(tds.td_rate_clip_applied)


def test_exposure_blend_alpha_zero_is_full_season():
    df = pd.DataFrame({"player_id": ["a"], "projected_games": [12.0]})
    out = apply_full_season_games_baseline(df, season_games=17.0, blend_alpha=0.0)
    assert out.projected_games.iloc[0] == 17.0
    assert out.projected_games_raw.iloc[0] == 12.0


def test_exposure_blend_alpha_one_uses_raw():
    df = pd.DataFrame({"player_id": ["a"], "projected_games": [12.0]})
    out = apply_full_season_games_baseline(df, season_games=17.0, blend_alpha=1.0)
    assert out.projected_games.iloc[0] == 12.0


def test_exposure_blend_alpha_half_interpolates():
    df = pd.DataFrame({"player_id": ["a"], "projected_games": [10.0]})
    out = apply_full_season_games_baseline(df, season_games=17.0, blend_alpha=0.5)
    assert out.projected_games.iloc[0] == 13.5


def test_pass_td_t1_lite_rederives_from_attempts():
    df = _qb_long(pass_tds=1.75)
    out = reconcile_pass_td_t1_lite(df)
    tds = out[out.stat == "passing_tds"].iloc[0]
    att = out[out.stat == "attempts"].iloc[0]
    assert tds.pred_pg <= att.pred_pg * 0.060 + 1e-9


def test_shrink_qb_prior_role_rate_blends_partial_season():
    from src.projection.transitions import shrink_qb_prior_role_rate

    prior = pd.Series([2.0, 2.0])
    games = pd.Series([6.0, 14.0])
    anchor = pd.Series([1.0, 1.0])
    out = shrink_qb_prior_role_rate(prior, games, anchor, threshold=12, enabled=True)
    assert out.iloc[0] < prior.iloc[0]
    assert out.iloc[1] == prior.iloc[1]


def test_rookie_vorp_haircut_does_not_change_fantasy_pts():
    df = pd.DataFrame(
        {
            "player_id": ["vet", "rook"],
            "position": ["RB", "RB"],
            "fantasy_pts": [14.0, 14.0],
            "fantasy_pts_season": [238.0, 238.0],
            "low_confidence": [False, True],
            "source": ["veteran_model", "rookie_rule"],
        }
    )
    out = add_vorp_columns(df, team_count=12, rookie_rank_scale=0.85)
    vet = out[out.player_id == "vet"].iloc[0]
    rook = out[out.player_id == "rook"].iloc[0]
    assert vet.fantasy_pts == 14.0
    assert rook.fantasy_pts == 14.0
    assert rook.vorp_input_pts == 238.0 * 0.85
    assert rook.vorp < vet.vorp
