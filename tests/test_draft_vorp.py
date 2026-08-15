"""Tests for draft VORP / replacement baselines."""

import pandas as pd

from src.draft_assistant.tiers import add_tier_columns
from src.draft_assistant.vorp import (
    OVERALL_VORP_TIER_GAP,
    add_vorp_columns,
    replacement_rank,
    replacement_ranks,
)


def test_replacement_ranks_for_common_league_sizes():
    assert replacement_ranks(12) == {"QB": 13, "RB": 29, "WR": 43, "TE": 14}
    assert replacement_rank("QB", 10) == 11
    assert replacement_rank("RB", 10) == 25  # floor(20 + 4) + 1
    assert replacement_rank("WR", 14) == 50  # floor(42 + 7) + 1
    assert replacement_rank("TE", 14) == 16  # floor(14 + 1.4) + 1


def test_vorp_ranks_scarce_rb_above_high_ppg_qb():
    """Elite RB PPG surplus beats a high-PPG mid QB on overall VORP."""
    rows = []
    # 13 QBs so replacement is QB13
    for i in range(13):
        rows.append(
            {
                "player_id": f"qb{i}",
                "display_name": f"QB {i}",
                "position": "QB",
                "fantasy_pts": 22.0 - i * 0.3,
            }
        )
    # Mid QB still high PPG but only modest VORP vs QB13
    rows[6]["player_id"] = "mid_qb"
    rows[6]["display_name"] = "Mid QB"
    rows[6]["fantasy_pts"] = 20.5

    # Enough RBs for RB29 replacement; elite RB far above
    for i in range(30):
        ppg = 18.5 if i == 0 else max(8.0, 14.0 - i * 0.15)
        rows.append(
            {
                "player_id": "elite_rb" if i == 0 else f"rb{i}",
                "display_name": "Elite RB" if i == 0 else f"RB {i}",
                "position": "RB",
                "fantasy_pts": ppg,
            }
        )

    df = pd.DataFrame(rows)
    df = add_vorp_columns(df, team_count=12)
    df = add_tier_columns(
        df,
        overall_points_col="vorp",
        overall_gap=OVERALL_VORP_TIER_GAP,
    )

    elite_rb = df[df.player_id == "elite_rb"].iloc[0]
    mid_qb = df[df.player_id == "mid_qb"].iloc[0]

    assert elite_rb.vorp > mid_qb.vorp
    assert elite_rb.overall_rank < mid_qb.overall_rank
    assert mid_qb.fantasy_pts > elite_rb.fantasy_pts


def test_add_vorp_floors_negative_at_zero():
    df = pd.DataFrame(
        {
            "player_id": ["a", "b"],
            "position": ["TE", "TE"],
            "fantasy_pts": [12.0, 4.0],
        }
    )
    out = add_vorp_columns(df, team_count=12)
    # Only 2 TEs; replacement is TE14 but kth clamps to last -> 4.0
    assert out.loc[out.player_id == "b", "vorp"].iloc[0] == 0.0
    assert out.loc[out.player_id == "a", "vorp"].iloc[0] == 8.0


def test_add_vorp_defaults_to_fantasy_pts_column():
    df = pd.DataFrame(
        {
            "player_id": ["a", "b"],
            "position": ["WR", "WR"],
            "fantasy_pts": [15.0, 10.0],
            "fantasy_pts_season": [250.0, 50.0],
        }
    )
    out = add_vorp_columns(df, team_count=12)
    # Baseline clamps to WR2 = 10 PPG, not season totals
    assert out.loc[out.player_id == "a", "vorp"].iloc[0] == 5.0
    assert out.loc[out.player_id == "a", "replacement_pts"].iloc[0] == 10.0
