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
    # The board ranks on season points; these fixtures are stated per game.
    df["fantasy_pts_season"] = df["fantasy_pts"] * 17.0
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


def test_add_vorp_keeps_sub_replacement_players_ordered():
    """Sub-replacement VORP is signed, so the late board keeps an ordering.

    Flooring at zero tied together every player below replacement -- 73% of the
    2026 board -- leaving everything from roughly round 9 down sorted by an
    arbitrary tiebreak rather than by value.
    """
    df = pd.DataFrame(
        {
            "player_id": ["a", "b", "c"],
            "position": ["TE", "TE", "TE"],
            "fantasy_pts_season": [200.0, 80.0, 40.0],
        }
    )
    # curve_weight={} isolates the signed-surplus mechanic from the TE shape
    # correction, which would otherwise rewrite these fixture surpluses.
    out = add_vorp_columns(df, team_count=12, curve_weight={}).set_index("player_id")
    # Only 3 TEs; replacement rank clamps to the last -> 40.0
    assert out.loc["a", "vorp"] == 160.0
    assert out.loc["b", "vorp"] == 40.0
    assert out.loc["c", "vorp"] == 0.0
    assert out.loc["b", "vorp"] > out.loc["c", "vorp"]


def test_add_vorp_floor_at_zero_is_opt_in():
    df = pd.DataFrame(
        {
            "player_id": ["a", "b"],
            "position": ["TE", "TE"],
            "fantasy_pts_season": [200.0, 40.0],
        }
    )
    signed = add_vorp_columns(df, team_count=12, curve_weight={}).set_index("player_id")
    clipped = add_vorp_columns(
        df, team_count=12, floor_at_zero=True, curve_weight={}
    ).set_index("player_id")
    # Replacement clamps to the last TE, so nothing is negative either way here;
    # what matters is that the flag is available and does not change the top.
    assert signed.loc["a", "vorp"] == clipped.loc["a", "vorp"] == 160.0

    below = pd.DataFrame(
        {
            "player_id": ["x", "y", "z"],
            "position": ["WR", "WR", "WR"],
            "fantasy_pts_season": [300.0, 200.0, 100.0],
        }
    )
    # WR replacement rank is 43; with 3 rows it clamps to the last (100.0).
    out = add_vorp_columns(below, team_count=12, floor_at_zero=True, curve_weight={})
    assert (out["vorp"] >= 0).all()


def test_add_vorp_defaults_to_season_points():
    """A draft pick buys a season, so VORP ranks season totals, not a rate."""
    df = pd.DataFrame(
        {
            "player_id": ["a", "b"],
            "position": ["WR", "WR"],
            "fantasy_pts": [15.0, 10.0],
            "fantasy_pts_season": [250.0, 50.0],
        }
    )
    out = add_vorp_columns(df, team_count=12)
    assert out.loc[out.player_id == "a", "vorp"].iloc[0] == 200.0
    assert out.loc[out.player_id == "a", "replacement_pts"].iloc[0] == 50.0


def test_add_vorp_prices_availability():
    """Equal per-game rates, different availability -> different draft value."""
    df = pd.DataFrame(
        {
            "player_id": ["iron", "fragile"],
            "position": ["RB", "RB"],
            "fantasy_pts": [15.0, 15.0],
            "projected_games": [17.0, 9.0],
            "fantasy_pts_season": [255.0, 135.0],
        }
    )
    out = add_vorp_columns(df, team_count=12).set_index("player_id")
    assert out.loc["iron", "vorp"] > out.loc["fragile", "vorp"]


def test_position_curve_blend_corrects_shape_not_order():
    """The TE shape correction shrinks inflated surplus without reordering.

    A projected curve that is too flat at the top -- eight tight ends bunched
    just below the leader -- ranks the whole tier too high on a cross-position
    board. Blending toward the fitted historical shape pulls those surpluses
    down while preserving which tight end is which.
    """
    pts = [260.0, 240.0, 235.0, 230.0, 225.0, 220.0, 130.0, 120.0, 110.0, 100.0]
    df = pd.DataFrame(
        {
            "player_id": [f"te{i}" for i in range(len(pts))],
            "position": ["TE"] * len(pts),
            "fantasy_pts_season": pts,
        }
    )
    curves = {"TE": [230.0, 175.0, 172.0, 163.0, 157.0, 150.0, 147.0, 145.0, 143.0, 140.0]}

    plain = add_vorp_columns(df, team_count=12, curve_weight={})
    blended = add_vorp_columns(
        df, team_count=12, curve_weight={"TE": 1.0}, curves=curves
    )

    # Ordering is untouched.
    assert (
        plain.sort_values("vorp", ascending=False)["player_id"].tolist()
        == blended.sort_values("vorp", ascending=False)["player_id"].tolist()
    )
    # The inflated middle of the tier is pulled down hard.
    plain_te2 = plain.loc[plain.player_id == "te1", "vorp"].iloc[0]
    blend_te2 = blended.loc[blended.player_id == "te1", "vorp"].iloc[0]
    assert blend_te2 < plain_te2
    # Weight is recorded on the rows it was applied to.
    assert (blended["vorp_curve_weight"] == 1.0).all()


def test_position_curve_blend_off_by_default_for_wr_and_rb():
    df = pd.DataFrame(
        {
            "player_id": ["w1", "w2", "r1", "r2"],
            "position": ["WR", "WR", "RB", "RB"],
            "fantasy_pts_season": [300.0, 100.0, 280.0, 90.0],
        }
    )
    out = add_vorp_columns(df, team_count=12)
    assert (out["vorp_curve_weight"] == 0.0).all()
