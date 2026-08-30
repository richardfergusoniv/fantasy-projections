"""Deterministic per-draw positional ranks for simulated overlays."""
from __future__ import annotations

import pandas as pd

TIE_POLICY = "minimum_competition_rank"
# Approved finish probabilities use pandas rank(method="first"). Simulated VORP
# rank moments use minimum competition rank per Phase 2 contract.
FINISH_PROBABILITY_TIE_POLICY = "first_occurrence"

FINISH_PROBABILITY_FIELDS = (
    "p_finish_top6",
    "p_finish_top12",
    "p_finish_top24",
    "p_finish_top36",
    "p_finish_top48",
)
SIMULATED_RANK_FIELDS = ("expected_pos_rank", "median_pos_rank")


def simulation_rank_metadata() -> dict:
    """Product/technical metadata for rank-derived simulation fields.

    ``p_finish_*`` and simulated positional-rank moments intentionally use
    different tie conventions. Consumers must not assume one policy applies to
    every rank-derived overlay field.
    """
    return {
        "summary": (
            "Finish probabilities and simulated positional-rank moments use "
            "different per-draw tie policies by design."
        ),
        "finish_probability_fields": {
            "fields": list(FINISH_PROBABILITY_FIELDS),
            "tie_policy": FINISH_PROBABILITY_TIE_POLICY,
            "pandas_rank_method": "first",
            "description": (
                "Per-draw positional rank uses first occurrence on ties "
                "(stable row order). A player qualifies for top-N when "
                "positional_rank_draw <= N under this policy."
            ),
        },
        "simulated_positional_rank_fields": {
            "fields": list(SIMULATED_RANK_FIELDS),
            "tie_policy": TIE_POLICY,
            "pandas_rank_method": "min",
            "description": (
                "Per-draw positional rank uses minimum competition rank on ties "
                "(equal points share a rank; next rank skips). "
                "expected_pos_rank is the draw mean; median_pos_rank is the "
                "draw median under this policy."
            ),
        },
        "top_n_cutoff_semantics": "positional_rank_draw <= N",
    }
def rank_positional_draws(
    frame: pd.DataFrame,
    *,
    points_col: str = "fantasy_pts_season",
    draw_col: str = "draw",
    position_col: str = "position",
    player_col: str = "player_id",
    tolerance: float = 0.0,
) -> pd.Series:
    """Minimum competition rank within each draw and position.

    Equal points within ``tolerance`` share the same rank; the next rank skips.
    Stable ``player_id`` ordering is used only for sort order, not rank values.
    """
    if frame.empty:
        return pd.Series(dtype=float)
    work = frame[[draw_col, position_col, player_col, points_col]].copy()
    work[points_col] = pd.to_numeric(work[points_col], errors="coerce")
    if tolerance > 0:
        work["_rounded_points"] = (work[points_col] / tolerance).round() * tolerance
        rank_col = "_rounded_points"
    else:
        rank_col = points_col
    work = work.sort_values(
        [draw_col, position_col, rank_col, player_col],
        ascending=[True, True, False, True],
    )
    return work.groupby([draw_col, position_col], observed=True)[rank_col].rank(
        ascending=False,
        method="min",
    )


def finish_probability_rank(
    frame: pd.DataFrame,
    *,
    points_col: str = "fantasy_pts_season",
) -> pd.Series:
    """Legacy approved finish-probability rank semantics (method=first)."""
    return (
        frame.groupby(["draw", "position"], observed=True)[points_col]
        .rank(ascending=False, method="first")
    )


def top_n_finish_rate(ranks: pd.Series, cutoff: int) -> float:
    return float((pd.to_numeric(ranks, errors="coerce") <= cutoff).mean())
