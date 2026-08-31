"""Simulated draft-value metrics from recentered fantasy-point draws."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.draft_assistant.vorp import DEFAULT_TEAM_COUNT, add_vorp_columns

FINISH_CUTOFFS: tuple[int, ...] = (6, 12, 24, 36, 48)
POINTS_COL = "fantasy_pts_season"


def _replacement_points_by_position(board: pd.DataFrame, *, team_count: int) -> dict[str, float]:
    """Fixed replacement season points from the displayed board."""
    enriched = add_vorp_columns(board.copy(), team_count=team_count, floor_at_zero=False)
    out: dict[str, float] = {}
    for pos in ("QB", "RB", "WR", "TE"):
        sub = enriched[enriched["position"] == pos]
        if sub.empty:
            continue
        val = pd.to_numeric(sub["replacement_pts"], errors="coerce").dropna()
        if not val.empty:
            out[pos] = float(val.iloc[0])
    return out


def compute_finish_probabilities(
    draws: pd.DataFrame,
    *,
    cutoffs: Iterable[int] = FINISH_CUTOFFS,
    points_col: str = POINTS_COL,
) -> pd.DataFrame:
    """Probability of finishing at or below each cutoff within position."""
    if draws.empty:
        return pd.DataFrame(columns=["player_id"])

    needed = {"player_id", "position", "draw", points_col}
    missing = needed - set(draws.columns)
    if missing:
        raise ValueError(f"draws missing columns: {sorted(missing)}")

    frame = draws[["player_id", "position", "draw", points_col]].copy()
    frame["player_id"] = frame["player_id"].astype(str)
    ranks = (
        frame.groupby(["draw", "position"], observed=True)[points_col]
        .rank(ascending=False, method="first")
    )
    frame["pos_rank_draw"] = ranks

    rows: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        col = f"p_finish_top{cutoff}"
        probs = (
            frame.groupby("player_id", observed=True)["pos_rank_draw"]
            .apply(lambda s, c=cutoff: float((s <= c).mean()))
            .rename(col)
            .reset_index()
        )
        rows.append(probs)

    out = rows[0]
    for extra in rows[1:]:
        out = out.merge(extra, on="player_id", how="outer")
    return out


def compute_simulated_vorp_metrics(
    draws: pd.DataFrame,
    board: pd.DataFrame,
    *,
    team_count: int = DEFAULT_TEAM_COUNT,
    points_col: str = POINTS_COL,
) -> pd.DataFrame:
    """Aggregate simulated VORP and positional-rank metrics per player."""
    if draws.empty:
        return pd.DataFrame(columns=["player_id"])

    replacement = _replacement_points_by_position(board, team_count=team_count)
    frame = draws[["player_id", "position", "draw", points_col]].copy()
    frame["player_id"] = frame["player_id"].astype(str)
    frame["replacement_pts"] = frame["position"].map(replacement).astype(float)
    frame["sim_vorp_draw"] = frame[points_col] - frame["replacement_pts"]

    ranks = (
        frame.groupby(["draw", "position"], observed=True)[points_col]
        .rank(ascending=False, method="first")
    )
    frame["pos_rank_draw"] = ranks

    grouped = frame.groupby("player_id", observed=True)
    out = grouped["sim_vorp_draw"].agg(
        sim_vorp_p10=lambda s: float(s.quantile(0.10)),
        sim_vorp_p50=lambda s: float(s.quantile(0.50)),
        sim_vorp_p90=lambda s: float(s.quantile(0.90)),
        p_vorp_positive=lambda s: float((s > 0).mean()),
    )
    rank_stats = grouped["pos_rank_draw"].agg(
        expected_pos_rank="mean",
        median_pos_rank="median",
    )
    out = out.join(rank_stats).reset_index()
    out["expected_pos_rank"] = out["expected_pos_rank"].round(2)
    out["median_pos_rank"] = out["median_pos_rank"].round(2)
    return out


def compute_draft_value_overlay(
    draws: pd.DataFrame,
    board: pd.DataFrame,
    *,
    team_count: int = DEFAULT_TEAM_COUNT,
) -> pd.DataFrame:
    """Merge finish probabilities and simulated VORP metrics."""
    finish = compute_finish_probabilities(draws)
    vorp = compute_simulated_vorp_metrics(draws, board, team_count=team_count)
    if finish.empty:
        return vorp
    if vorp.empty:
        return finish
    return finish.merge(vorp, on="player_id", how="outer")
