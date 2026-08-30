"""Recenter v3 fantasy-point draws on a selected accuracy-first forecast.

The transform preserves v3 residual shape while anchoring the median to the
displayed board:

    recentered = selected + (draw - v3_p50)

Values are floored at zero, then each player receives a median correction so
the recentered p50 matches ``selected`` exactly.
"""
from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np
import pandas as pd

TRANSFORM_VERSION = "v1_median_correction"
POINTS_COL = "fantasy_pts_season"


def player_draw_medians(
    draws: pd.DataFrame,
    *,
    points_col: str = POINTS_COL,
    player_col: str = "player_id",
) -> pd.Series:
    """Per-player median fantasy points across simulation draws."""
    return (
        draws.groupby(player_col, observed=True)[points_col]
        .median()
        .astype(float)
    )


def recenter_draws(
    draws: pd.DataFrame,
    selected_points: Mapping[str, float] | pd.Series,
    *,
    points_col: str = POINTS_COL,
    player_col: str = "player_id",
) -> pd.DataFrame:
    """Shift draws so residual uncertainty is anchored on ``selected_points``.

    Parameters
    ----------
    draws
        Long-form simulation output with ``player_id``, ``draw``, and
        ``points_col``.
    selected_points
        Displayed season fantasy points keyed by ``player_id``.
    """
    if draws.empty:
        return draws.copy()

    selected = pd.Series(selected_points, dtype=float)
    selected.index = selected.index.astype(str)
    v3_p50 = player_draw_medians(draws, points_col=points_col, player_col=player_col)
    v3_p50.index = v3_p50.index.astype(str)

    out = draws.copy()
    out[player_col] = out[player_col].astype(str)
    anchor = out[player_col].map(selected).astype(float)
    baseline = out[player_col].map(v3_p50).astype(float)
    residual = pd.to_numeric(out[points_col], errors="coerce").astype(float) - baseline
    raw = anchor + residual
    out[points_col] = np.maximum(raw.to_numpy(dtype=float), 0.0)

    # Median correction after flooring can require iteration when many draws hit zero.
    for _ in range(12):
        post_medians = (
            out.groupby(player_col, observed=True)[points_col]
            .median()
            .astype(float)
        )
        correction = selected.reindex(post_medians.index).astype(float) - post_medians
        if correction.abs().max() < 1e-9:
            break
        out[points_col] = out[points_col] + out[player_col].map(correction).fillna(0.0)
        out[points_col] = np.maximum(out[points_col].to_numpy(dtype=float), 0.0)
    return out


def recenter_summary(
    draws: pd.DataFrame,
    selected_points: Mapping[str, float] | pd.Series,
    *,
    points_col: str = POINTS_COL,
) -> pd.DataFrame:
    """Percentile summary from recentered draws."""
    from src.projection.inference.simulate import summarize_simulations

    recentered = recenter_draws(draws, selected_points, points_col=points_col)
    return summarize_simulations(recentered)


def board_points_series(board: pd.DataFrame) -> pd.Series:
    """Extract player_id -> fantasy_pts_season from a fantasy-points board."""
    if "fantasy_pts_season" not in board.columns:
        raise ValueError("board is missing fantasy_pts_season")
    out = (
        board.drop_duplicates("player_id")
        .set_index(board["player_id"].astype(str))["fantasy_pts_season"]
        .astype(float)
    )
    return out


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
