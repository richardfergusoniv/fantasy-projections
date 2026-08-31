"""Shadow 0A team-level receiving target pool."""
from __future__ import annotations

import pandas as pd

from src.projection.team_reconcile import TARGETS_PER_ATTEMPT


def estimate_team_target_pool(
    long_board: pd.DataFrame,
    *,
    team_col: str = "team",
    pass_attempts_col: str = "team_pass_attempts_pg_pred",
    games_col: str = "projected_games",
) -> pd.DataFrame:
    """Estimate team season target pool from team passing environment."""
    frame = long_board.copy()
    if pass_attempts_col in frame.columns:
        grouped = (
            frame.groupby(team_col, observed=True)
            .agg(
                team_pass_attempts_pg=(pass_attempts_col, "first"),
                projected_games=(games_col, "mean"),
            )
            .reset_index()
        )
        grouped["team_targets_season_pred"] = (
            pd.to_numeric(grouped["team_pass_attempts_pg"], errors="coerce").fillna(0.0)
            * TARGETS_PER_ATTEMPT
            * pd.to_numeric(grouped["projected_games"], errors="coerce").fillna(17.0)
        )
        return grouped[[team_col, "team_targets_season_pred"]]

    # Fallback: sum incumbent player target season totals when team anchors absent.
    target_rows = frame[frame["stat"].eq("targets")].copy()
    if target_rows.empty:
        return pd.DataFrame(columns=[team_col, "team_targets_season_pred"])
    target_rows["pred_season"] = (
        pd.to_numeric(target_rows["pred_pg"], errors="coerce").fillna(0.0)
        * pd.to_numeric(target_rows[games_col], errors="coerce").fillna(17.0)
    )
    return (
        target_rows.groupby(team_col, observed=True)["pred_season"]
        .sum()
        .rename("team_targets_season_pred")
        .reset_index()
    )
