"""Weekly feature pipeline for hierarchical partial pooling."""
from __future__ import annotations

import pandas as pd

from src.projection.data_prep import get_conn
from src.projection.features import load_weekly_usage


def build_player_week_features(conn=None, seasons: list[int] | None = None) -> pd.DataFrame:
    """Aggregate player-week usage with rolling share blends."""
    own_conn = conn is None
    if own_conn:
        conn = get_conn()
    usage = load_weekly_usage(conn)
    if seasons:
        usage = usage[usage["season"].isin(seasons)]
    if usage.empty:
        if own_conn:
            conn.close()
        return usage
    usage = usage.copy()
    usage["targets_share"] = usage.groupby(["season", "week", "team"])["targets"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else 0.0
    )
    usage["carries_share"] = usage.groupby(["season", "week", "team"])["carries"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else 0.0
    )
    rolling = (
        usage.sort_values(["player_id", "season", "week"])
        .groupby("player_id")[["targets_share", "carries_share"]]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    usage["targets_share_roll3"] = rolling["targets_share"]
    usage["carries_share_roll3"] = rolling["carries_share"]
    if own_conn:
        conn.close()
    return usage


def aggregate_weekly_to_season(weekly: pd.DataFrame) -> pd.DataFrame:
    """Sum weekly simulated stat lines to season totals."""
    stat_cols = [c for c in weekly.columns if c not in {"player_id", "season", "week", "team", "position"}]
    return weekly.groupby(["player_id", "season", "team", "position"], observed=True)[stat_cols].sum().reset_index()
