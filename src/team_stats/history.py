"""Load recent REG-season box-score history for player cards."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.projection.fantasy_points import SCORING

HISTORY_STAT_COLS = [
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
]

DEFAULT_N_SEASONS = 3


def _to_num(val) -> float | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except TypeError:
        pass
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _half_ppr_points(row: dict[str, float | None]) -> float:
    total = 0.0
    for stat, weight in SCORING.items():
        total += (row.get(stat) or 0.0) * weight
    return total


def load_player_history(
    target_season: int,
    *,
    n_seasons: int = DEFAULT_N_SEASONS,
) -> dict[str, list[dict[str, Any]]]:
    """Return ``player_id -> [season rows newest-first]`` for prior REG seasons.

    Stats are season totals plus half-PPR fantasy points. Uses nflreadpy's
    season-level player stats (REG). Missing history degrades to ``{}``.
    """
    if n_seasons <= 0:
        return {}
    seasons = list(range(target_season - n_seasons, target_season))
    try:
        import nflreadpy as nfl
    except ImportError:
        return {}

    try:
        frame = nfl.load_player_stats(seasons=seasons, summary_level="reg")
    except Exception:
        return {}

    if hasattr(frame, "to_pandas"):
        df = frame.to_pandas()
    else:
        df = pd.DataFrame(frame)

    if df.empty or "player_id" not in df.columns:
        return {}

    rename = {"passing_interceptions": "interceptions"}
    df = df.rename(columns=rename)
    if "interceptions" not in df.columns and "passing_interceptions" in df.columns:
        df["interceptions"] = df["passing_interceptions"]

    keep = ["player_id", "season", "games", "position"] + [
        c for c in HISTORY_STAT_COLS if c in df.columns
    ]
    missing = [c for c in HISTORY_STAT_COLS if c not in df.columns]
    for col in missing:
        df[col] = 0.0
    keep = ["player_id", "season", "games", "position"] + HISTORY_STAT_COLS
    df = df[keep].copy()
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()

    out: dict[str, list[dict[str, Any]]] = {}
    for row in df.itertuples(index=False):
        games = _to_num(row.games) or 0.0
        if games <= 0:
            continue
        stats = {c: _to_num(getattr(row, c, None)) or 0.0 for c in HISTORY_STAT_COLS}
        fpts = _half_ppr_points(stats)
        record: dict[str, Any] = {
            "season": int(row.season),
            "games": round(games, 1),
            "fantasy_pts": round(fpts / games, 2) if games else None,
            "fantasy_pts_season": round(fpts, 1),
        }
        for c in HISTORY_STAT_COLS:
            val = stats[c]
            if c.endswith("tds") or c == "interceptions":
                record[c] = round(val, 1)
            else:
                record[c] = int(round(val))
        pid = str(row.player_id)
        out.setdefault(pid, []).append(record)

    for pid, rows in out.items():
        rows.sort(key=lambda r: r["season"], reverse=True)
    return out
