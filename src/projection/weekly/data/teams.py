"""NFL team abbreviation helpers."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import polars as pl

# Rosters sometimes use AZ while schedules/depth charts use ARI
TEAM_ABBR_ALIASES: dict[str, str] = {
    "AZ": "ARI",
    "ARZ": "ARI",
    "WSH": "WAS",
    "JAC": "JAX",
    "STL": "LA",
    "SD": "LAC",
    "OAK": "LV",
}


def normalize_team_abbr(team: str | None) -> str | None:
    if team is None:
        return None
    t = str(team).strip().upper()
    if not t:
        return None
    return TEAM_ABBR_ALIASES.get(t, t)


def normalize_team_column(df: pl.DataFrame, col: str = "team") -> pl.DataFrame:
    """Map alias abbreviations onto schedule/depth canonical codes."""
    if col not in df.columns or df.is_empty():
        return df
    return df.with_columns(
        pl.col(col)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_uppercase()
        .replace(TEAM_ABBR_ALIASES)
        .alias(col)
    )
