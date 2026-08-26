"""Shared realized-history loader for the calibration scripts.

`weekly.season_type` is NULL for every 2025 row (those come from a pbp
fallback ingest), so a plain `season_type = 'REG'` filter silently drops the
whole most recent season. Those same rows also ship with a NULL `position`, so a position filter drops
them a second time. `src/projection/data_prep.load_weekly_usage` handles both by
deriving season_type from the week number and backfilling position from the
`players` master; this module applies the same two rules so the calibration
references cover the same seasons the model trains on.
"""

from __future__ import annotations

import os
import sqlite3

import pandas as pd

DB_PATH = os.environ.get("FANTASY_PROJECTIONS_DB_PATH") or os.path.join(
    os.environ.get("FANTASY_PROJECTIONS_DATA_DIR", "data"), "projections.db"
)


def connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def realized_weekly(
    seasons: list[int],
    stats: list[str],
    positions: tuple[str, ...] = ("QB", "RB", "WR", "TE"),
) -> pd.DataFrame:
    """Per-(season, player, position) regular-season stat totals."""
    conn = connect()
    cols = ", ".join(f"COALESCE({c},0) AS {c}" for c in stats)
    df = pd.read_sql(
        f"""SELECT season, week, season_type, player_id, position, {cols}
            FROM weekly
            WHERE season IN ({','.join(str(s) for s in seasons)})""",
        conn,
    )
    master = pd.read_sql(
        "SELECT gsis_id AS player_id, position AS master_position FROM players", conn
    )
    conn.close()

    # 2025 ships with both columns null; derive/backfill rather than filter away.
    missing = df["season_type"].isna()
    df.loc[missing, "season_type"] = df.loc[missing, "week"].apply(
        lambda wk: "REG" if pd.notna(wk) and wk <= 18 else "POST"
    )
    df = df.merge(master.drop_duplicates("player_id"), on="player_id", how="left")
    df["position"] = df["position"].fillna(df["master_position"])

    df = df[(df["season_type"] == "REG") & df["position"].isin(positions)]
    return df.groupby(["season", "player_id", "position"], as_index=False)[stats].sum()


def seasons_covered(df: pd.DataFrame) -> list[int]:
    return sorted(int(s) for s in df["season"].unique())
