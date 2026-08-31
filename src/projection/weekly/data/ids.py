"""ID helpers and crosswalk utilities."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import polars as pl


ID_COLUMNS = (
    "gsis_id",
    "pfr_id",
    "espn_id",
    "yahoo_id",
    "sleeper_id",
    "pff_id",
    "fantasypros_id",
    "cfb_player_id",
    "cfbref_id",
    "nfl_id",
)


def coerce_id_columns(df: pl.DataFrame, columns: tuple[str, ...] | None = None) -> pl.DataFrame:
    """Coerce ID columns to nullable Utf8 strings (preserve leading zeros)."""
    cols = columns or tuple(c for c in ID_COLUMNS if c in df.columns)
    exprs = []
    for col in cols:
        if col in df.columns:
            exprs.append(
                pl.col(col)
                .cast(pl.Utf8, strict=False)
                .str.strip_chars()
                .replace("", None)
                .alias(col)
            )
    return df.with_columns(exprs) if exprs else df


def coalesce_player_id(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure a single player_id column, preferring gsis_id."""
    if "player_id" in df.columns and "gsis_id" not in df.columns:
        return df.with_columns(pl.col("player_id").cast(pl.Utf8).alias("gsis_id"))
    if "gsis_id" in df.columns and "player_id" not in df.columns:
        return df.with_columns(pl.col("gsis_id").alias("player_id"))
    if "gsis_id" in df.columns and "player_id" in df.columns:
        return df.with_columns(
            pl.coalesce([pl.col("gsis_id"), pl.col("player_id")]).alias("player_id")
        )
    raise ValueError("DataFrame must contain gsis_id or player_id")


def normalize_position(df: pl.DataFrame, column: str = "position") -> pl.DataFrame:
    """Map fantasy-relevant positions; collapse FB into RB."""
    if column not in df.columns:
        return df
    return df.with_columns(
        pl.when(pl.col(column).is_in(["FB", "HB"]))
        .then(pl.lit("RB"))
        .when(pl.col(column).is_in(["QB", "RB", "WR", "TE"]))
        .then(pl.col(column))
        .otherwise(pl.col(column))
        .alias(column)
    )
