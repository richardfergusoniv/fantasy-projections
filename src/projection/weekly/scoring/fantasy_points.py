"""Fantasy points conversion from box-score stats."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import polars as pl

from src.projection.weekly.config.scoring import ScoringConfig


STAT_COLUMNS = (
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
)


def fantasy_points_expr(
    config: ScoringConfig,
    *,
    alias: str = "fantasy_points",
) -> pl.Expr:
    """Polars expression computing fantasy points from box-score columns."""
    return (
        pl.col("passing_yards").fill_null(0) * config.pass_yard_points
        + pl.col("passing_tds").fill_null(0) * config.pass_td_points
        + pl.col("interceptions").fill_null(0) * config.interception_points
        + pl.col("rushing_yards").fill_null(0) * config.rush_rec_yard_points
        + pl.col("rushing_tds").fill_null(0) * config.rush_rec_td_points
        + pl.col("receptions").fill_null(0) * config.reception_points
        + pl.col("receiving_yards").fill_null(0) * config.rush_rec_yard_points
        + pl.col("receiving_tds").fill_null(0) * config.rush_rec_td_points
        + pl.col("fumbles_lost").fill_null(0) * config.fumble_lost_points
    ).alias(alias)


def compute_fantasy_points(
    df: pl.DataFrame,
    config: ScoringConfig | None = None,
    *,
    alias: str = "fantasy_points",
) -> pl.DataFrame:
    """Add fantasy points column to a DataFrame of box-score stats."""
    config = config or ScoringConfig()
    missing = [c for c in STAT_COLUMNS if c not in df.columns]
    out = df
    for col in missing:
        out = out.with_columns(pl.lit(0.0).alias(col))
    return out.with_columns(fantasy_points_expr(config, alias=alias))


def fantasy_points_from_dict(
    stats: dict[str, float],
    config: ScoringConfig | None = None,
) -> float:
    """Compute fantasy points from a single player's projected/actual stats."""
    config = config or ScoringConfig()
    return float(
        stats.get("passing_yards", 0.0) * config.pass_yard_points
        + stats.get("passing_tds", 0.0) * config.pass_td_points
        + stats.get("interceptions", 0.0) * config.interception_points
        + stats.get("rushing_yards", 0.0) * config.rush_rec_yard_points
        + stats.get("rushing_tds", 0.0) * config.rush_rec_td_points
        + stats.get("receptions", 0.0) * config.reception_points
        + stats.get("receiving_yards", 0.0) * config.rush_rec_yard_points
        + stats.get("receiving_tds", 0.0) * config.rush_rec_td_points
        + stats.get("fumbles_lost", 0.0) * config.fumble_lost_points
    )
