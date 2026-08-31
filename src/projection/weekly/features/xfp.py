"""ffopportunity (xFP) feature helpers — CC-BY-SA via nflverse."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


def attach_xfp_features(panel: pl.DataFrame, opp: pl.DataFrame) -> pl.DataFrame:
    """Join weekly xFP / OE columns and build lagged residuals.

    Actual week values are joined then lagged via rolling so week-W features
    never include week-W opportunity outcomes as predictors.
    """
    if opp.is_empty():
        return panel.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("xfp"),
                pl.lit(None).cast(pl.Float64).alias("fp_minus_xfp"),
                pl.lit(None).cast(pl.Float64).alias("rec_yards_oe"),
                pl.lit(None).cast(pl.Float64).alias("rush_yards_oe"),
            ]
        )

    df = opp
    id_col = "gsis_id" if "gsis_id" in df.columns else ("player_id" if "player_id" in df.columns else None)
    if id_col is None or "season" not in df.columns or "week" not in df.columns:
        logger.warning("ff_opportunity missing id/season/week; skipping")
        return panel

    xfp_col = "total_fantasy_points_exp" if "total_fantasy_points_exp" in df.columns else None
    sel = [id_col, "season", "week"]
    rename = {id_col: "gsis_id"}
    if xfp_col:
        sel.append(xfp_col)
        rename[xfp_col] = "xfp"
    if "rec_yards_gained_diff" in df.columns:
        sel.append("rec_yards_gained_diff")
        rename["rec_yards_gained_diff"] = "rec_yards_oe"
    elif "rec_yards_gained" in df.columns and "rec_yards_gained_exp" in df.columns:
        df = df.with_columns(
            (pl.col("rec_yards_gained") - pl.col("rec_yards_gained_exp")).alias("rec_yards_oe")
        )
        sel.append("rec_yards_oe")
    if "rush_yards_gained_diff" in df.columns:
        sel.append("rush_yards_gained_diff")
        rename["rush_yards_gained_diff"] = "rush_yards_oe"
    elif "rush_yards_gained" in df.columns and "rush_yards_gained_exp" in df.columns:
        df = df.with_columns(
            (pl.col("rush_yards_gained") - pl.col("rush_yards_gained_exp")).alias("rush_yards_oe")
        )
        sel.append("rush_yards_oe")

    feats = (
        df.select([c for c in sel if c in df.columns])
        .rename({k: v for k, v in rename.items() if k in df.columns or k in sel})
    )
    feats = feats.with_columns(
        [
            pl.col("gsis_id").cast(pl.Utf8),
            pl.col("season").cast(pl.Int64, strict=False),
            pl.col("week").cast(pl.Int64, strict=False),
        ]
    ).unique(subset=["gsis_id", "season", "week"], keep="first")
    # Ensure expected columns exist after rename
    for c in ("xfp", "rec_yards_oe", "rush_yards_oe"):
        if c not in feats.columns:
            feats = feats.with_columns(pl.lit(None).cast(pl.Float64).alias(c))

    panel_join = panel.with_columns(
        [
            pl.col("gsis_id").cast(pl.Utf8),
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
        ]
    )
    out = panel_join.join(feats, on=["gsis_id", "season", "week"], how="left")
    if "fantasy_points" in out.columns and "xfp" in out.columns:
        out = out.with_columns((pl.col("fantasy_points") - pl.col("xfp")).alias("fp_minus_xfp"))
    else:
        out = out.with_columns(pl.lit(None).cast(pl.Float64).alias("fp_minus_xfp"))

    logger.info(
        "xFP attached: xfp non-null=%d / %d",
        out.filter(pl.col("xfp").is_not_null()).height,
        out.height,
    )
    return out
