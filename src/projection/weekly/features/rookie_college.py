"""Prospect-level features derived from CFBD season tables.

The output is one row per college player and uses only seasons supplied by the
caller. Callers projecting draft class Y must therefore pass seasons < Y.
"""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import polars as pl

CFBD_SOURCE_METADATA = {
    "source": "CollegeFootballData.com REST API",
    "access": "free API key; cache annual batch pulls",
    "license_note": "Follow CFBD terms; retain endpoint and retrieval metadata",
}

PRODUCTION_COLS = (
    "rec_yards",
    "rec_tds",
    "receptions",
    "rush_yards",
    "rush_tds",
    "rush_attempts",
)


def _empty() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "college_player_id": pl.Utf8,
            "college_player": pl.Utf8,
            "college_season": pl.Int64,
            "college_dominator": pl.Float64,
            "college_breakout_season": pl.Int64,
            "college_breakout_age": pl.Float64,
        }
    )


def build_cfbd_prospect_features(
    player_seasons: pl.DataFrame,
    *,
    team_context: pl.DataFrame | None = None,
    breakout_threshold: float = 0.20,
) -> pl.DataFrame:
    """Collapse normalized CFBD player-seasons into prospect features.

    Expected keys are ``college_player_id``, ``college_season`` and ``team``.
    Missing production/context fields remain null. ``college_breakout_age`` is
    populated when the input supplies an ``age`` column; otherwise the
    breakout season remains a useful hook for a later birth-date crosswalk.
    """
    if player_seasons.is_empty():
        return _empty()
    required = {"college_player_id", "college_season", "team"}
    if not required.issubset(player_seasons.columns):
        return _empty()

    df = player_seasons.with_columns(
        pl.col("college_player_id").cast(pl.Utf8)
    ).filter(
        pl.col("college_player_id").is_not_null()
        & (pl.col("college_player_id").str.strip_chars() != "")
    )
    if df.is_empty():
        return _empty()
    exprs = []
    for col in PRODUCTION_COLS:
        if col not in df.columns:
            exprs.append(pl.lit(None).cast(pl.Float64).alias(col))
        else:
            exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
    df = df.with_columns(exprs)

    # Team denominators come from the complete category pulls, so shares are
    # comparable across conferences and do not require a separate vendor.
    totals = df.group_by(["college_season", "team"]).agg(
        [
            pl.col("rec_yards").fill_null(0.0).sum().alias("_team_rec_yards"),
            pl.col("rec_tds").fill_null(0.0).sum().alias("_team_rec_tds"),
            pl.col("rush_yards").fill_null(0.0).sum().alias("_team_rush_yards"),
            pl.col("rush_tds").fill_null(0.0).sum().alias("_team_rush_tds"),
        ]
    )
    df = df.join(totals, on=["college_season", "team"], how="left")

    def share(num: str, den: str) -> pl.Expr:
        return (
            pl.when(pl.col(den) > 0)
            .then(pl.col(num).fill_null(0.0) / pl.col(den))
            .otherwise(None)
        )

    df = df.with_columns(
        [
            share("rec_yards", "_team_rec_yards").alias("college_rec_yard_share"),
            share("rec_tds", "_team_rec_tds").alias("college_rec_td_share"),
            share("rush_yards", "_team_rush_yards").alias("college_rush_yard_share"),
            share("rush_tds", "_team_rush_tds").alias("college_rush_td_share"),
        ]
    ).with_columns(
        [
            (
                0.5
                * (
                    pl.col("college_rec_yard_share").fill_null(0.0)
                    + pl.col("college_rec_td_share").fill_null(0.0)
                )
            ).alias("college_rec_dominator"),
            (
                0.5
                * (
                    pl.max_horizontal(
                        "college_rec_yard_share", "college_rush_yard_share"
                    ).fill_null(0.0)
                    + pl.max_horizontal(
                        "college_rec_td_share", "college_rush_td_share"
                    ).fill_null(0.0)
                )
            ).alias("college_dominator"),
        ]
    )

    context = team_context if team_context is not None else pl.DataFrame()
    if not context.is_empty() and {"college_season", "team"}.issubset(context.columns):
        context = context.unique(subset=["college_season", "team"], keep="last")
        df = df.join(context, on=["college_season", "team"], how="left")

    games_col = next((c for c in ("games", "team_games", "college_team_games") if c in df.columns), None)
    per_game_cols: list[str] = []
    if games_col:
        per_game_exprs = []
        for col in PRODUCTION_COLS:
            name = f"college_{col}_per_game"
            per_game_cols.append(name)
            per_game_exprs.append(
                pl.when(pl.col(games_col).cast(pl.Float64, strict=False) > 0)
                .then(pl.col(col) / pl.col(games_col).cast(pl.Float64, strict=False))
                .otherwise(None)
                .alias(name)
            )
        df = df.with_columns(per_game_exprs)

    # Recency weight 1..N within a career. This is deterministic and uses no
    # information after the final supplied college season.
    df = df.sort(["college_player_id", "college_season"]).with_columns(
        (pl.int_range(pl.len()).over("college_player_id") + 1)
        .cast(pl.Float64)
        .alias("_career_weight")
    )

    breakout = (
        df.filter(pl.col("college_dominator") >= breakout_threshold)
        .sort(["college_player_id", "college_season"])
        .group_by("college_player_id")
        .agg(
            [
                pl.col("college_season").first().alias("college_breakout_season"),
                (
                    pl.col("age").cast(pl.Float64, strict=False).first()
                    if "age" in df.columns
                    else pl.lit(None).cast(pl.Float64)
                ).alias("college_breakout_age"),
            ]
        )
    )

    weighted_cols = [
        "college_dominator",
        "college_rec_dominator",
        "college_rec_yard_share",
        "college_rec_td_share",
        "college_rush_yard_share",
        "college_rush_td_share",
        *per_game_cols,
    ]
    for optional in (
        "college_team_srs",
        "college_team_core_offense",
        "college_team_core_overall",
        "college_team_sp_offense",
        "college_team_sp_rating",
    ):
        if optional in df.columns:
            weighted_cols.append(optional)

    aggs: list[pl.Expr] = [
        pl.col("college_season").n_unique().alias("college_seasons_played"),
    ]
    for col in PRODUCTION_COLS:
        aggs.append(pl.col(col).sum().alias(f"college_{col}_career"))
    for col in weighted_cols:
        denom = (
            pl.when(pl.col(col).is_not_null())
            .then(pl.col("_career_weight"))
            .otherwise(0.0)
            .sum()
        )
        aggs.append(
            pl.when(denom > 0)
            .then(
                (pl.col(col).fill_null(0.0) * pl.col("_career_weight")).sum()
                / denom
            )
            .otherwise(None)
            .alias(f"{col}_career_weighted")
        )

    career = df.group_by("college_player_id").agg(aggs)
    final = df.unique(subset=["college_player_id"], keep="last").drop(
        [c for c in df.columns if c.startswith("_team_")] + ["_career_weight"]
    )
    # Explicit final-year aliases make downstream feature selection stable
    # while retaining legacy rec_yards/rush_yards columns.
    final_aliases = []
    for col in [*PRODUCTION_COLS, "college_dominator", "college_rec_dominator", *per_game_cols]:
        if col in final.columns:
            final_aliases.append(pl.col(col).alias(f"{col}_final"))
    if final_aliases:
        final = final.with_columns(final_aliases)
    return (
        final.join(career, on="college_player_id", how="left")
        .join(breakout, on="college_player_id", how="left")
        .sort("college_player_id")
    )
