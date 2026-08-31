"""Team totals models: pass/rush attempts and TDs per team-week."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging

import polars as pl

from src.projection.weekly.models.base import MultiTargetModel, available_features, dataframe_to_matrix
from src.projection.weekly.models.registry import load_model, save_model

logger = logging.getLogger(__name__)

TEAM_TARGETS = [
    "team_pass_attempts",
    "team_rush_attempts",
    "team_pass_tds",
    "team_rush_tds",
]

TEAM_FEATURE_CANDIDATES = [
    "total_line",
    "spread_line",
    "implied_team_total",
    "implied_opp_total",
    "is_home",
    "rest_days",
    "temp",
    "wind",
    "team_pass_rate_l5",
    "team_pass_rate_prior_season",
]

# League-average anchors when building team-strength priors for unpriced games
_LEAGUE_AVG_TOTAL = 45.0
_LEAGUE_AVG_PASS_ATT = 34.0
_LEAGUE_AVG_RUSH_ATT = 27.0


def fill_missing_vegas_from_team_strength(
    team_df: pl.DataFrame,
    *,
    history: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Replace missing Vegas lines with prior-season / recent strength priors.

    Unpriced future games otherwise collapse to the imputer median, wiping
    cross-team differentiation. Prefer prior-season team totals when present.
    ``history`` should include prior seasons (e.g. full panel team-weeks).
    """
    if team_df.is_empty():
        return team_df
    out = team_df
    source = history if history is not None and not history.is_empty() else team_df

    if {"team", "season", "team_pass_attempts", "team_rush_attempts"}.issubset(source.columns):
        prior = (
            source.group_by(["season", "team"])
            .agg(
                [
                    pl.col("team_pass_attempts").mean().alias("_prior_pass"),
                    pl.col("team_rush_attempts").mean().alias("_prior_rush"),
                ]
            )
            .with_columns((pl.col("season") + 1).alias("season"))
        )
        # Drop existing helper cols then join
        drop_h = [c for c in ("_prior_pass", "_prior_rush") if c in out.columns]
        if drop_h:
            out = out.drop(drop_h)
        out = out.join(prior, on=["season", "team"], how="left")
    else:
        out = out.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("_prior_pass"),
                pl.lit(None).cast(pl.Float64).alias("_prior_rush"),
            ]
        )

    pass_p = pl.col("_prior_pass").fill_null(_LEAGUE_AVG_PASS_ATT)
    rush_p = pl.col("_prior_rush").fill_null(_LEAGUE_AVG_RUSH_ATT)
    strength = (pass_p + rush_p) / (_LEAGUE_AVG_PASS_ATT + _LEAGUE_AVG_RUSH_ATT)
    synth_total = (_LEAGUE_AVG_TOTAL * strength).clip(38.0, 55.0)
    if "is_home" in out.columns:
        synth_spread = pl.when(pl.col("is_home") == 1).then(-2.5).otherwise(2.5)
    else:
        synth_spread = pl.lit(0.0)

    exprs = []
    if "total_line" in out.columns:
        exprs.append(pl.col("total_line").fill_null(synth_total).alias("total_line"))
    if "spread_line" in out.columns:
        exprs.append(pl.col("spread_line").fill_null(synth_spread).alias("spread_line"))
    if exprs:
        out = out.with_columns(exprs)

    if "implied_team_total" in out.columns and "total_line" in out.columns and "spread_line" in out.columns:
        fill_exprs = [
            pl.col("implied_team_total")
            .fill_null(pl.col("total_line") / 2.0 - pl.col("spread_line") / 2.0)
            .alias("implied_team_total"),
        ]
        if "implied_opp_total" in out.columns:
            fill_exprs.append(
                pl.col("implied_opp_total")
                .fill_null(pl.col("total_line") / 2.0 + pl.col("spread_line") / 2.0)
                .alias("implied_opp_total")
            )
        out = out.with_columns(fill_exprs)

    drop_helpers = [c for c in ("_prior_pass", "_prior_rush") if c in out.columns]
    if drop_helpers:
        out = out.drop(drop_helpers)
    return out


def build_team_week_labels(panel: pl.DataFrame) -> pl.DataFrame:
    """Aggregate player-week panel into team-week totals + context features."""
    agg = panel.group_by(["season", "week", "team"]).agg(
        [
            pl.col("attempts").sum().alias("team_pass_attempts"),
            pl.col("carries").sum().alias("team_rush_attempts"),
            pl.col("passing_tds").sum().alias("team_pass_tds"),
            pl.col("rushing_tds").sum().alias("team_rush_tds"),
            pl.col("total_line").first(),
            pl.col("spread_line").first(),
            pl.col("implied_team_total").first(),
            pl.col("implied_opp_total").first(),
            pl.col("is_home").first(),
            pl.col("rest_days").first() if "rest_days" in panel.columns else pl.lit(None).alias("rest_days"),
            pl.col("temp").first() if "temp" in panel.columns else pl.lit(None).alias("temp"),
            pl.col("wind").first() if "wind" in panel.columns else pl.lit(None).alias("wind"),
            pl.col("team_pass_rate_l5").first()
            if "team_pass_rate_l5" in panel.columns
            else pl.lit(None).alias("team_pass_rate_l5"),
            pl.col("team_pass_rate_prior_season").first()
            if "team_pass_rate_prior_season" in panel.columns
            else pl.lit(None).alias("team_pass_rate_prior_season"),
        ]
    )
    return agg


def train_team_totals(
    panel: pl.DataFrame,
    *,
    train_seasons: list[int],
    model_type: str = "ridge",
    persist: bool = True,
) -> MultiTargetModel:
    team_df = build_team_week_labels(panel)
    train = team_df.filter(pl.col("season").is_in(train_seasons))
    features = available_features(train, TEAM_FEATURE_CANDIDATES)
    X = dataframe_to_matrix(train, features)
    y = train.select(TEAM_TARGETS)
    model = MultiTargetModel(targets=TEAM_TARGETS, feature_cols=features)
    model.fit(X, y, model_type=model_type)
    if persist:
        save_model(
            "team_totals",
            model,
            meta={"features": features, "targets": TEAM_TARGETS, "train_seasons": train_seasons},
        )
    logger.info("Trained team totals model on %d team-weeks", train.height)
    return model


def predict_team_totals(
    team_context: pl.DataFrame,
    model: MultiTargetModel | None = None,
    *,
    history: pl.DataFrame | None = None,
) -> pl.DataFrame:
    model = model or load_model("team_totals")
    ctx = fill_missing_vegas_from_team_strength(team_context, history=history)
    return model.predict_frame(ctx, prefix="pred_")
