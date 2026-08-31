"""Efficiency models with shrinkage toward positional means."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging

import polars as pl

from src.projection.weekly.models.base import MultiTargetModel, available_features, dataframe_to_matrix
from src.projection.weekly.models.registry import load_model, save_model

logger = logging.getLogger(__name__)

EFFICIENCY_TARGETS_BY_POS: dict[str, list[str]] = {
    # rush_td_rate is required for dual-threat QBs (Allen/Hurts/Lamar); without
    # it accounting falls back to a flat 0.03 rate and under-projects rush TDs.
    "QB": ["ypa", "pass_td_rate", "int_rate", "ypc", "rush_td_rate"],
    "RB": ["ypc", "ypr", "rush_td_rate", "rec_td_rate", "catch_rate"],
    "WR": ["catch_rate", "ypr", "rec_td_rate"],
    "TE": ["catch_rate", "ypr", "rec_td_rate"],
}

EFFICIENCY_FEATURE_CANDIDATES = [
    "games_played_prior",
    "prior_season_games_played",
    "age",
    "is_rookie",
    "implied_team_total",
    "total_line",
    "is_home",
    "wind",
    "temp",
    "team_pass_rate_l5",
    "ypa_l5",
    "ypa_prior_season",
    "ypa_l5_shrunk",
    "ypc_l5",
    "ypc_prior_season",
    "ypc_l5_shrunk",
    "ypr_l5",
    "ypr_prior_season",
    "ypr_l5_shrunk",
    "catch_rate_l5",
    "catch_rate_prior_season",
    "catch_rate_l5_shrunk",
    "pass_td_rate_l5",
    "pass_td_rate_prior_season",
    "int_rate_l5",
    "int_rate_prior_season",
    "rush_td_rate_l5",
    "rush_td_rate_prior_season",
    "rec_td_rate_l5",
    "rec_td_rate_prior_season",
    "redzone_target_share_l5",
    "redzone_target_share_prior_season",
    "target_share_l5",
    "carry_share_l5",
    "air_yards_share_l5",
    # EPA / WOPR / RACR
    "wopr_l5",
    "wopr_prior_season",
    "racr_l5",
    "passing_epa_per_play_l5",
    "passing_epa_per_play_l5_shrunk",
    "receiving_epa_l5",
    "rushing_epa_l5",
    # Opponent defense
    "opp_ypa_allowed_l5",
    "opp_ypc_allowed_l5",
    "opp_ypr_allowed_l5",
    "opp_pass_epa_allowed_l5",
    "opp_rush_epa_allowed_l5",
    "opp_pass_rate_allowed_l5",
    # xFP residuals
    "xfp_l5",
    "fp_minus_xfp_l5",
    "rec_yards_oe_l5",
    "rush_yards_oe_l5",
    # Tracking-derived talent/context, all lagged by the panel builder
    "ngs_cpoe_l5",
    "ngs_cpoe_prior_season",
    "ngs_expected_completion_pct_l5",
    "ngs_time_to_throw_l5",
    "ngs_air_yards_to_sticks_l5",
    "ngs_avg_cushion_l5",
    "ngs_avg_separation_l5",
    "ngs_yac_above_expectation_l5",
    "ngs_yac_above_expectation_prior_season",
    "ngs_rush_efficiency_l5",
    "ngs_stacked_box_rate_l5",
    "ngs_time_to_los_l5",
    "ngs_ryoe_per_attempt_l5",
    "ngs_ryoe_per_attempt_prior_season",
    "ngs_rush_pct_over_expected_l5",
    # FTN charting is pre-lagged at team-week grain
    "team_motion_rate_l5",
    "team_play_action_rate_l5",
    "team_screen_rate_l5",
    "team_rpo_rate_l5",
    "team_catchable_rate_l5",
    "team_drop_rate_l5",
    "team_int_worthy_rate_l5",
    "team_qb_fault_sack_rate_l5",
]


def train_efficiency_models(
    panel: pl.DataFrame,
    *,
    train_seasons: list[int],
    model_type: str = "ridge",
    persist: bool = True,
) -> dict[str, MultiTargetModel]:
    """Ridge preferred: efficiency should be heavily shrunk / linear."""
    models: dict[str, MultiTargetModel] = {}
    train = panel.filter(pl.col("season").is_in(train_seasons))

    for pos, targets in EFFICIENCY_TARGETS_BY_POS.items():
        pos_df = train.filter(pl.col("position") == pos)
        if pos == "QB":
            pos_df = pos_df.filter(pl.col("attempts") >= 10)
        elif pos == "RB":
            pos_df = pos_df.filter(pl.col("carries") >= 5)
        else:
            pos_df = pos_df.filter(pl.col("targets") >= 3)

        present_targets = [t for t in targets if t in pos_df.columns]
        features = available_features(pos_df, EFFICIENCY_FEATURE_CANDIDATES)
        if pos_df.height < 50 or not present_targets:
            logger.warning("Skipping efficiency model for %s", pos)
            continue

        # Drop absurd rate labels (e.g. ypc when carries==0 used to explode via 1e-6)
        clean = pos_df
        for t in present_targets:
            if t == "ypc":
                clean = clean.filter(pl.col(t).is_null() | ((pl.col(t) >= 0.0) & (pl.col(t) <= 15.0)))
            elif t == "ypa":
                clean = clean.filter(pl.col(t).is_null() | ((pl.col(t) >= 2.0) & (pl.col(t) <= 15.0)))
            elif t == "ypr":
                clean = clean.filter(pl.col(t).is_null() | ((pl.col(t) >= 0.0) & (pl.col(t) <= 40.0)))
            elif t == "catch_rate":
                clean = clean.filter(pl.col(t).is_null() | ((pl.col(t) >= 0.0) & (pl.col(t) <= 1.0)))
            elif t == "rush_td_rate":
                # Zero-carry rows produce unstable rates; null the label so other
                # QB targets still train on the row while rush_td_rate skips it.
                if "carries" in clean.columns:
                    clean = clean.with_columns(
                        pl.when(pl.col("carries").fill_null(0.0) < 1.0)
                        .then(pl.lit(None))
                        .otherwise(pl.col(t))
                        .alias(t)
                    )
                clean = clean.filter(
                    pl.col(t).is_null() | ((pl.col(t) >= 0.0) & (pl.col(t) <= 0.5))
                )
            elif t.endswith("rate") or t == "int_rate":
                clean = clean.filter(pl.col(t).is_null() | ((pl.col(t) >= 0.0) & (pl.col(t) <= 0.5)))
        if clean.height < 50:
            clean = pos_df

        X = dataframe_to_matrix(clean, features)
        y = clean.select(present_targets)
        model = MultiTargetModel(targets=present_targets, feature_cols=features)
        model.fit(X, y, model_type=model_type)
        # Stronger shrink for efficiency: overwrite predict shrink in meta
        for t in present_targets:
            model.positional_means[t] = float(clean[t].drop_nulls().mean() or 0.0)

        if persist:
            save_model(
                f"efficiency_{pos}",
                model,
                meta={
                    "position": pos,
                    "features": features,
                    "targets": present_targets,
                    "train_seasons": train_seasons,
                },
            )
        models[pos] = model
        logger.info("Trained efficiency_%s on %d rows", pos, pos_df.height)

    return models


def predict_efficiency(
    panel_slice: pl.DataFrame,
    models: dict[str, MultiTargetModel] | None = None,
) -> pl.DataFrame:
    frames = []
    for pos in panel_slice["position"].unique().to_list():
        sub = panel_slice.filter(pl.col("position") == pos)
        try:
            model = (models or {}).get(pos) or load_model(f"efficiency_{pos}")
        except FileNotFoundError:
            frames.append(sub)
            continue
        # Extra shrink for TD / INT rates; blend YPA / rush TD rate toward lagged prior
        pred_df = model.predict_frame(sub, prefix="pred_")
        for td_col in ("pred_pass_td_rate", "pred_rush_td_rate", "pred_rec_td_rate", "pred_int_rate"):
            if td_col in pred_df.columns:
                default = 0.025 if td_col == "pred_int_rate" else 0.05
                mean = model.positional_means.get(td_col.replace("pred_", ""), default)
                if td_col == "pred_int_rate" and (mean is None or mean < 0.01):
                    mean = 0.025
                # Rush TD rates are sticky for workhorse RBs / dual-threat QBs;
                # shrink less than pass/rec TD rates so Allen/Taylor retain signal.
                if td_col == "pred_rush_td_rate":
                    pred_df = pred_df.with_columns(
                        (0.75 * pl.col(td_col) + 0.25 * mean).alias(td_col)
                    )
                else:
                    pred_df = pred_df.with_columns(
                        (0.6 * pl.col(td_col) + 0.4 * mean).alias(td_col)
                    )
        if "pred_ypa" in pred_df.columns:
            ypa_mean = model.positional_means.get("ypa", 7.0)
            lag = (
                pl.col("ypa_l5_shrunk")
                if "ypa_l5_shrunk" in pred_df.columns
                else (pl.col("ypa_l5") if "ypa_l5" in pred_df.columns else pl.lit(ypa_mean))
            )
            # Slightly less lag weight so pocket passers (high recent YPA) are not
            # over-boosted relative to dual-threat QBs whose value is rushing.
            pred_df = pred_df.with_columns(
                (
                    0.50 * pl.col("pred_ypa")
                    + 0.30 * lag.fill_null(ypa_mean).clip(4.0, 10.5)
                    + 0.20 * ypa_mean
                )
                .clip(5.0, 10.0)
                .alias("pred_ypa")
            )
        if "pred_rush_td_rate" in pred_df.columns:
            rtd_mean = model.positional_means.get("rush_td_rate", 0.04)
            lag_rtd = (
                pl.col("rush_td_rate_l5")
                if "rush_td_rate_l5" in pred_df.columns
                else pl.lit(rtd_mean)
            )
            prior_rtd = (
                pl.col("rush_td_rate_prior_season")
                if "rush_td_rate_prior_season" in pred_df.columns
                else pl.lit(rtd_mean)
            )
            # Clip ceiling higher for QBs (goal-line scrambles) than for the
            # shared accounting default; RBs still sit under 0.12 historically.
            rtd_hi = 0.20 if pos == "QB" else 0.12
            pred_df = pred_df.with_columns(
                (
                    0.45 * pl.col("pred_rush_td_rate")
                    + 0.35 * lag_rtd.fill_null(rtd_mean).clip(0.0, rtd_hi)
                    + 0.20 * prior_rtd.fill_null(rtd_mean).clip(0.0, rtd_hi)
                )
                .clip(0.0, rtd_hi)
                .alias("pred_rush_td_rate")
            )
        if "pred_int_rate" in pred_df.columns:
            pred_df = pred_df.with_columns(
                pl.col("pred_int_rate").fill_null(0.025).clip(0.01, 0.06).alias("pred_int_rate")
            )
        frames.append(pred_df)
    if not frames:
        return panel_slice
    return pl.concat(frames, how="diagonal_relaxed")
