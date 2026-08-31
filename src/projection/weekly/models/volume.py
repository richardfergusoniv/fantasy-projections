"""Volume allocation models: predict usage shares by position."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.projection.weekly.models.base import MultiTargetModel, available_features, dataframe_to_matrix
from src.projection.weekly.models.registry import load_model, save_model

logger = logging.getLogger(__name__)

DEFAULT_PARTICIPATION_THRESHOLD = 1e-6
DEFAULT_RECENCY_HALF_LIFE_SEASONS: float | None = None

VOLUME_TARGETS_BY_POS: dict[str, list[str]] = {
    "QB": ["dropback_share", "carry_share"],
    "RB": ["carry_share", "target_share", "snap_share", "redzone_target_share"],
    "WR": ["target_share", "air_yards_share", "snap_share", "redzone_target_share"],
    "TE": ["target_share", "air_yards_share", "snap_share", "redzone_target_share"],
}

VOLUME_FEATURE_CANDIDATES = [
    "games_played_prior",
    "prior_season_games_played",
    "age",
    "is_rookie",
    "draft_pick",
    "draft_round",
    "total_line",
    "spread_line",
    "implied_team_total",
    "is_home",
    "rest_days",
    "team_pass_rate_l5",
    "team_pass_rate_prior_season",
    # injury / play probability
    "is_out",
    "is_doubtful",
    "is_questionable",
    "play_prob",
    # lagged shares
    "target_share_l3",
    "target_share_l5",
    "target_share_season_td",
    "target_share_l3_shrunk",
    "target_share_l5_shrunk",
    "target_share_prior_season",
    "carry_share_l3",
    "carry_share_l5",
    "carry_share_season_td",
    "carry_share_l3_shrunk",
    "carry_share_l5_shrunk",
    "carry_share_prior_season",
    "snap_share_l3",
    "snap_share_l5",
    "snap_share_season_td",
    "snap_share_prior_season",
    "air_yards_share_l3",
    "air_yards_share_l5",
    "air_yards_share_prior_season",
    "dropback_share_l3",
    "dropback_share_l5",
    "dropback_share_prior_season",
    "redzone_target_share_l3",
    "redzone_target_share_l5",
    "redzone_target_share_prior_season",
    "fantasy_points_l3",
    "fantasy_points_l5",
    "fantasy_points_prior_season",
    "targets_l3",
    "carries_l3",
    "attempts_l3",
    # EPA / WOPR / RACR
    "wopr_l5",
    "wopr_prior_season",
    "racr_l5",
    "racr_prior_season",
    "passing_epa_per_play_l5",
    "passing_epa_per_play_l5_shrunk",
    "passing_epa_per_play_prior_season",
    "receiving_epa_l5",
    "receiving_epa_prior_season",
    "rushing_epa_l5",
    "rushing_epa_prior_season",
    # Opponent defense
    "opp_ypa_allowed_l5",
    "opp_ypc_allowed_l5",
    "opp_ypr_allowed_l5",
    "opp_pass_epa_allowed_l5",
    "opp_rush_epa_allowed_l5",
    "opp_pass_rate_allowed_l5",
    # Depth / role
    "depth_rank",
    "is_listed_starter",
    "same_pos_depth_count",
    # OverTheCap / nflverse contracts
    "contract_apy_cap_pct",
    "contract_inflated_apy",
    "contract_guaranteed_pct",
    "contract_years",
    "contract_years_remaining",
    # xFP residuals
    "xfp_l5",
    "fp_minus_xfp_l5",
    "rec_yards_oe_l5",
    "rush_yards_oe_l5",
    # Play-level participation (lagged; current week is never a feature)
    "offense_play_participation_l3",
    "offense_play_participation_l5",
    "offense_play_participation_prior_season",
    "pass_play_participation_l3",
    "pass_play_participation_l5",
    "pass_play_participation_prior_season",
    # Prior roster status/history
    "roster_active_prev_week",
    "roster_reserve_prev_week",
    "roster_active_rate_prior",
    "roster_team_changed_prev_week",
    # Lagged team scheme from FTN charting
    "team_motion_rate_l5",
    "team_play_action_rate_l5",
    "team_screen_rate_l5",
    "team_rpo_rate_l5",
    "team_no_huddle_rate_l5",
]


def recency_sample_weights(
    seasons: pl.Series | np.ndarray,
    *,
    half_life_seasons: float | None = DEFAULT_RECENCY_HALF_LIFE_SEASONS,
) -> np.ndarray | None:
    """Exponential season-recency weights, or ``None`` when disabled."""
    if half_life_seasons is None:
        return None
    half_life = float(half_life_seasons)
    if half_life <= 0:
        raise ValueError("recency_half_life_seasons must be > 0")
    values = np.asarray(seasons, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.ones(len(values), dtype=float)
    latest = float(np.nanmax(values[finite]))
    weights = np.ones(len(values), dtype=float)
    weights[finite] = np.power(0.5, (latest - values[finite]) / half_life)
    return weights


def _make_participation_pipeline(model_type: str) -> Pipeline:
    if model_type == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
            ]
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=42,
                    max_depth=4,
                    learning_rate=0.06,
                    max_iter=200,
                ),
            ),
        ]
    )


@dataclass
class TwoStageVolumeModel:
    """Participation probability × share conditional on positive usage."""

    targets: list[str]
    feature_cols: list[str]
    participation_threshold: float = DEFAULT_PARTICIPATION_THRESHOLD
    participation_models: dict[str, Pipeline] = field(default_factory=dict)
    participation_rates: dict[str, float] = field(default_factory=dict)
    conditional_model: MultiTargetModel | None = None
    conditional_means: dict[str, float] = field(default_factory=dict)

    def fit(
        self,
        X: np.ndarray,
        y: pl.DataFrame,
        *,
        conditional_model_type: str = "hgb",
        participation_model_type: str = "hgb",
        sample_weight: np.ndarray | None = None,
        min_classifier_rows: int = 20,
    ) -> TwoStageVolumeModel:
        conditional_targets: list[str] = []
        conditional_columns: list[pl.Series] = []
        weights = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None

        for target in self.targets:
            values = y[target].to_numpy().astype(float)
            valid = np.isfinite(values)
            positive = valid & (values > float(self.participation_threshold))
            labels = positive[valid].astype(int)
            target_weights = weights[valid] if weights is not None else None
            if labels.size:
                self.participation_rates[target] = float(
                    np.average(labels, weights=target_weights)
                    if target_weights is not None
                    else labels.mean()
                )
            else:
                self.participation_rates[target] = 0.0

            if (
                labels.size >= int(min_classifier_rows)
                and np.unique(labels).size == 2
            ):
                classifier = _make_participation_pipeline(participation_model_type)
                fit_params = (
                    {"model__sample_weight": target_weights}
                    if target_weights is not None
                    else {}
                )
                classifier.fit(X[valid], labels, **fit_params)
                self.participation_models[target] = classifier

            if positive.any():
                positive_values = values[positive]
                positive_weights = weights[positive] if weights is not None else None
                self.conditional_means[target] = float(
                    np.average(positive_values, weights=positive_weights)
                    if positive_weights is not None
                    else positive_values.mean()
                )
                conditional_targets.append(target)
                conditional_columns.append(
                    pl.Series(
                        target,
                        np.where(positive, values, np.nan),
                        dtype=pl.Float64,
                    )
                )
            else:
                self.conditional_means[target] = 0.0

        if conditional_targets:
            conditional_y = pl.DataFrame(conditional_columns)
            conditional = MultiTargetModel(
                targets=conditional_targets,
                feature_cols=self.feature_cols,
            )
            conditional.fit(
                X,
                conditional_y,
                model_type=conditional_model_type,
                sample_weight=sample_weight,
            )
            self.conditional_model = conditional
        return self

    def predict(self, X: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        conditional = self.conditional_model.predict(X) if self.conditional_model else {}
        shares: dict[str, np.ndarray] = {}
        probabilities: dict[str, np.ndarray] = {}
        for target in self.targets:
            classifier = self.participation_models.get(target)
            if classifier is not None:
                probability = np.asarray(classifier.predict_proba(X)[:, 1], dtype=float)
            else:
                probability = np.full(X.shape[0], self.participation_rates.get(target, 0.0))
            probability = np.clip(np.nan_to_num(probability, nan=0.0), 0.0, 1.0)
            conditional_share = conditional.get(
                target,
                np.full(X.shape[0], self.conditional_means.get(target, 0.0)),
            )
            probabilities[target] = probability
            shares[target] = np.clip(probability * conditional_share, 0.0, 1.0)
        return shares, probabilities

    def predict_frame(self, df: pl.DataFrame, prefix: str = "pred_") -> pl.DataFrame:
        X = dataframe_to_matrix(df, self.feature_cols)
        shares, probabilities = self.predict(X)
        columns = [pl.Series(f"{prefix}{target}", values) for target, values in shares.items()]
        columns.extend(
            pl.Series(f"{prefix}{target}_participation_prob", values)
            for target, values in probabilities.items()
        )
        return df.with_columns(columns)


_COMPOSITION_POSITIONS: dict[str, tuple[str, ...]] = {
    "dropback_share": ("QB",),
    # QB carries stay absolute (lag-anchored team-rush fractions). Composing
    # them with RBs into a unit simplex lets Henry/CMC crush Lamar/Hurts.
    "carry_share": ("RB",),
    "target_share": ("RB", "WR", "TE"),
    "air_yards_share": ("RB", "WR", "TE"),
    "redzone_target_share": ("RB", "WR", "TE"),
}


def compose_team_volume_predictions(
    df: pl.DataFrame,
    *,
    group_keys: list[str] | None = None,
) -> pl.DataFrame:
    """Convert non-negative expected usage weights into team compositions.

    Snap share is intentionally excluded because several players can be on the
    field on one play. Accounting may subsequently reserve unassigned volume,
    apply depth priors, and cap individual shares without changing these ratios.

    Positions outside a target's composition set keep their predicted weights
    unchanged (QB carry shares are lag-anchored team-rush fractions and must
    not be zeroed when only RBs are composed).
    """
    keys = group_keys or [c for c in ("season", "week", "team") if c in df.columns]
    if not keys or "position" not in df.columns:
        return df
    out = df
    for target, positions in _COMPOSITION_POSITIONS.items():
        col = f"pred_{target}"
        if col not in out.columns:
            continue
        eligible = pl.col("position").is_in(list(positions))
        weight = pl.when(eligible).then(pl.col(col).fill_null(0.0).clip(0.0, 1.0)).otherwise(0.0)
        total = weight.sum().over(keys)
        composed = pl.when(eligible & (total > 1e-12)).then(weight / total).otherwise(0.0)
        out = out.with_columns(
            pl.when(eligible).then(composed).otherwise(pl.col(col)).alias(col)
        )
    return out


def train_volume_models(
    panel: pl.DataFrame,
    *,
    train_seasons: list[int],
    model_type: str = "hgb",
    participation_model_type: str = "hgb",
    participation_threshold: float = DEFAULT_PARTICIPATION_THRESHOLD,
    min_participation_rows: int = 20,
    recency_half_life_seasons: float | None = DEFAULT_RECENCY_HALF_LIFE_SEASONS,
    two_stage: bool = True,
    persist: bool = True,
) -> dict[str, MultiTargetModel | TwoStageVolumeModel]:
    """Train volume models on the full eligible roster-week population.

    The default two-stage model learns whether a player participates from all
    rows (including realized zero usage), then estimates share conditional on
    positive usage. ``two_stage=False`` remains available for controlled legacy
    comparisons, while previously persisted ``MultiTargetModel`` artifacts are
    still accepted by :func:`predict_volume`.
    """
    models: dict[str, MultiTargetModel | TwoStageVolumeModel] = {}
    train = panel.filter(pl.col("season").is_in(train_seasons))

    for pos, targets in VOLUME_TARGETS_BY_POS.items():
        pos_df = train.filter(pl.col("position") == pos)
        present_targets = [t for t in targets if t in pos_df.columns]
        features = available_features(pos_df, VOLUME_FEATURE_CANDIDATES)
        if pos_df.height < 50 or not present_targets or not features:
            logger.warning("Skipping volume model for %s (n=%d)", pos, pos_df.height)
            continue

        X = dataframe_to_matrix(pos_df, features)
        y = pos_df.select(present_targets)
        sample_weight = recency_sample_weights(
            pos_df["season"], half_life_seasons=recency_half_life_seasons
        )
        if two_stage:
            model = TwoStageVolumeModel(
                targets=present_targets,
                feature_cols=features,
                participation_threshold=float(participation_threshold),
            )
            model.fit(
                X,
                y,
                conditional_model_type=model_type,
                participation_model_type=participation_model_type,
                sample_weight=sample_weight,
                min_classifier_rows=min_participation_rows,
            )
        else:
            model = MultiTargetModel(targets=present_targets, feature_cols=features)
            model.fit(X, y, model_type=model_type, sample_weight=sample_weight)
        if persist:
            save_model(
                f"volume_{pos}",
                model,
                meta={
                    "position": pos,
                    "features": features,
                    "targets": present_targets,
                    "train_seasons": train_seasons,
                    "architecture": "two_stage_participation_conditional"
                    if two_stage
                    else "legacy_direct",
                    "participation_threshold": float(participation_threshold),
                    "min_participation_rows": int(min_participation_rows),
                    "recency_half_life_seasons": recency_half_life_seasons,
                    "schema_version": 2,
                },
            )
        models[pos] = model
        logger.info("Trained volume_%s on %d rows (%s)", pos, pos_df.height, present_targets)

    return models


def predict_volume(
    panel_slice: pl.DataFrame,
    models: dict[str, MultiTargetModel | TwoStageVolumeModel] | None = None,
    *,
    compose: bool = True,
) -> pl.DataFrame:
    """Add pred_* shares and coherently compose expected team usage.

    Legacy direct-regression artifacts implement the same ``predict_frame``
    interface and continue to load without migration.
    """
    frames = []
    for pos in panel_slice["position"].unique().to_list():
        sub = panel_slice.filter(pl.col("position") == pos)
        try:
            model = (models or {}).get(pos) or load_model(f"volume_{pos}")
        except FileNotFoundError:
            frames.append(sub)
            continue
        frames.append(model.predict_frame(sub, prefix="pred_"))
    if not frames:
        return panel_slice
    out = pl.concat(frames, how="diagonal_relaxed")
    out = _blend_volume_with_lagged_priors(out)
    return compose_team_volume_predictions(out) if compose else out


def _blend_volume_with_lagged_priors(df: pl.DataFrame) -> pl.DataFrame:
    """Pull sticky usage toward recent/prior shares so elites are not flattened.

    Helps workhorse RBs (Taylor) and TE1s (Bowers) whose two-stage predictions
    compress toward committee means, and keeps vacated WR volume from pinning
    entirely on WR1 when WR2 still has a real prior (Flowers/Bateman).

    Dual-threat QB carry shares are prior-heavy: Ridge otherwise collapses
    Lamar/Hurts toward pocket-passer means while noisier rush weeks can leave
    pocket QBs over-weighted. An asymmetric lag floor also stops the model
    term from crushing established usage more than ~25% below the better lag.
    """
    out = df
    # pred_col, l5_col, prior_col, w_model, w_l5, w_prior, position_filter|None
    blends: list[tuple[str, str, str, float, float, float, str | None]] = [
        # Dual-threat QB rushes are sticky; Ridge collapses them toward pocket means.
        ("pred_carry_share", "carry_share_l5", "carry_share_prior_season", 0.15, 0.35, 0.50, "QB"),
        ("pred_carry_share", "carry_share_l5", "carry_share_prior_season", 0.45, 0.30, 0.25, None),
        ("pred_target_share", "target_share_l5", "target_share_prior_season", 0.35, 0.30, 0.35, None),
    ]
    for pred_col, l5_col, prior_col, w_m, w_l5, w_p, pos_filter in blends:
        if pred_col not in out.columns:
            continue
        l5 = pl.col(l5_col) if l5_col in out.columns else pl.lit(None)
        prior = pl.col(prior_col) if prior_col in out.columns else pl.lit(None)
        has_anchor = l5.is_not_null() | prior.is_not_null()
        model = pl.col(pred_col).fill_null(0.0)
        l5_f = l5.fill_null(model)
        prior_f = prior.fill_null(model)
        blended = (w_m * model + w_l5 * l5_f + w_p * prior_f).clip(0.0, 1.0)
        # Floor sticky usage near the stronger lag. QBs use a higher floor so
        # Lamar/Hurts cannot lose a third of their historical rush share.
        floor_frac = 0.90 if pos_filter == "QB" else 0.75
        lag_floor = pl.max_horizontal(l5_f, prior_f) * floor_frac
        blended = pl.max_horizontal(blended, lag_floor).clip(0.0, 1.0)
        apply_mask = has_anchor
        if pos_filter is not None and "position" in out.columns:
            apply_mask = has_anchor & (pl.col("position") == pos_filter)
        elif pos_filter is None and pred_col == "pred_carry_share" and "position" in out.columns:
            # QB row already handled by the prior-heavy blend above.
            apply_mask = has_anchor & (pl.col("position") != "QB")
        out = out.with_columns(
            pl.when(apply_mask).then(blended).otherwise(pl.col(pred_col)).alias(pred_col)
        )
    return _seed_depth_share_anchors(out)


def _seed_depth_share_anchors(df: pl.DataFrame) -> pl.DataFrame:
    """Floor listed-starter shares when no material usage lag exists (rookies).

    Without this, depth-chart RB1/WR1 rookies keep near-zero model shares while
    veterans with priors absorb the room — then the rookie FP prior invents
    elite fantasy lines with no counting stats (Love/Tate-style disconnects).
    Tiny stub priors (< material threshold) are treated as missing so they
    cannot block the role floor.
    """
    if df.is_empty() or "depth_rank" not in df.columns or "position" not in df.columns:
        return df
    out = df
    depth = pl.col("depth_rank").cast(pl.Float64)
    is_one = depth.is_not_null() & (depth.round() == 1.0)
    material = 0.05

    if "pred_carry_share" in out.columns:
        carry_anchor = pl.lit(0.0)
        if "carry_share_l5" in out.columns:
            carry_anchor = pl.max_horizontal(carry_anchor, pl.col("carry_share_l5").fill_null(0.0))
        if "carry_share_prior_season" in out.columns:
            carry_anchor = pl.max_horizontal(
                carry_anchor, pl.col("carry_share_prior_season").fill_null(0.0)
            )
        rb1_floor = is_one & (pl.col("position") == "RB") & (carry_anchor < material)
        out = out.with_columns(
            pl.when(rb1_floor)
            .then(pl.max_horizontal(pl.col("pred_carry_share").fill_null(0.0), pl.lit(0.42)))
            .otherwise(pl.col("pred_carry_share"))
            .alias("pred_carry_share")
        )

    if "pred_target_share" in out.columns:
        tgt_anchor = pl.lit(0.0)
        if "target_share_l5" in out.columns:
            tgt_anchor = pl.max_horizontal(tgt_anchor, pl.col("target_share_l5").fill_null(0.0))
        if "target_share_prior_season" in out.columns:
            tgt_anchor = pl.max_horizontal(
                tgt_anchor, pl.col("target_share_prior_season").fill_null(0.0)
            )
        wr1_floor = is_one & (pl.col("position") == "WR") & (tgt_anchor < material)
        te1_floor = is_one & (pl.col("position") == "TE") & (tgt_anchor < material)
        out = out.with_columns(
            pl.when(wr1_floor)
            .then(pl.max_horizontal(pl.col("pred_target_share").fill_null(0.0), pl.lit(0.18)))
            .when(te1_floor)
            .then(pl.max_horizontal(pl.col("pred_target_share").fill_null(0.0), pl.lit(0.14)))
            .otherwise(pl.col("pred_target_share"))
            .alias("pred_target_share")
        )
    return out
