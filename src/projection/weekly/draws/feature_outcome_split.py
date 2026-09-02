"""Strict separation of pre-kickoff prediction features from same-week outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from src.projection.weekly.features.leakage import OUTCOME_COLUMNS

# Same-week actuals and shares that must never enter inference builders.
SAME_WEEK_OUTCOME_DENYLIST = frozenset(
    OUTCOME_COLUMNS
    | {
        "target_share",
        "carry_share",
        "snap_share",
        "wopr",
        "racr",
        "team_targets",
        "team_carries",
        "offense_snaps",
        "offense_pct",
        "passing_first_downs",
        "rushing_first_downs",
        "receiving_first_downs",
        "fantasy_points",
        "fantasy_points_ppr",
        "fantasy_points_half_ppr",
        "active_label",
        "participated_label",
        "positive_usage_label",
        "row_outcome_state",
        "outcome_missing",
        "has_boxscore",
    }
)

# Allowed pre-kickoff feature prefixes (lagged columns).
LAGGED_FEATURE_SUFFIXES = ("_l1", "_l3", "_l5", "_prior", "_prev_week", "_lag1", "_lag3", "_lag5")


@dataclass(frozen=True)
class FeatureOutcomeManifest:
    prediction_columns: tuple[str, ...]
    outcome_columns: tuple[str, ...]
    denylist: frozenset[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_columns": list(self.prediction_columns),
            "outcome_columns": list(self.outcome_columns),
            "denylist": sorted(self.denylist),
        }


def is_allowed_prediction_column(name: str) -> bool:
    if name in SAME_WEEK_OUTCOME_DENYLIST:
        return False
    # Raw same-week shares without lag suffix are denied.
    base = name
    for suf in LAGGED_FEATURE_SUFFIXES:
        if name.endswith(suf):
            base = name[: -len(suf)]
            break
    if base in SAME_WEEK_OUTCOME_DENYLIST and not any(name.endswith(s) for s in LAGGED_FEATURE_SUFFIXES):
        return False
    return True


def assert_no_outcome_columns(columns: list[str] | tuple[str, ...]) -> None:
    blocked = [c for c in columns if not is_allowed_prediction_column(c)]
    if blocked:
        raise ValueError(
            f"inference frame contains same-week outcome columns: {sorted(blocked)[:20]}"
        )


def split_prediction_outcome_frames(
    frame: pl.DataFrame,
    *,
    extra_prediction_cols: tuple[str, ...] = (),
    extra_outcome_cols: tuple[str, ...] = (),
) -> tuple[pl.DataFrame, pl.DataFrame, FeatureOutcomeManifest]:
    """Split a cohort frame into prediction features vs outcome labels."""
    keys = [c for c in ("gsis_id", "season", "week", "team", "position", "game_id") if c in frame.columns]
    prediction_cols = list(keys) + [
        c
        for c in frame.columns
        if c not in keys
        and is_allowed_prediction_column(c)
        and c not in SAME_WEEK_OUTCOME_DENYLIST
    ]
    prediction_cols.extend(c for c in extra_prediction_cols if c in frame.columns and c not in prediction_cols)
    outcome_cols = list(keys) + [
        c
        for c in frame.columns
        if c in SAME_WEEK_OUTCOME_DENYLIST or c in extra_outcome_cols
    ]
    outcome_cols = list(dict.fromkeys(outcome_cols))
    manifest = FeatureOutcomeManifest(
        prediction_columns=tuple(prediction_cols),
        outcome_columns=tuple(outcome_cols),
        denylist=SAME_WEEK_OUTCOME_DENYLIST,
    )
    return frame.select(prediction_cols), frame.select(outcome_cols), manifest


def poison_outcomes_must_not_change_predictions(
    predict_fn,
    features: pl.DataFrame,
    outcomes: pl.DataFrame,
    *,
    keys: tuple[str, ...] = ("gsis_id", "season", "week"),
) -> None:
    """Changing target-week outcomes must not change predictions from predict_fn."""
    base = predict_fn(features)
    poisoned = outcomes.with_columns(
        [
            (pl.col("fantasy_points").fill_null(0.0) + 99.0).alias("fantasy_points")
            if "fantasy_points" in outcomes.columns
            else pl.lit(99.0).alias("fantasy_points"),
            (pl.col("targets").fill_null(0.0) + 50.0).alias("targets")
            if "targets" in outcomes.columns
            else pl.lit(50.0).alias("targets"),
        ]
    )
    merged_poison = features.join(poisoned.select(list(keys) + ["fantasy_points", "targets"]), on=list(keys), how="left")
    alt = predict_fn(merged_poison)
    if hasattr(base, "to_numpy"):
        same = (base == alt).all()
        if hasattr(same, "item"):
            same = same.item()
        if not same:
            raise AssertionError("poisoned outcomes changed predictions")
    elif isinstance(base, dict) and isinstance(alt, dict):
        if base != alt:
            raise AssertionError("poisoned outcomes changed predictions")
    else:
        import numpy as np

        if not np.allclose(np.asarray(base), np.asarray(alt), equal_nan=True):
            raise AssertionError("poisoned outcomes changed predictions")
