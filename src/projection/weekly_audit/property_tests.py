"""As-of property tests for weekly feature contracts."""
from __future__ import annotations

import pandas as pd


def prior_weeks_unchanged_when_future_mutated(
    features_before: dict[int, float],
    features_after: dict[int, float],
    *,
    mutated_week: int,
) -> bool:
    """Features for weeks < mutated_week must be identical before/after mutation."""
    for week in features_before:
        if week < mutated_week and features_before[week] != features_after.get(week):
            return False
    return True


def target_week_unchanged_when_target_outcomes_mutated(
    feature_before: float,
    feature_after: float,
  *,
  tolerance: float = 1e-9,
) -> bool:
    return abs(feature_before - feature_after) <= tolerance


def future_week_may_change_after_target_mutation(
    feature_before: float,
    feature_after: float,
    *,
    tolerance: float = 1e-9,
) -> bool:
    return abs(feature_before - feature_after) > tolerance


def rolling_window_respects_kickoff_cutoff(
    observation_weeks: list[int],
    *,
    target_week: int,
) -> bool:
    """No observation at or after target week may enter a week-w feature."""
    return all(week < target_week for week in observation_weeks)


def week1_uses_only_prior_season(
    feature_week1: float,
    *,
    prior_season_only_value: float,
    tolerance: float = 1e-9,
) -> bool:
    return abs(feature_week1 - prior_season_only_value) <= tolerance


def build_weekly_feature_matrix(
    weekly: pd.DataFrame,
    *,
    value_col: str,
    player_id: str,
    shift_lag: int = 1,
) -> pd.DataFrame:
    """Leakage-safe rolling mean using only prior weeks (shifted)."""
    sub = weekly[weekly["player_id"].eq(player_id)].sort_values("week")
    values = pd.to_numeric(sub[value_col], errors="coerce")
    rolled = values.shift(shift_lag).rolling(3, min_periods=1).mean()
    out = sub[["season", "week"]].copy()
    out["feature_value"] = rolled.to_numpy()
    return out
