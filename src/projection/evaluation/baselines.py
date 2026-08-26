"""Reference baselines for position-stat model comparison.

Every candidate model must beat these on rolling out-of-sample folds before
promotion (see implementation guide).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, RidgeCV

from src.projection.transitions import (
    ROLE_PRIOR_3Y_FEATURE,
    ROLE_PRIOR_FEATURE,
    TEAM_TOTAL_LABEL,
    role_rate_label,
    role_features_for,
)

BASELINE_NAMES = (
    "prior_year_rate",
    "weighted_3y",
    "team_share_volume",
    "ridge",
    "elastic_net",
    "empirical_bayes",
)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def prior_year_rate(frame: pd.DataFrame, stat: str) -> pd.Series:
    """Carry forward season_from per-eligible-week rate unchanged."""
    if "naive_pred" in frame.columns:
        return _safe_numeric(frame["naive_pred"]).fillna(0.0)
    col = role_rate_label(stat)
    if col in frame.columns:
        return _safe_numeric(frame[col]).fillna(0.0)
    return pd.Series(0.0, index=frame.index)


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return _safe_numeric(frame[name])
    return pd.Series(np.nan, index=frame.index)


def weighted_3y_average(frame: pd.DataFrame, stat: str) -> pd.Series:
    """Blend prior-year and 3-year role rates with mild age decay."""
    prior = _col(frame, ROLE_PRIOR_FEATURE).fillna(0.0)
    prior_3y = _col(frame, ROLE_PRIOR_3Y_FEATURE)
    blended = prior_3y.fillna(prior)
    age = _col(frame, "age").fillna(27.0)
    decay = np.clip(1.0 - 0.015 * (age - 27.0), 0.85, 1.15)
    return (blended * decay).clip(lower=0.0)


def team_share_times_volume(frame: pd.DataFrame, stat: str) -> pd.Series:
    """Player share of team volume × projected team total (receiving proxy)."""
    share_col = f"{stat}_share_elig" if f"{stat}_share_elig" in frame.columns else None
    if share_col is None and stat == "receiving_yards":
        share_col = "receiving_yards_share_elig"
    team_total = _col(frame, TEAM_TOTAL_LABEL).fillna(0.0)
    if share_col and share_col in frame.columns:
        share = _safe_numeric(frame[share_col]).fillna(0.0).clip(0.0, 1.5)
        return (share * team_total).clip(lower=0.0)
    prior = prior_year_rate(frame, stat)
    team_prior = _col(frame, "team_naive_pred").fillna(team_total)
    denom = _col(frame, "team_pass_attempts_pg").replace(0, np.nan)
    if stat in {"receiving_yards", "receptions", "targets"} and denom is not None:
        player_att = _col(frame, "pass_attempts_pg").fillna(0.0)
        share = (player_att / denom).fillna(0.0).clip(0.0, 1.0)
        return (share * team_prior).clip(lower=0.0)
    return prior


def ridge_elastic_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    position: str,
    stat: str,
    *,
    model_kind: str = "ridge",
) -> pd.Series:
    """Simple linear baseline on role features."""
    y_col = role_rate_label(stat)
    features = [c for c in role_features_for(position, stat) if c in train.columns]
    if not features or y_col not in train.columns or train.empty or test.empty:
        return prior_year_rate(test, stat)
    x_train = train[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_test = test[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_train = _safe_numeric(train[y_col]).fillna(0.0)
    if model_kind == "elastic_net":
        model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000)
    else:
        model = RidgeCV(alphas=np.logspace(-2, 2, 15))
    model.fit(x_train, y_train)
    return pd.Series(np.clip(model.predict(x_test), 0, None), index=test.index)


def empirical_bayes_shrunk_rate(
    frame: pd.DataFrame,
    stat: str,
    *,
    position_mean: float | None = None,
    k: float = 25.0,
) -> pd.Series:
    """Shrink player prior toward position mean with strength k."""
    prior = prior_year_rate(frame, stat)
    pos = frame.get("position")
    if position_mean is None and pos is not None:
        position_mean = float(prior.groupby(pos).transform("mean").mean())
    position_mean = position_mean if position_mean is not None else float(prior.mean())
    games = _col(frame, "games_played").fillna(0.0)
    weight = games / (games + k)
    return (weight * prior + (1.0 - weight) * position_mean).clip(lower=0.0)


def attach_all_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    position: str,
    stat: str,
) -> pd.DataFrame:
    """Return test frame with every baseline prediction column attached."""
    out = test.copy()
    out["baseline_prior_year"] = prior_year_rate(out, stat)
    out["baseline_weighted_3y"] = weighted_3y_average(out, stat)
    out["baseline_team_share_volume"] = team_share_times_volume(out, stat)
    out["baseline_ridge"] = ridge_elastic_baseline(train, out, position, stat, model_kind="ridge")
    out["baseline_elastic_net"] = ridge_elastic_baseline(
        train, out, position, stat, model_kind="elastic_net"
    )
    out["baseline_empirical_bayes"] = empirical_bayes_shrunk_rate(out, stat)
    return out


def baseline_mae(actual: pd.Series, predicted: pd.Series) -> float:
    a = _safe_numeric(actual)
    p = _safe_numeric(predicted)
    mask = a.notna() & p.notna()
    if not mask.any():
        return float("nan")
    return float((a[mask] - p[mask]).abs().mean())
