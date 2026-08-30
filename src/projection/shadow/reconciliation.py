"""Reconciliation diagnostics for shadow opportunity models."""
from __future__ import annotations

import pandas as pd


def team_target_residual(
    shadow_targets: pd.DataFrame,
    incumbent_targets: pd.DataFrame,
    *,
    team_col: str = "team",
    value_col: str = "pred_season",
    shadow_col: str = "pred_season_shadow",
) -> pd.DataFrame:
    """Absolute team-level residual between shadow and incumbent target totals."""
    inc = (
        incumbent_targets.groupby(team_col, observed=True)[value_col]
        .sum()
        .rename("incumbent_total")
    )
    sh = (
        shadow_targets.groupby(team_col, observed=True)[shadow_col]
        .sum()
        .rename("shadow_total")
    )
    merged = pd.concat([inc, sh], axis=1).fillna(0.0)
    merged["abs_residual"] = (merged["shadow_total"] - merged["incumbent_total"]).abs()
    return merged.reset_index()


def reconciliation_burden_score(team_residuals: pd.DataFrame) -> float:
    if team_residuals.empty:
        return 0.0
    return float(team_residuals["abs_residual"].mean())
