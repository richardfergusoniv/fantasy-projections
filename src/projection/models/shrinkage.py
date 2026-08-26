"""Empirical-Bayes shrinkage utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def shrink_rate(
    observed: pd.Series,
    prior_mean: float,
    *,
    n: pd.Series | None = None,
    strength: float = 25.0,
) -> pd.Series:
    obs = pd.to_numeric(observed, errors="coerce").fillna(0.0)
    if n is None:
        weight = pd.Series(0.5, index=obs.index)
    else:
        weight = pd.to_numeric(n, errors="coerce").fillna(0.0) / (
            pd.to_numeric(n, errors="coerce").fillna(0.0) + strength
        )
    return weight * obs + (1.0 - weight) * prior_mean


def hierarchical_offsets(
    frame: pd.DataFrame,
    rate_col: str,
    *,
    group_cols: list[str],
    strength: float = 20.0,
) -> pd.Series:
    """Partial-pool rates toward group means."""
    rates = pd.to_numeric(frame[rate_col], errors="coerce").fillna(0.0)
    global_mean = float(rates.mean()) if len(rates) else 0.0
    adjusted = rates.copy()
    for keys, grp in frame.groupby(group_cols, observed=True):
        idx = grp.index
        group_mean = float(rates.loc[idx].mean())
        n = len(idx)
        weight = n / (n + strength)
        adjusted.loc[idx] = weight * rates.loc[idx] + (1.0 - weight) * group_mean
    return shrink_rate(adjusted, global_mean, strength=strength)
