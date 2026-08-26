"""Stochastic availability process."""
from __future__ import annotations

import numpy as np
import pandas as pd


def draw_games_played(
    expected_games: pd.Series,
    *,
    rng: np.random.Generator,
    max_games: int = 17,
) -> pd.Series:
    """Draw season games from a truncated normal around the point forecast."""
    mu = pd.to_numeric(expected_games, errors="coerce").fillna(max_games * 0.85)
    sigma = np.clip(mu * 0.15, 1.0, 4.0)
    draws = rng.normal(mu.to_numpy(), sigma)
    return pd.Series(np.clip(np.rint(draws), 0, max_games), index=expected_games.index)


def draw_weekly_availability(
    n_weeks: int,
    p_active: float,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bernoulli weekly active process with persistence."""
    states = np.zeros(n_weeks, dtype=int)
    state = 1 if rng.random() < p_active else 0
    for w in range(n_weeks):
        if rng.random() < 0.08:
            state = 1 - state
        elif rng.random() > p_active:
            state = 0
        states[w] = state
    return states
