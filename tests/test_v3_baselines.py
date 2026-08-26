"""Tests for v3 baseline reference models."""
from __future__ import annotations

import pandas as pd

from src.projection.evaluation.baselines import (
    attach_all_baselines,
    empirical_bayes_shrunk_rate,
    prior_year_rate,
    weighted_3y_average,
)


def test_prior_year_rate_uses_naive_pred():
    frame = pd.DataFrame({"naive_pred": [1.5, 2.0]})
    out = prior_year_rate(frame, "receiving_yards")
    assert out.tolist() == [1.5, 2.0]


def test_weighted_3y_blends_prior_columns():
    frame = pd.DataFrame({
        "prior_role_rate": [10.0, 8.0],
        "prior_role_rate_3y": [9.0, 7.0],
        "age": [27.0, 30.0],
    })
    out = weighted_3y_average(frame, "receiving_yards")
    assert out.iloc[0] == 9.0
    assert out.iloc[1] < 7.0


def test_empirical_bayes_shrinks_toward_mean():
    frame = pd.DataFrame({
        "naive_pred": [20.0, 0.5],
        "games_played": [1.0, 16.0],
        "position": ["WR", "WR"],
    })
    out = empirical_bayes_shrunk_rate(frame, "receiving_yards", k=8.0)
    assert out.iloc[0] < 20.0
    assert out.iloc[1] > 0.4


def test_attach_all_baselines_adds_columns():
    train = pd.DataFrame({
        "prior_role_rate": [5.0],
        "prior_role_rate_3y": [5.0],
        "age": [26.0],
        "games_played": [16.0],
        "naive_pred": [5.0],
        "receiving_yards_pg": [5.0],
        "position": ["WR"],
        "depth_tier": [1],
        "team_passing_yards_pg": [250.0],
        "team_naive_pred": [240.0],
    })
    test = train.copy()
    out = attach_all_baselines(train, test, "WR", "receiving_yards")
    baseline_cols = [c for c in out.columns if c.startswith("baseline_")]
    assert len(baseline_cols) >= 5
