"""Tests for v3 Monte Carlo simulation."""
from __future__ import annotations

import pandas as pd

from src.projection.inference.simulate import simulate_season_distributions, summarize_simulations


def test_simulate_produces_draws():
    projections = pd.DataFrame({
        "player_id": ["p1", "p1", "p2", "p2"],
        "position": ["WR", "WR", "RB", "RB"],
        "team": ["AAA", "AAA", "BBB", "BBB"],
        "stat": ["receiving_yards", "receptions", "rushing_yards", "carries"],
        "pred_pg": [60.0, 5.0, 40.0, 10.0],
        "projected_games": [16.0, 16.0, 15.0, 15.0],
    })
    draws = simulate_season_distributions(projections, n_draws=50, seed=1)
    assert len(draws) == 100
    assert "fantasy_pts_season" in draws.columns


def test_summarize_percentiles():
    draws = pd.DataFrame({
        "player_id": ["p1"] * 10,
        "position": ["WR"] * 10,
        "team": ["AAA"] * 10,
        "fantasy_pts_season": list(range(100, 110)),
        "draw": list(range(10)),
    })
    summary = summarize_simulations(draws)
    assert "p50" in summary.columns
    assert summary.iloc[0]["p50"] == 104.5
