"""Tests for compositional share allocation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.models.opportunity_shares import allocate_opportunities, draw_dirichlet_shares


def test_dirichlet_shares_sum_to_one():
    players = pd.DataFrame({"pred_pg": [10.0, 5.0, 2.0]})
    rng = np.random.default_rng(0)
    shares = draw_dirichlet_shares(players, concentration=10.0, rng=rng)
    assert abs(shares.sum() - 1.0) < 1e-9


def test_allocate_opportunities_preserves_total():
    players = pd.DataFrame({
        "team": ["AAA", "AAA"],
        "position": ["WR", "WR"],
        "stat": ["targets", "targets"],
        "pred_pg": [8.0, 4.0],
    })
    rng = np.random.default_rng(1)
    out = allocate_opportunities(players, team_volume=100.0, rng=rng)
    assert abs(out["allocated_volume"].sum() - 100.0) < 1e-6
