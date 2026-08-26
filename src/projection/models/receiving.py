"""Conditional generative receiving models."""
from __future__ import annotations

import numpy as np
import pandas as pd


# Measured SD of log(actual season efficiency / predicted), from
# output/backtest/residuals_rolling.parquet: RB 0.315, WR 0.273, TE 0.242.
# The previous 0.35 was chosen when this drew PER-GAME lines, where a
# player's efficiency genuinely swings more than it does over a season.
SEASON_YPR_SIGMA = 0.28


def draw_receiving_line(
    targets: float,
    *,
    catch_rate: float = 0.65,
    yards_per_rec: float = 12.0,
    td_rate: float = 0.04,
    sigma: float = SEASON_YPR_SIGMA,
    rng: np.random.Generator,
) -> dict[str, float]:
    t = max(float(rng.poisson(max(targets, 0.01))), 0)
    rec = int(rng.binomial(t, min(max(catch_rate, 0.01), 0.99))) if t else 0
    yards = float(rng.lognormal(mean=np.log(max(yards_per_rec, 1.0)), sigma=sigma) * rec)
    tds = int(rng.poisson(td_rate * max(t, 1)))
    return {"targets": t, "receptions": rec, "receiving_yards": yards, "receiving_tds": tds}
