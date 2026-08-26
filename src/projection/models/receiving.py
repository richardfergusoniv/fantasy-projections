"""Conditional generative receiving models."""
from __future__ import annotations

import numpy as np
import pandas as pd


def draw_receiving_line(
    targets: float,
    *,
    catch_rate: float = 0.65,
    yards_per_rec: float = 12.0,
    td_rate: float = 0.04,
    rng: np.random.Generator,
) -> dict[str, float]:
    t = max(float(rng.poisson(max(targets, 0.01))), 0)
    rec = int(rng.binomial(t, min(max(catch_rate, 0.01), 0.99))) if t else 0
    yards = float(rng.lognormal(mean=np.log(max(yards_per_rec, 1.0)), sigma=0.35) * rec)
    tds = int(rng.poisson(td_rate * max(t, 1)))
    return {"targets": t, "receptions": rec, "receiving_yards": yards, "receiving_tds": tds}
