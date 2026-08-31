"""Conditional generative passing models."""
from __future__ import annotations

import numpy as np


def draw_passing_line(
    attempts: float,
    *,
    comp_rate: float = 0.64,
    yards_per_comp: float = 11.0,
    td_rate: float = 0.045,
    int_rate: float = 0.025,
    sigma: float = 0.27,
    rng: np.random.Generator,
) -> dict[str, float]:
    att = max(float(rng.poisson(max(attempts, 0.01))), 0)
    comp = int(rng.binomial(att, min(max(comp_rate, 0.01), 0.99))) if att else 0
    yards = float(rng.lognormal(mean=np.log(max(yards_per_comp, 1.0)), sigma=sigma) * comp)
    tds = int(rng.poisson(td_rate * max(att, 1)))
    ints = int(rng.binomial(att, min(max(int_rate, 0.001), 0.15))) if att else 0
    return {
        "pass_attempts": att,
        "completions": comp,
        "passing_yards": yards,
        "passing_tds": tds,
        "interceptions": ints,
    }
