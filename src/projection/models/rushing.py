"""Conditional generative rushing models."""
from __future__ import annotations

import numpy as np


def draw_rushing_line(
    carries: float,
    *,
    ypc: float = 4.3,
    td_rate: float = 0.02,
    rng: np.random.Generator,
) -> dict[str, float]:
    c = max(float(rng.poisson(max(carries, 0.01))), 0)
    yards = float(rng.lognormal(mean=np.log(max(ypc, 0.5)), sigma=0.25) * c)
    tds = int(rng.poisson(td_rate * max(c, 1)))
    return {"carries": c, "rushing_yards": yards, "rushing_tds": tds}
