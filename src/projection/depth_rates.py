"""Gate B depth-rate ladder lookup.

Leaf module — imports contracts only (plus numpy). Do not import predict.
"""
from __future__ import annotations

import numpy as np

from src.projection.contracts import (
    DEPTH_RATE_DEEP,
    DEPTH_RATE_LADDER,
    DEPTH_RATE_OFF_CHART,
)


def depth_rate_factor(position, rank):
    """The Gate B volume multiplier for one (position, preseason rank).
    NaN rank = off the chart. Unknown position falls back to 1.0 (no
    discount) rather than to a guess: this ladder was fit per position and
    has nothing to say about one it never saw."""
    if position not in DEPTH_RATE_LADDER:
        return 1.0
    if rank is None or (isinstance(rank, float) and np.isnan(rank)):
        return DEPTH_RATE_OFF_CHART[position]
    rung = DEPTH_RATE_LADDER[position]
    return rung.get(int(rank), DEPTH_RATE_DEEP[position])
