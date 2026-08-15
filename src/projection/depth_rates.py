"""Gate B depth-rate ladder lookup, and the ONE rule for applying it.

Leaf module — imports contracts only (plus numpy/pandas). Do not import
predict.

Why the application rule lives here
-----------------------------------
The ladder used to be applied in three places, by three different rules:

  * ``depth_gating.apply_depth_chart_gating`` applied it only when a curated
    depth chart existed for the target season — and ``load_depth_chart``
    returns a chart for 2026 and nothing else, so the shipped path applied
    the ladder for exactly one season;
  * ``fantasy_evaluation._veteran_forecasts`` applied it unconditionally,
    which is what every leakage-safe fold has actually been measuring;
  * ``backtest.py`` never applied it at all, so ``interval_residuals.csv``
    and the elite-shrinkage coefficients in ``corrections.joblib`` were fit
    on UNDISCOUNTED predictions and then consumed by a path that ships
    discounted ones.

The curated-chart gate was a leftover. Before Gate B (``720fa8e``) the
multiplier was selected by the curated ``role`` column, so "no curated chart
⇒ no multiplier" was simply true. Gate B re-keyed the multiplier onto
``nfl_depth_rank``, which ``depth_history.py`` reconstructs for every season
from nflverse — and the guard was never revisited. See
GATE_B_UNIFICATION.md.

The rule, stated once
---------------------
The Gate-B factor is a function of ``(position, nfl_depth_rank)`` and
NOTHING else. It applies to every veteran row, in every season, on every
path. The curated chart still governs the things it is actually
authoritative for — membership, team, displayed role, ``low_confidence`` —
and none of those select a multiplier.

Rookie rows are excluded, and that exclusion is a fit boundary rather than a
coverage gap: the ladder was fit on veteran transition pairs, and the
veteran-only ladder was measured harmful on the dedicated rookie test. See
``team_reconcile._apply_rookie_depth_rate_gating``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.contracts import (
    DEPTH_RATE_DEEP,
    DEPTH_RATE_LADDER,
    DEPTH_RATE_OFF_CHART,
)

# The columns the ladder scales wherever a frame carries them. pred_pg is the
# point prediction (a rate, or a receiving SHARE on reframed rows — the share
# is discounted before renormalization, which is what lets the team-level cap
# see discounted shares). low/high are scaled only where they already exist;
# on the shipped veteran path they are attached AFTER this runs, so the
# interval residual is added to an already-discounted prediction on the same
# basis backtest.py now fits it.
LADDER_SCALED_COLUMNS = ("pred_pg", "pred_pg_low", "pred_pg_high")

DEPTH_RANK_COLUMN = "nfl_depth_rank"
DEPTH_FACTOR_COLUMN = "role_discount_factor"


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


def depth_rate_factors(positions, ranks):
    """Vectorised :func:`depth_rate_factor` over aligned sequences.

    Returns a float ndarray, never a Series, so a caller cannot pick up an
    index alignment it did not ask for.
    """
    positions = list(positions)
    ranks = list(ranks)
    if len(positions) != len(ranks):
        raise ValueError(
            f"depth_rate_factors got {len(positions)} positions and "
            f"{len(ranks)} ranks; they must be aligned row for row")
    return np.array(
        [depth_rate_factor(p, r) for p, r in zip(positions, ranks)], dtype=float)


def attach_depth_rate_factor(df, rank_col=DEPTH_RANK_COLUMN,
                             out_col=DEPTH_FACTOR_COLUMN):
    """Write the Gate-B factor into ``out_col`` without scaling anything.

    Raises when ``rank_col`` is absent rather than defaulting it to NaN:
    NaN means "off the preseason chart" and carries a real discount, so a
    missing column would silently apply the off-chart factor to every row.
    """
    if rank_col not in df.columns:
        raise ValueError(
            f"the Gate B depth-rate ladder needs {rank_col} — call "
            f"depth_history.attach_depth_rank(df, target_season) first. "
            f"Defaulting it to NaN would silently apply the off-chart factor "
            f"to every player.")
    out = df.copy()
    out[out_col] = depth_rate_factors(out["position"], out[rank_col])
    return out


def apply_depth_rate_ladder(df, columns=LADDER_SCALED_COLUMNS,
                            rank_col=DEPTH_RANK_COLUMN,
                            out_col=DEPTH_FACTOR_COLUMN):
    """THE Gate-B application: attach the factor and scale ``columns`` by it.

    Unconditional on the curated depth chart by design — see this module's
    docstring. Columns absent from ``df`` are skipped rather than created,
    so a frame that has not yet built its interval endpoints is left alone
    instead of gaining two columns of NaN.
    """
    out = attach_depth_rate_factor(df, rank_col=rank_col, out_col=out_col)
    factor = out[out_col].to_numpy(dtype=float)
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").to_numpy() * factor
    return out
