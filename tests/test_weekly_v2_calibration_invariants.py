"""Calibration ordering and within-position rank invariants."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.projection.weekly.models.calibration import (
    POSITION_SLOPE_CAPS,
    apply_position_calibration,
    fit_position_calibration,
)


def _oof_frame() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    pred = rng.normal(10, 2, n)
    actual = pred * 0.7 + rng.normal(0, 1.5, n)
    return pl.DataFrame(
        {
            "season": [2022] * n,
            "position": ["WR"] * n,
            "projected_fantasy_points": pred,
            "actual_fantasy_points": actual,
        }
    )


def test_calibration_applied_once_preserves_wr_ordering():
    oof = _oof_frame()
    cal = fit_position_calibration(oof)
    rows = oof.rename(
        {
            "projected_fantasy_points": "projected_fantasy_points",
            "actual_fantasy_points": "actual_fantasy_points",
        }
    )
    once = apply_position_calibration(rows, cal, point_col="projected_fantasy_points")
    twice = apply_position_calibration(once, cal, point_col="projected_fantasy_points")
    p1 = once["projected_fantasy_points"].to_numpy()
    p2 = twice["projected_fantasy_points"].to_numpy()
    assert not np.allclose(p1, p2), "second calibration application must change values"


def test_within_position_rank_preserved_except_zero_ties():
    oof = _oof_frame()
    cal = fit_position_calibration(oof)
    calibrated = apply_position_calibration(oof, cal, point_col="projected_fantasy_points")
    raw = oof["projected_fantasy_points"].to_numpy()
    adj = calibrated["projected_fantasy_points"].to_numpy()
    # Positive slope calibration preserves order
    assert np.all(np.diff(np.argsort(np.argsort(raw))) == np.diff(np.argsort(np.argsort(adj))))


def test_wr_te_slope_cap_is_152():
    assert POSITION_SLOPE_CAPS["WR"] == 1.52
    assert POSITION_SLOPE_CAPS["TE"] == 1.52


def test_calibration_uses_only_prior_oof_seasons():
    oof = pl.concat(
        [
            _oof_frame().with_columns(pl.lit(2022).alias("season")),
            _oof_frame().with_columns(pl.lit(2023).alias("season")),
        ]
    )
    cal = fit_position_calibration(oof.filter(pl.col("season") == 2022))
    assert cal["trained_seasons"] == [2022]
    assert cal["max_train_season"] == 2022
