"""Tests for calibration metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.evaluation.calibration import (
    coverage_by_group,
    crps_gaussian,
    crps_sample,
    pinball_loss,
    reliability_table,
    summarize_interval_calibration,
)


def test_pinball_loss_symmetric_at_median():
    actual = np.array([1.0, 2.0, 3.0])
    pred = np.array([2.0, 2.0, 2.0])
    loss = pinball_loss(actual, 0.5, pred)
    assert loss >= 0.0


def test_crps_sample_perfect_forecast():
    actual = np.array([1.0, 2.0])
    samples = np.array([[1.0, 1.0], [2.0, 2.0]])
    assert crps_sample(actual, samples) == 0.0


def test_crps_gaussian_finite():
    actual = np.array([10.0, 20.0])
    mean = np.array([9.0, 21.0])
    std = np.array([2.0, 3.0])
    score = crps_gaussian(actual, mean, std)
    assert np.isfinite(score)


def test_reliability_table_bins():
    actual = pd.Series(range(100), dtype=float)
    predicted = pd.Series(range(100), dtype=float) + 0.5
    table = reliability_table(actual, predicted, n_bins=5)
    assert len(table) >= 1
    assert "mean_actual" in table.columns


def test_summarize_interval_calibration():
    residuals = pd.DataFrame({
        "position": ["WR"] * 20,
        "stat": ["receiving_yards"] * 20,
        "pred": np.linspace(50, 70, 20),
        "actual": np.linspace(48, 72, 20),
        "resid": np.linspace(-2, 2, 20),
        "test_season": [2024] * 20,
    })
    summary = summarize_interval_calibration(residuals)
    assert summary["n"] == 20
    assert "mean_coverage" in summary


def test_coverage_by_group():
    frame = pd.DataFrame({
        "position": ["WR", "WR", "RB", "RB"],
        "actual": [10, 20, 5, 15],
        "pred_low": [8, 18, 4, 14],
        "pred_high": [12, 22, 6, 16],
    })
    cov = coverage_by_group(frame, group_cols=["position"])
    assert len(cov) == 2
    assert cov["coverage"].eq(1.0).all()


def _residuals_with_a_biased_late_season():
    """Bands calibrated on 2023-24 meet a 2025 whose residuals shifted.

    In-sample scoring cannot see this: it refits the band on 2025's own
    residuals. Held-out scoring must.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    rows = []
    for season, shift in ((2023, 0.0), (2024, 0.0), (2025, 30.0)):
        resid = rng.normal(shift, 1.0, 200)
        for r in resid:
            rows.append({
                "position": "QB", "stat": "attempts", "test_season": season,
                "pred": 30.0, "actual": 30.0 + r, "resid": r,
            })
    return pd.DataFrame(rows)


def test_in_sample_coverage_hits_target_by_construction():
    """Pins the defect, so a gate cannot go back to resting on this.

    The band IS the empirical quantile of the rows it scores, so coverage
    lands on 0.80 even when the last season is wildly mis-predicted.
    """
    from src.projection.evaluation.calibration import summarize_interval_calibration

    out = summarize_interval_calibration(_residuals_with_a_biased_late_season())
    assert out["basis"] == "in_sample"
    assert abs(out["mean_coverage"] - 0.80) < 0.02


def test_forward_coverage_catches_what_in_sample_cannot():
    from src.projection.evaluation.calibration import (
        summarize_forward_interval_calibration,
        summarize_interval_calibration,
    )
    residuals = _residuals_with_a_biased_late_season()
    in_sample = summarize_interval_calibration(residuals)
    forward = summarize_forward_interval_calibration(residuals)

    assert forward["basis"] == "forward_holdout"
    # 2025's shifted residuals fall entirely outside a 2023-24 band.
    assert forward["mean_coverage"] < 0.60
    assert forward["mean_coverage"] < in_sample["mean_coverage"] - 0.20
    # The first season of a group has nothing before it and is not scored.
    assert forward["n_scored"] < len(residuals)


def test_forward_coverage_is_empty_without_a_season_column():
    from src.projection.evaluation.calibration import summarize_forward_interval_calibration

    frame = pd.DataFrame({
        "position": ["QB"], "stat": ["attempts"],
        "pred": [30.0], "actual": [31.0], "resid": [1.0],
    })
    out = summarize_forward_interval_calibration(frame)
    assert out["n_scored"] == 0
    assert "mean_coverage" not in out
