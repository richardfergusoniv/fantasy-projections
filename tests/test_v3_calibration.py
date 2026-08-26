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
