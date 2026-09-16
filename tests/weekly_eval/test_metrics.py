"""Primary losses for Role 2 weekly prop comparison (synthetic)."""
from __future__ import annotations

import numpy as np
import pytest

from src.projection.weekly_eval.metrics import (
    brier_score,
    gaussian_crps,
    mean_absolute_error,
    pinball_loss,
    root_mean_squared_error,
)


def test_mae_and_rmse_on_known_errors():
    pred = np.array([10.0, 20.0, 30.0])
    actual = np.array([12.0, 18.0, 30.0])
    assert mean_absolute_error(pred, actual) == pytest.approx(4.0 / 3.0)
    assert root_mean_squared_error(pred, actual) == pytest.approx(
        float(np.sqrt((4.0 + 4.0 + 0.0) / 3.0))
    )


def test_brier_is_proper_for_over_under():
    # Over realized. Confident 0.8 beats a coin-flip 0.5.
    overs = np.array([1.0, 1.0, 0.0])
    sharp = np.array([0.8, 0.9, 0.2])
    flat = np.array([0.5, 0.5, 0.5])
    assert brier_score(overs, sharp) < brier_score(overs, flat)
    assert brier_score(overs, flat) == pytest.approx(0.25)


def test_gaussian_crps_is_proper_and_finite():
    actual = np.array([80.0, 90.0])
    close = gaussian_crps(actual, mean=np.array([81.0, 88.0]), std=np.array([12.0, 12.0]))
    far = gaussian_crps(actual, mean=np.array([40.0, 40.0]), std=np.array([12.0, 12.0]))
    assert close < far
    assert np.isfinite(close)


def test_pinball_at_median_is_half_mae():
    pred = np.array([10.0, 20.0])
    actual = np.array([14.0, 18.0])
    assert pinball_loss(actual, 0.5, pred) == pytest.approx(0.5 * mean_absolute_error(pred, actual))
