"""Primary losses for Role 2 weekly fantasy / prop comparison."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

# Used when a row has no std and p_over is ~0.5 so sigma is unidentified.
DEFAULT_LOCATION_CV = 0.25
MIN_STD = 1.0


def mean_absolute_error(pred: np.ndarray, actual: np.ndarray) -> float:
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    return float(np.mean(np.abs(p - a)))


def root_mean_squared_error(pred: np.ndarray, actual: np.ndarray) -> float:
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    return float(np.sqrt(np.mean((p - a) ** 2)))


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    """Brier score for a binary over/under event. Proper scoring rule."""
    y = np.asarray(y_true, dtype=float)
    prob = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    return float(np.mean((prob - y) ** 2))


def pinball_loss(actual: np.ndarray, quantile: float, predicted: np.ndarray) -> float:
    """Pinball (quantile) loss. At q=0.5 this is half of MAE."""
    a = np.asarray(actual, dtype=float)
    qhat = np.asarray(predicted, dtype=float)
    err = a - qhat
    return float(np.mean(np.maximum(quantile * err, (quantile - 1.0) * err)))


def gaussian_crps(actual: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Closed-form CRPS under a Gaussian predictive distribution. Proper scoring rule."""
    a = np.asarray(actual, dtype=float)
    mu = np.asarray(mean, dtype=float)
    sigma = np.asarray(std, dtype=float).clip(min=1e-9)
    z = (a - mu) / sigma
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    return float(np.mean(sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))))


def interval_coverage_80(actual: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Coverage of the central 80% Gaussian interval (p10–p90)."""
    a = np.asarray(actual, dtype=float)
    mu = np.asarray(mean, dtype=float)
    sigma = np.asarray(std, dtype=float).clip(min=1e-9)
    lo = mu + sigma * float(norm.ppf(0.10))
    hi = mu + sigma * float(norm.ppf(0.90))
    return float(np.mean((a >= lo) & (a <= hi)))


def fallback_std(location: float) -> float:
    return max(abs(float(location)) * DEFAULT_LOCATION_CV, MIN_STD)


def implied_sigma(
    *,
    line: float | None,
    implied_mean: float | None,
    implied_p_over: float | None,
) -> float:
    """Gaussian sigma implied by (mean, line, P(over)), else a CV fallback."""
    mu = implied_mean if implied_mean is not None else line
    if mu is None:
        return MIN_STD
    if (
        line is not None
        and implied_p_over is not None
        and 1e-6 < float(implied_p_over) < 1.0 - 1e-6
        and abs(float(implied_p_over) - 0.5) > 1e-3
    ):
        z = float(norm.ppf(1.0 - float(implied_p_over)))
        if abs(z) > 1e-6:
            return max(abs((float(line) - float(mu)) / z), 1e-6)
    return fallback_std(float(mu))


def gaussian_p_over(mean: float, std: float, line: float) -> float:
    return float(1.0 - norm.cdf(float(line), loc=float(mean), scale=max(float(std), 1e-9)))
