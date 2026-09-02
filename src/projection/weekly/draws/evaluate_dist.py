"""Probabilistic evaluation metrics for weekly joint draws."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def crps_sample(draws: np.ndarray, actual: float) -> float:
    """Sample-based CRPS for a univariate predictive distribution."""
    x = np.sort(np.asarray(draws, dtype=float).ravel())
    n = x.size
    if n == 0:
        return float("nan")
    y = float(actual)
    # CRPS = E|X-y| - 0.5 E|X-X'|
    term1 = float(np.mean(np.abs(x - y)))
    # Efficient second term for sorted samples
    i = np.arange(1, n + 1)
    term2 = float(np.sum((2 * i - n - 1) * x) / (n * n))
    return term1 - term2


def weighted_interval_score(
    draws: np.ndarray,
    actual: float,
    *,
    alpha: float = 0.2,
) -> float:
    """WIS for a central (1-alpha) interval from samples."""
    x = np.asarray(draws, dtype=float).ravel()
    if x.size == 0:
        return float("nan")
    y = float(actual)
    lower = float(np.quantile(x, alpha / 2))
    upper = float(np.quantile(x, 1.0 - alpha / 2))
    width = upper - lower
    below = (lower - y) * (y < lower)
    above = (y - upper) * (y > upper)
    return float((alpha / 2) * width + below + above) * (2.0 / alpha)


def interval_coverage(
    draws: np.ndarray,
    actual: float,
    *,
    q_lo: float = 0.1,
    q_hi: float = 0.9,
) -> dict[str, float]:
    x = np.asarray(draws, dtype=float).ravel()
    lo = float(np.quantile(x, q_lo))
    hi = float(np.quantile(x, q_hi))
    y = float(actual)
    return {
        "p_lo": lo,
        "p_hi": hi,
        "width": hi - lo,
        "covered": float(lo <= y <= hi),
    }


def pit_value(draws: np.ndarray, actual: float) -> float:
    x = np.asarray(draws, dtype=float).ravel()
    if x.size == 0:
        return float("nan")
    return float(np.mean(x <= float(actual)))


def mean_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    err = p - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    if p.size < 2 or np.std(p) < 1e-12 or np.std(a) < 1e-12:
        rank_corr = float("nan")
    else:
        rank_corr = float(np.corrcoef(p.argsort().argsort(), a.argsort().argsort())[0, 1])
    return {"mae": mae, "rmse": rmse, "rank_corr": rank_corr}


def zero_mass_calibration(
    draws_by_player: Sequence[np.ndarray],
    actuals: Sequence[float],
    *,
    eps: float = 1e-9,
) -> dict[str, float]:
    pred_zero = [float(np.mean(np.asarray(d) <= eps)) for d in draws_by_player]
    act_zero = [1.0 if float(a) <= eps else 0.0 for a in actuals]
    return {
        "mean_pred_zero_mass": float(np.mean(pred_zero)) if pred_zero else float("nan"),
        "actual_zero_rate": float(np.mean(act_zero)) if act_zero else float("nan"),
        "abs_gap": float(abs(np.mean(pred_zero) - np.mean(act_zero))) if pred_zero else float("nan"),
    }


def correlation_diagnostic(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize_distributional_eval(
    *,
    player_draws: dict[str, np.ndarray],
    actuals: dict[str, float],
    point_means: dict[str, float] | None = None,
) -> dict[str, Any]:
    crps_vals = []
    wis_vals = []
    cover = []
    widths = []
    pits = []
    aligned_pred = []
    aligned_act = []
    draw_arrays = []
    act_list = []
    for pid, y in actuals.items():
        if pid not in player_draws:
            continue
        d = np.asarray(player_draws[pid], dtype=float)
        crps_vals.append(crps_sample(d, y))
        wis_vals.append(weighted_interval_score(d, y))
        cov = interval_coverage(d, y)
        cover.append(cov["covered"])
        widths.append(cov["width"])
        pits.append(pit_value(d, y))
        mean_p = float(point_means[pid]) if point_means and pid in point_means else float(d.mean())
        aligned_pred.append(mean_p)
        aligned_act.append(float(y))
        draw_arrays.append(d)
        act_list.append(float(y))
    return {
        "n": len(crps_vals),
        "crps_mean": float(np.nanmean(crps_vals)) if crps_vals else float("nan"),
        "wis_mean": float(np.nanmean(wis_vals)) if wis_vals else float("nan"),
        "coverage_p10_p90": float(np.mean(cover)) if cover else float("nan"),
        "mean_width_p10_p90": float(np.mean(widths)) if widths else float("nan"),
        "pit_mean": float(np.mean(pits)) if pits else float("nan"),
        "point_metrics": mean_metrics(np.array(aligned_pred), np.array(aligned_act)),
        "zero_mass": zero_mass_calibration(draw_arrays, act_list),
        "note": (
            "Deterministic-mean dispersion must be reported separately from "
            "distributional calibration; sample variance does not pass the point gate."
        ),
    }
