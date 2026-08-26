"""Calibration metrics for probabilistic and interval forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pinball_loss(actual: np.ndarray, quantile: float, predicted: np.ndarray) -> float:
    """Pinball (quantile) loss at a single quantile level."""
    err = actual - predicted
    return float(np.mean(np.maximum(quantile * err, (quantile - 1.0) * err)))


def crps_sample(actual: np.ndarray, samples: np.ndarray) -> float:
    """CRPS from empirical sample draws (rows=observations, cols=draws)."""
    actual = np.asarray(actual, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    term1 = np.mean(np.abs(samples - actual[:, None]), axis=1)
    term2 = 0.5 * np.mean(np.abs(samples[:, :, None] - samples[:, None, :]), axis=(1, 2))
    return float(np.mean(term1 - term2))


def crps_gaussian(actual: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """CRPS under Gaussian predictive distribution (closed form)."""
    from scipy.stats import norm

    actual = np.asarray(actual, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float).clip(min=1e-9)
    z = (actual - mean) / std
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    return float(np.mean(std * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))))


def coverage(actual: pd.Series, low: pd.Series, high: pd.Series) -> float:
    a = pd.to_numeric(actual, errors="coerce")
    lo = pd.to_numeric(low, errors="coerce")
    hi = pd.to_numeric(high, errors="coerce")
    mask = a.notna() & lo.notna() & hi.notna()
    if not mask.any():
        return float("nan")
    return float(a[mask].between(lo[mask], hi[mask]).mean())


def coverage_by_group(
    frame: pd.DataFrame,
    *,
    actual_col: str = "actual",
    low_col: str = "pred_low",
    high_col: str = "pred_high",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Coverage fraction by optional grouping columns."""
    group_cols = group_cols or []
    rows = []
    groups = [frame.groupby(group_cols, observed=True)] if group_cols else [(None, frame)]
    if group_cols:
        groups = list(frame.groupby(group_cols, observed=True))
    else:
        groups = [((), frame)]
    for keys, grp in groups:
        row = {col: keys[i] if isinstance(keys, tuple) else keys for i, col in enumerate(group_cols)}
        row["n"] = len(grp)
        row["coverage"] = coverage(grp[actual_col], grp[low_col], grp[high_col])
        rows.append(row)
    return pd.DataFrame(rows)


def reliability_table(
    actual: pd.Series,
    predicted: pd.Series,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Decile reliability: mean predicted vs mean actual per bin."""
    df = pd.DataFrame({"actual": pd.to_numeric(actual, errors="coerce"),
                       "predicted": pd.to_numeric(predicted, errors="coerce")}).dropna()
    if df.empty:
        return pd.DataFrame()
    df["bin"] = pd.qcut(df["predicted"], n_bins, duplicates="drop").astype(str)
    return (
        df.groupby("bin", observed=True)
        .agg(n=("actual", "size"), mean_predicted=("predicted", "mean"), mean_actual=("actual", "mean"))
        .reset_index()
    )


def summarize_interval_calibration(
    residuals: pd.DataFrame,
    *,
    quantiles: tuple[float, float] = (0.10, 0.90),
    group_cols: list[str] | None = None,
) -> dict:
    """Summarize pooled and forward-fold interval calibration."""
    if residuals.empty:
        return {"n": 0}
    lo_q, hi_q = quantiles
    target = hi_q - lo_q
    group_cols = group_cols or ["position", "stat"]
    rows = []
    for keys, grp in residuals.groupby(group_cols, observed=True):
        lo, hi = np.quantile(grp["resid"], quantiles)
        covered = grp["actual"].between(grp["pred"] + lo, grp["pred"] + hi)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "n": len(grp),
            "resid_low": float(lo),
            "resid_high": float(hi),
            "coverage": float(covered.mean()),
            "target_coverage": target,
            "coverage_gap": float(covered.mean() - target),
            "crps_gaussian": crps_gaussian(
                grp["actual"].to_numpy(),
                grp["pred"].to_numpy(),
                grp["resid"].std() if len(grp) > 1 else np.full(len(grp), 1.0),
            ),
        })
        rows.append(row)
    summary = pd.DataFrame(rows)
    return {
        "n": int(len(residuals)),
        "mean_coverage": float(summary["coverage"].mean()) if not summary.empty else float("nan"),
        "mean_coverage_gap": float(summary["coverage_gap"].mean()) if not summary.empty else float("nan"),
        "mean_crps_gaussian": float(summary["crps_gaussian"].mean()) if not summary.empty else float("nan"),
        "by_group": summary.to_dict("records"),
    }
