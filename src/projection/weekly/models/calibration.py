"""Out-of-fold point-dispersion and conformal interval calibration."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

#: Per-position maximum linear expansion slope. WR/TE OOF residuals routinely
#: request more spread than QB/RB; a uniform 1.35 cap leaves calibrated
#: dispersion below policy for those positions while 1.6+ caused rank blow-ups.
POSITION_SLOPE_CAPS: dict[str, float] = {
    "QB": 1.35,
    "RB": 1.40,
    "WR": 1.52,
    "TE": 1.52,
}
DEFAULT_SLOPE_CAP = 1.35


def fit_position_calibration(
    oof: pl.DataFrame,
    *,
    prediction_col: str = "projected_fantasy_points",
    actual_col: str = "actual_fantasy_points",
    alpha: float = 0.20,
    min_samples: int = 50,
) -> dict[str, Any]:
    """Fit position-wise linear dispersion and split-conformal residual bands.

    ``oof`` must contain genuinely out-of-fold predictions.  The fitted slope
    expands under-dispersed projections while residual quantiles provide
    empirical, asymmetric prediction intervals.
    """
    required = {"position", prediction_col, actual_col}
    if not required.issubset(oof.columns):
        raise ValueError(f"calibration rows missing {sorted(required - set(oof.columns))}")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    positions: dict[str, Any] = {}
    for pos in oof["position"].drop_nulls().unique().to_list():
        sub = oof.filter(pl.col("position") == pos).select(
            [prediction_col, actual_col]
        ).drop_nulls()
        x = sub[prediction_col].to_numpy().astype(float)
        y = sub[actual_col].to_numpy().astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < min_samples:
            continue
        pred_sd = float(np.std(x))
        actual_sd = float(np.std(y))
        # OLS slope = corr(x, y) * sd(y) / sd(x), so it remains
        # under-dispersed whenever correlation is below one.  Calibration's
        # job here is to match the observed cross-player spread; use the
        # standard-deviation ratio and preserve the mean with the intercept.
        slope = actual_sd / pred_sd if pred_sd > 1e-9 else 1.0
        cap = POSITION_SLOPE_CAPS.get(str(pos), DEFAULT_SLOPE_CAP)
        slope = float(np.clip(slope, 0.5, cap))
        intercept = float(np.mean(y) - slope * np.mean(x))
        calibrated = intercept + slope * x
        residual = y - calibrated
        lo = float(np.quantile(residual, alpha / 2.0, method="lower"))
        hi = float(np.quantile(residual, 1.0 - alpha / 2.0, method="higher"))
        coverage = float(np.mean((residual >= lo) & (residual <= hi)))
        positions[str(pos)] = {
            "n": int(len(x)),
            "intercept": intercept,
            "slope": slope,
            "residual_low": lo,
            "residual_high": hi,
            "empirical_coverage": coverage,
        }

    seasons = (
        sorted(int(x) for x in oof["season"].drop_nulls().unique().to_list())
        if "season" in oof.columns
        else []
    )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "alpha": alpha,
        "trained_seasons": seasons,
        "max_train_season": max(seasons) if seasons else None,
        "positions": positions,
    }


def apply_position_calibration(
    projections: pl.DataFrame,
    calibration: dict[str, Any],
    *,
    point_col: str = "fantasy_points",
) -> pl.DataFrame:
    """Apply fitted point and interval calibration to a projection frame."""
    if projections.is_empty() or "position" not in projections.columns:
        return projections
    params = calibration.get("positions") or {}
    raw = pl.col(point_col).cast(pl.Float64)
    point_expr = raw
    low_offset = pl.lit(-5.0)
    high_offset = pl.lit(8.0)
    for pos, rec in params.items():
        calibrated = float(rec["intercept"]) + float(rec["slope"]) * raw
        point_expr = pl.when(pl.col("position") == pos).then(calibrated).otherwise(point_expr)
        low_offset = (
            pl.when(pl.col("position") == pos)
            .then(pl.lit(float(rec["residual_low"])))
            .otherwise(low_offset)
        )
        high_offset = (
            pl.when(pl.col("position") == pos)
            .then(pl.lit(float(rec["residual_high"])))
            .otherwise(high_offset)
        )
    out = projections.with_columns(raw.alias("fantasy_points_raw"))
    out = out.with_columns(point_expr.clip(0.0, None).alias(point_col))
    return out.with_columns(
        [
            (pl.col(point_col) + low_offset).clip(0.0, None).alias("floor"),
            (pl.col(point_col) + high_offset).clip(0.0, None).alias("ceiling"),
        ]
    )


def save_calibration(calibration: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    return path


def load_calibration_for_season(path: Path, *, target_season: int) -> dict[str, Any] | None:
    """Load only a calibration artifact trained strictly before the target."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    max_year = payload.get("max_train_season")
    if max_year is None or int(max_year) >= int(target_season):
        return None
    return payload
