"""Dispersion decomposition diagnostics for weekly-v2 under-dispersion analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.projection.weekly.evaluate.metrics import dispersion_ratio


def _segment_dispersion(
    pred: np.ndarray,
    actual: np.ndarray,
) -> dict[str, float | None]:
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() < 10:
        return {"n": int(mask.sum()), "dispersion_ratio": None, "pred_sd": None, "actual_sd": None}
    p = pred[mask]
    a = actual[mask]
    return {
        "n": int(mask.sum()),
        "dispersion_ratio": dispersion_ratio(p, a),
        "pred_sd": float(np.std(p)),
        "actual_sd": float(np.std(a)),
        "pred_mean": float(np.mean(p)),
        "actual_mean": float(np.mean(a)),
        "zero_pred_frac": float(np.mean(p <= 1e-6)),
        "zero_actual_frac": float(np.mean(a <= 1e-6)),
    }


def _depth_band(df: pl.DataFrame) -> pl.Series:
    if "depth_rank" not in df.columns:
        return pl.lit("unknown")
    depth = pl.col("depth_rank").cast(pl.Float64)
    return (
        pl.when(depth.is_null())
        .then(pl.lit("unlisted"))
        .when(depth.round() == 1)
        .then(pl.lit("starter"))
        .when(depth.round() <= 3)
        .then(pl.lit("rotational"))
        .otherwise(pl.lit("deep"))
    )


def decompose_dispersion(
    oof: pl.DataFrame,
    *,
    pred_col: str = "projected_fantasy_points",
    actual_col: str = "actual_fantasy_points",
) -> dict[str, Any]:
    """Decompose dispersion by season, position, depth band, and availability."""
    required = {pred_col, actual_col, "season", "position"}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"OOF missing columns: {sorted(missing)}")

    work = oof.with_columns(_depth_band(oof).alias("depth_band"))
    if "play_prob" in work.columns:
        work = work.with_columns(
            pl.when(pl.col("play_prob").fill_null(1.0) < 0.5)
            .then(pl.lit("availability_limited"))
            .otherwise(pl.lit("healthy_active"))
            .alias("availability_band")
        )
    else:
        work = work.with_columns(pl.lit("unknown").alias("availability_band"))
    if "is_rookie" in work.columns:
        work = work.with_columns(
            pl.when(pl.col("is_rookie").fill_null(False))
            .then(pl.lit("rookie"))
            .otherwise(pl.lit("veteran"))
            .alias("experience_band")
        )
    else:
        work = work.with_columns(pl.lit("unknown").alias("experience_band"))

    overall = _segment_dispersion(
        work[pred_col].to_numpy(),
        work[actual_col].to_numpy(),
    )
    by_season: dict[str, Any] = {}
    for season in sorted(work["season"].unique().to_list()):
        sub = work.filter(pl.col("season") == season)
        by_season[str(season)] = {
            "overall": _segment_dispersion(
                sub[pred_col].to_numpy(), sub[actual_col].to_numpy()
            ),
            "by_position": {},
            "by_depth_band": {},
            "by_availability": {},
            "by_experience": {},
        }
        for pos in sub["position"].unique().to_list():
            pos_sub = sub.filter(pl.col("position") == pos)
            by_season[str(season)]["by_position"][str(pos)] = _segment_dispersion(
                pos_sub[pred_col].to_numpy(), pos_sub[actual_col].to_numpy()
            )
        for band_col, key in (
            ("depth_band", "by_depth_band"),
            ("availability_band", "by_availability"),
            ("experience_band", "by_experience"),
        ):
            for band in sub[band_col].unique().to_list():
                band_sub = sub.filter(pl.col(band_col) == band)
                by_season[str(season)][key][str(band)] = _segment_dispersion(
                    band_sub[pred_col].to_numpy(), band_sub[actual_col].to_numpy()
                )

    component_cols = [
        c
        for c in (
            "pred_target_share",
            "pred_carry_share",
            "pred_dropback_share",
            "targets",
            "carries",
            "attempts",
            "receptions",
            "receiving_yards",
            "rushing_yards",
            "passing_yards",
        )
        if c in work.columns
    ]
    component_stats: dict[str, Any] = {}
    for col in component_cols:
        actual_col_name = col.replace("pred_", "") if col.startswith("pred_") else col
        if actual_col_name in work.columns:
            component_stats[col] = _segment_dispersion(
                work[col].to_numpy(), work[actual_col_name].to_numpy()
            )

    return {
        "overall": overall,
        "by_season": by_season,
        "component_dispersion": component_stats,
        "n_rows": work.height,
    }


def write_dispersion_diagnostics(
    oof: pl.DataFrame,
    output_path: Path,
    *,
    pred_col: str = "projected_fantasy_points",
    actual_col: str = "actual_fantasy_points",
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    payload = decompose_dispersion(oof, pred_col=pred_col, actual_col=actual_col)
    if extra_metadata:
        payload["metadata"] = extra_metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_path
