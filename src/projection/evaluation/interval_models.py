"""Conditional residual interval models."""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from src.projection.backtest import INTERVAL_QUANTILES as DEFAULT_QUANTILES
from src.projection.contracts import INTERVAL_MODELS_DIR

INTERVAL_FEATURE_COLS = [
    "pred",
    "depth_tier",
    "experience_bucket",
    "volume_bucket",
]


def _experience_bucket(games: pd.Series) -> pd.Series:
    g = pd.to_numeric(games, errors="coerce").fillna(0)
    return pd.cut(g, bins=[-1, 3, 8, 16, 100], labels=[0, 1, 2, 3]).astype(float)


def _volume_bucket(pred: pd.Series) -> pd.Series:
    p = pd.to_numeric(pred, errors="coerce").fillna(0)
    return pd.qcut(p.rank(method="first"), 5, labels=False, duplicates="drop").astype(float)


def _col(frame: pd.DataFrame, name: str, default=0.0) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def prepare_interval_features(residuals: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for conditional residual quantile models."""
    frame = residuals.copy()
    frame["depth_tier"] = _col(frame, "depth_tier", 2.0).fillna(2.0)
    frame["experience_bucket"] = _experience_bucket(_col(frame, "games_played", 0.0))
    frame["volume_bucket"] = _volume_bucket(frame["pred"])
    for col in INTERVAL_FEATURE_COLS:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame


def fit_conditional_intervals(
    residuals: pd.DataFrame,
    *,
    quantiles: tuple[float, float] = DEFAULT_QUANTILES,
    out_dir: str | Path | None = None,
) -> dict:
    """Fit per-(position, stat) quantile regressors on cross-fitted residuals."""
    out_dir = Path(out_dir or INTERVAL_MODELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = prepare_interval_features(residuals)
    manifest = {"quantiles": list(quantiles), "cells": {}}
    for (position, stat), grp in frame.groupby(["position", "stat"], observed=True):
        if len(grp) < 30:
            continue
        x = grp[INTERVAL_FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y = pd.to_numeric(grp["resid"], errors="coerce").fillna(0.0)
        cell_key = f"{position}:{stat}"
        models = {}
        for q in quantiles:
            model = QuantileRegressor(quantile=q, alpha=0.1, solver="highs")
            model.fit(x, y)
            path = out_dir / f"{position}_{stat}_q{int(q * 100)}.joblib"
            joblib.dump(model, path)
            models[str(q)] = path.name
        manifest["cells"][cell_key] = {
            "n_train": int(len(grp)),
            "models": models,
            "features": INTERVAL_FEATURE_COLS,
        }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def predict_interval_residuals(
    frame: pd.DataFrame,
    *,
    manifest: dict | None = None,
    models_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Predict conditional residual low/high for each row."""
    models_dir = Path(models_dir or INTERVAL_MODELS_DIR)
    manifest_path = models_dir / "manifest.json"
    if manifest is None:
        if not manifest_path.exists():
            return pd.DataFrame(columns=["position", "stat", "resid_low", "resid_high", "low_n_flag"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quantiles = tuple(float(q) for q in manifest.get("quantiles", DEFAULT_QUANTILES))
    prepared = prepare_interval_features(frame)
    out = frame.copy()
    out["resid_low"] = np.nan
    out["resid_high"] = np.nan
    out["low_n_flag"] = True
    for (position, stat), idx in out.groupby(["position", "stat"], observed=True).groups.items():
        cell = manifest.get("cells", {}).get(f"{position}:{stat}")
        if not cell:
            continue
        x = prepared.loc[idx, INTERVAL_FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        lo_model = joblib.load(models_dir / cell["models"][str(quantiles[0])])
        hi_model = joblib.load(models_dir / cell["models"][str(quantiles[1])])
        out.loc[idx, "resid_low"] = lo_model.predict(x)
        out.loc[idx, "resid_high"] = hi_model.predict(x)
        out.loc[idx, "low_n_flag"] = False
    return out[["position", "stat", "resid_low", "resid_high", "low_n_flag"]]
