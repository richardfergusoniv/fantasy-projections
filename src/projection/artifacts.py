"""Saved model / residual / correction artifact loaders.

Leaf-ish module — no predict import.
"""
from __future__ import annotations

import os
import json

import joblib
import pandas as pd

from src.projection.contracts import (
    CORRECTIONS_PATH,
    CONCENTRATION_PATH,
    INTERVAL_MODELS_DIR,
    INTERVAL_RESIDUALS_PATH,
    MODELS_DIR,
    RECONCILE_CALIBRATION_PATH,
)
from src.projection.features import TARGET_STATS


def load_availability_models():
    """Per-position games-played models (Phase 11). Returns {} - a real
    "no availability estimate is produced" state, not an error - when the
    files predate this feature, so an older models/ directory still
    predicts rather than failing."""
    out = {}
    for position in TARGET_STATS:
        path = os.path.join(MODELS_DIR, f"{position}_games.joblib")
        if os.path.exists(path):
            out[position] = joblib.load(path)
    return out


def load_models():
    models = {}
    for position, stats in TARGET_STATS.items():
        for stat in stats:
            path = os.path.join(MODELS_DIR, f"{position}_{stat}.joblib")
            models[(position, stat)] = joblib.load(path)
    # Joint/multi-output Phase A team-total model (train.py) - same
    # ("TEAM", "passing_yards") key backtest.py's own rows use.
    models[("TEAM", "passing_yards")] = joblib.load(os.path.join(MODELS_DIR, "team_passing_yards.joblib"))
    models[("TEAM", "pass_attempts")] = joblib.load(os.path.join(MODELS_DIR, "team_pass_attempts.joblib"))
    models[("TEAM", "carries")] = joblib.load(os.path.join(MODELS_DIR, "team_carries.joblib"))
    models[("TEAM", "rushing_yards")] = joblib.load(os.path.join(MODELS_DIR, "team_rushing_yards.joblib"))
    return models


def load_interval_residuals():
    if not os.path.exists(INTERVAL_RESIDUALS_PATH):
        raise FileNotFoundError(
            f"{INTERVAL_RESIDUALS_PATH} not found - run `python -m src.projection.backtest` "
            "once (after train.py) to build it before calling project_season."
        )
    return pd.read_csv(INTERVAL_RESIDUALS_PATH)


def load_corrections():
    """Elite-shrinkage correction parameters fit by train.py (see
    corrections.py). Returns {} - a real "no correction is applied" state,
    not an error - when the file predates this feature, so an older
    models/ directory still predicts rather than failing."""
    if not os.path.exists(CORRECTIONS_PATH):
        return {}
    return joblib.load(CORRECTIONS_PATH)


def load_concentration_calibration():
    """Load the promoted share-concentration calibration.

    Missing artifacts are an explicit identity transform so older model
    directories remain runnable; publishing records that unfitted state in
    every row and in the run manifest.
    """
    if not os.path.exists(CONCENTRATION_PATH):
        return {"version": "unfitted_identity", "cells": {}}
    with open(CONCENTRATION_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def load_reconcile_calibration():
    """Learned team-volume reconciliation alpha; falls back to contracts default."""
    from src.projection.contracts import TEAM_RECONCILE_ALPHA

    if not os.path.exists(RECONCILE_CALIBRATION_PATH):
        return {"default_alpha": TEAM_RECONCILE_ALPHA, "version": "unfitted_default"}
    with open(RECONCILE_CALIBRATION_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def reconcile_alpha_for(position: str, stat: str, calibration: dict | None = None) -> float:
    """Return alpha for a (position, stat) cell, else global default."""
    from src.projection.contracts import TEAM_RECONCILE_ALPHA

    calibration = calibration or load_reconcile_calibration()
    cells = calibration.get("cells", {})
    cell = cells.get(f"{position}:{stat}")
    if cell and "alpha" in cell:
        return float(cell["alpha"])
    return float(calibration.get("default_alpha", TEAM_RECONCILE_ALPHA))


def load_interval_model_manifest():
    path = os.path.join(INTERVAL_MODELS_DIR, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
