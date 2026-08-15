"""Saved model / residual / correction artifact loaders.

Leaf-ish module — no predict import.
"""
from __future__ import annotations

import os

import joblib
import pandas as pd

from src.projection.contracts import (
    CORRECTIONS_PATH,
    INTERVAL_RESIDUALS_PATH,
    MODELS_DIR,
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
