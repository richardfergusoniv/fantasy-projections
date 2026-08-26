"""Team environment distributional forecasts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from src.projection.contracts import V3_MODELS_DIR
from src.projection.transitions import TEAM_MODEL_FEATURES, TEAM_TOTAL_LABEL, build_team_transition_pairs

TEAM_ENV_STATS = (
    ("passing_yards", TEAM_TOTAL_LABEL),
    ("pass_attempts", "team_pass_attempts_pg"),
    ("carries", "team_carries_pg"),
    ("rushing_yards", "team_rushing_yards_pg"),
)


def fit_team_environment(feat, train_pairs) -> dict:
    """Fit Ridge mean models and residual dispersion per team volume stat."""
    out_dir = Path(V3_MODELS_DIR) / "team_environment"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"cells": {}}
    for stat_name, label_col in TEAM_ENV_STATS:
        train = build_team_transition_pairs(feat, train_pairs, label_col=label_col)
        if train.empty or label_col not in train.columns:
            continue
        model = RidgeCV(alphas=np.logspace(-2, 3, 20))
        model.fit(train[TEAM_MODEL_FEATURES], train[label_col])
        resid_std = float((train[label_col] - model.predict(train[TEAM_MODEL_FEATURES])).std())
        cell = {
            "label_col": label_col,
            "resid_std": resid_std,
            "dispersion": max(resid_std ** 2, 1e-6),
        }
        import joblib

        path = out_dir / f"team_{stat_name}.joblib"
        joblib.dump(model, path)
        cell["model_path"] = path.name
        manifest["cells"][stat_name] = cell
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def predict_team_environment(frame: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    """Return mean + std columns for each team environment stat."""
    import joblib

    out = frame.copy()
    base_dir = Path(V3_MODELS_DIR) / "team_environment"
    for stat_name, cell in manifest.get("cells", {}).items():
        model = joblib.load(base_dir / cell["model_path"])
        mean = model.predict(out[TEAM_MODEL_FEATURES])
        out[f"team_{stat_name}_mean"] = np.clip(mean, 0, None)
        out[f"team_{stat_name}_std"] = np.sqrt(cell["dispersion"])
    return out
