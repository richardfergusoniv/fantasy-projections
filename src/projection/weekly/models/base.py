"""Shared sklearn helpers for regression models."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _clip_prediction(target: str, pred: np.ndarray, mean: float) -> np.ndarray:
    """Keep model outputs in football-realistic ranges."""
    if "share" in target:
        return np.clip(pred, 0.0, 1.0)
    if target == "catch_rate" or target.endswith("catch_rate"):
        return np.clip(pred, 0.30, 0.95)
    if "td_rate" in target or target == "int_rate" or target.endswith("int_rate"):
        return np.clip(pred, 0.0, 0.25)
    if target.endswith("ypa") or target == "ypa":
        return np.clip(pred, 3.0, 12.0)
    if target.endswith("ypc") or target == "ypc":
        return np.clip(pred, 1.5, 7.0)
    if target.endswith("ypr") or target == "ypr":
        return np.clip(pred, 3.0, 20.0)
    if "fp" in target:
        return np.clip(pred, 0.0, 40.0)
    if "pass_attempts" in target:
        return np.clip(pred, 5.0, 55.0)
    if "rush_attempts" in target:
        return np.clip(pred, 5.0, 45.0)
    if "tds" in target:
        return np.clip(pred, 0.0, 5.0)
    if np.isfinite(mean) and abs(mean) > 1e-6:
        return np.clip(pred, mean - 5 * abs(mean) - 5, mean + 5 * abs(mean) + 5)
    return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class MultiTargetModel:
    """Dictionary of sklearn pipelines, one per target column."""

    targets: list[str]
    feature_cols: list[str]
    models: dict[str, Pipeline] = field(default_factory=dict)
    positional_means: dict[str, float] = field(default_factory=dict)
    # Weight on raw prediction vs positional mean. Holdout optimum is ≥ 1.0;
    # default to 1.0 (no shrink). Legacy 0.85 shrink was never fitted.
    shrink_weight: float = 1.0

    def fit(
        self,
        X: np.ndarray,
        y: pl.DataFrame | np.ndarray,
        *,
        model_type: str = "hgb",
        sample_weight: np.ndarray | None = None,
    ) -> MultiTargetModel:
        if isinstance(y, pl.DataFrame):
            y_df = y
        else:
            y_df = pl.DataFrame({t: y[:, i] for i, t in enumerate(self.targets)})

        for target in self.targets:
            y_vec = y_df[target].to_numpy().astype(float)
            mask = np.isfinite(y_vec)
            X_fit = X[mask]
            y_fit = y_vec[mask]
            weight_fit = (
                np.asarray(sample_weight, dtype=float)[mask]
                if sample_weight is not None
                else None
            )
            self.positional_means[target] = float(np.nanmean(y_fit)) if len(y_fit) else 0.0
            pipe = _make_pipeline(model_type)
            if len(y_fit) < 10:
                pipe = _make_pipeline("ridge")
            fit_params = {"model__sample_weight": weight_fit} if weight_fit is not None else {}
            pipe.fit(X_fit, y_fit, **fit_params)
            self.models[target] = pipe
        return self

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        w = float(self.shrink_weight)
        w = min(1.0, max(0.0, w))
        for target, model in self.models.items():
            pred = np.asarray(model.predict(X), dtype=float)
            mean = self.positional_means.get(target, 0.0)
            pred = np.nan_to_num(pred, nan=mean, posinf=mean, neginf=mean)
            # Ridge can explode on poisoned lag features; snap extremes back to the mean
            if np.isfinite(mean):
                max_dev = 5.0 * (abs(mean) + 1.0)
                pred = np.where(np.abs(pred - mean) > max_dev, mean, pred)
            if w < 1.0 - 1e-12:
                pred = w * pred + (1.0 - w) * mean
            out[target] = _clip_prediction(target, pred, mean)
        return out

    def predict_frame(self, df: pl.DataFrame, prefix: str = "pred_") -> pl.DataFrame:
        X = dataframe_to_matrix(df, self.feature_cols)
        preds = self.predict(X)
        return df.with_columns([pl.Series(f"{prefix}{k}", v) for k, v in preds.items()])


def _make_pipeline(model_type: str) -> Pipeline:
    if model_type == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=5.0)),
            ]
        )
    if model_type == "gb":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    GradientBoostingRegressor(
                        random_state=42,
                        max_depth=3,
                        n_estimators=150,
                        learning_rate=0.05,
                    ),
                ),
            ]
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=42,
                    max_depth=4,
                    learning_rate=0.06,
                    max_iter=200,
                ),
            ),
        ]
    )


def dataframe_to_matrix(df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    cols = []
    for c in feature_cols:
        if c in df.columns:
            cols.append(df[c].cast(pl.Float64, strict=False).fill_null(np.nan).to_numpy())
        else:
            cols.append(np.full(df.height, np.nan))
    if not cols:
        return np.zeros((df.height, 0))
    return np.column_stack(cols)


def available_features(df: pl.DataFrame, candidates: list[str]) -> list[str]:
    """Return candidate columns that exist and are not entirely null."""
    out = []
    for c in candidates:
        if c not in df.columns:
            continue
        if df[c].null_count() == df.height:
            continue
        out.append(c)
    return out
