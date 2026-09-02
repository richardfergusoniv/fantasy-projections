"""Calibrated discrete-event models for active / participation / positive usage."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.projection.weekly.draws.contracts import np_clip_prob
from src.projection.weekly.draws.contracts_v2 import event_denominator_mask

EventName = Literal["active_label", "participated_label", "positive_usage_label"]

DEFAULT_FEATURES = (
    "play_prob",
    "is_out",
    "is_doubtful",
    "is_questionable",
    "depth_rank",
    "age",
    "is_rookie",
    "games_played_prior",
    "target_share_l5",
    "carry_share_l5",
    "snap_share_l5",
)


@dataclass
class EventModelSpec:
    event: EventName
    position: str
    feature_cols: list[str]
    n_train: int
    n_positive: int
    random_seed: int = 42
    notes: str = ""


@dataclass
class EventModelBundle:
    """Per-position logistic baselines for discrete mixture events."""

    schema_version: int = 1
    models: dict[str, Any] = field(default_factory=dict)  # key: f"{event}:{position}"
    specs: list[EventModelSpec] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def predict_proba(
        self,
        event: EventName,
        position: str,
        frame: pl.DataFrame,
    ) -> np.ndarray:
        key = f"{event}:{position}"
        if key not in self.models:
            raise KeyError(f"no event model for {key}")
        pipe, spec = self.models[key]
        X = _design_matrix(frame, spec.feature_cols)
        raw = pipe.predict_proba(X)[:, 1]
        return np.array([np_clip_prob(float(p)) for p in raw], dtype=float)


def _design_matrix(frame: pl.DataFrame, cols: list[str]) -> np.ndarray:
    rows = []
    for col in cols:
        if col in frame.columns:
            series = frame[col]
            if series.dtype == pl.Boolean:
                rows.append(series.fill_null(False).cast(pl.Float64).to_numpy())
            else:
                rows.append(series.cast(pl.Float64, strict=False).fill_null(0.0).to_numpy())
        else:
            rows.append(np.zeros(frame.height, dtype=float))
    if not rows:
        return np.zeros((frame.height, 1), dtype=float)
    return np.column_stack(rows)


def _available_features(frame: pl.DataFrame, candidates: tuple[str, ...]) -> list[str]:
    return [c for c in candidates if c in frame.columns]


def fit_event_models(
    panel: pl.DataFrame,
    *,
    events: tuple[EventName, ...] = (
        "active_label",
        "participated_label",
        "positive_usage_label",
    ),
    positions: tuple[str, ...] = ("QB", "RB", "WR", "TE"),
    feature_candidates: tuple[str, ...] = DEFAULT_FEATURES,
    min_positive: int = 20,
    random_seed: int = 42,
    class_weight: str | None = None,
) -> EventModelBundle:
    """Fit simple calibrated logistic baselines per event×position.

    Rows without a scheduled game are excluded (bye is not a failed event).
    Current/live-only statuses must not appear in historical fold training;
    callers must pass leakage-safe panels.
    """
    if "has_scheduled_game" not in panel.columns:
        raise ValueError("mixture panel required (has_scheduled_game)")
    bundle = EventModelBundle(
        config={
            "min_positive": min_positive,
            "random_seed": random_seed,
            "feature_candidates": list(feature_candidates),
            "class_weight": class_weight,
        }
    )
    for event in events:
        if event not in panel.columns:
            continue
        for position in positions:
            sub = panel.filter(event_denominator_mask(event, panel)).filter(
                (pl.col("position") == position) & pl.col(event).is_not_null()
            )
            y = sub[event].cast(pl.Int8).to_numpy()
            n_pos = int(y.sum())
            if sub.height < max(2 * min_positive, 40) or n_pos < min_positive or n_pos >= sub.height:
                # Fail cleanly: record skipped cell rather than a degenerate classifier.
                bundle.specs.append(
                    EventModelSpec(
                        event=event,
                        position=position,
                        feature_cols=[],
                        n_train=sub.height,
                        n_positive=n_pos,
                        random_seed=random_seed,
                        notes="skipped: insufficient positive/negative examples",
                    )
                )
                continue
            feats = _available_features(sub, feature_candidates)
            if not feats:
                bundle.specs.append(
                    EventModelSpec(
                        event=event,
                        position=position,
                        feature_cols=[],
                        n_train=sub.height,
                        n_positive=n_pos,
                        random_seed=random_seed,
                        notes="skipped: no features",
                    )
                )
                continue
            X = _design_matrix(sub, feats)
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=500,
                            random_state=random_seed,
                            class_weight=class_weight,
                        ),
                    ),
                ]
            )
            pipe.fit(X, y)
            spec = EventModelSpec(
                event=event,
                position=position,
                feature_cols=feats,
                n_train=sub.height,
                n_positive=n_pos,
                random_seed=random_seed,
            )
            bundle.models[f"{event}:{position}"] = (pipe, spec)
            bundle.specs.append(spec)
    return bundle


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss_safe(y_true: np.ndarray, p: np.ndarray, *, eps: float = 1e-6) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def calibration_bins(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not np.any(mask):
            continue
        bins.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "count": float(mask.sum()),
                "mean_p": float(p[mask].mean()),
                "freq": float(y[mask].mean()),
            }
        )
    return bins


def evaluate_event_predictions(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    baseline_rate: float | None = None,
    baseline_probs: np.ndarray | None = None,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    rate = float(y.mean()) if y.size else 0.0
    if baseline_probs is not None:
        base_p = np.asarray(baseline_probs, dtype=float)
        base = float(base_p.mean()) if base_p.size else rate
    elif baseline_rate is not None:
        base = float(baseline_rate)
        base_p = np.full_like(p, base)
    else:
        raise ValueError(
            "baseline_rate or baseline_probs is required; "
            "test-fold prevalence must not define the deployable baseline"
        )
    return {
        "n": int(y.size),
        "prevalence": rate,
        "brier": brier_score(y, p),
        "log_loss": log_loss_safe(y, p),
        "brier_baseline": brier_score(y, base_p),
        "log_loss_baseline": log_loss_safe(y, base_p),
        "sharpness": float(np.std(p)) if y.size else 0.0,
        "calibration_bins": calibration_bins(y, p),
    }


def save_event_bundle_meta(bundle: EventModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": bundle.schema_version,
        "config": bundle.config,
        "specs": [asdict(s) for s in bundle.specs],
        "fitted_keys": sorted(bundle.models.keys()),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
