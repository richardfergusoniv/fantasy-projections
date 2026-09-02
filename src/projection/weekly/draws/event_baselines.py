"""Training-only deployable baselines for discrete event models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.projection.weekly.draws.contracts_v2 import event_denominator_mask
from src.projection.weekly.draws.contracts import np_clip_prob
from src.projection.weekly.draws.event_models import (
    EventName,
    _available_features,
    _design_matrix,
    DEFAULT_FEATURES,
)

BaselineKind = Literal["constant_prevalence", "depth_status_logistic", "play_prob_heuristic"]


@dataclass
class EventBaselineCell:
    event: EventName
    position: str
    kind: BaselineKind
    train_seasons: list[int]
    n_train: int
    prevalence: float
    params: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EventBaselineBundle:
    schema_version: int = 1
    cells: dict[str, EventBaselineCell] = field(default_factory=dict)

    def predict(
        self,
        event: EventName,
        position: str,
        frame: pl.DataFrame,
    ) -> np.ndarray:
        key = f"{event}:{position}"
        if key not in self.cells:
            raise KeyError(f"no baseline for {key}")
        cell = self.cells[key]
        if cell.kind == "constant_prevalence":
            return np.full(frame.height, np_clip_prob(cell.prevalence), dtype=float)
        if cell.kind == "play_prob_heuristic":
            if "play_prob" not in frame.columns:
                return np.full(frame.height, np_clip_prob(cell.prevalence), dtype=float)
            return np.array(
                [np_clip_prob(float(p)) for p in frame["play_prob"].fill_null(cell.prevalence).to_numpy()],
                dtype=float,
            )
        if cell.kind == "depth_status_logistic":
            pipe = cell.params.get("_pipeline")
            feats = cell.params.get("feature_cols", [])
            if pipe is None or not feats:
                return np.full(frame.height, np_clip_prob(cell.prevalence), dtype=float)
            X = _design_matrix(frame, feats)
            raw = pipe.predict_proba(X)[:, 1]
            return np.array([np_clip_prob(float(p)) for p in raw], dtype=float)
        return np.full(frame.height, np_clip_prob(cell.prevalence), dtype=float)


def _cell_hash(cell: EventBaselineCell) -> str:
    payload = cell.to_dict()
    payload.pop("content_hash", None)
    payload.get("params", {}).pop("_pipeline", None)
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def fit_training_baselines(
    train: pl.DataFrame,
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
) -> EventBaselineBundle:
    """Fit baselines using training-fold rows only (never test prevalence)."""
    bundle = EventBaselineBundle()
    train_seasons = sorted(int(s) for s in train["season"].unique().to_list())
    for event in events:
        for position in positions:
            mask = event_denominator_mask(event, train)
            sub = train.filter(mask).filter(
                (pl.col("position") == position) & pl.col(event).is_not_null()
            )
            key = f"{event}:{position}"
            if sub.is_empty():
                continue
            y = sub[event].cast(pl.Float64).to_numpy()
            prevalence = float(y.mean()) if y.size else 0.5
            cell = EventBaselineCell(
                event=event,
                position=position,
                kind="constant_prevalence",
                train_seasons=train_seasons,
                n_train=sub.height,
                prevalence=prevalence,
            )
            # Depth/status logistic when enough examples.
            feats = [c for c in _available_features(sub, feature_candidates) if c not in {"play_prob"}]
            n_pos = int(y.sum())
            if (
                feats
                and sub.height >= max(2 * min_positive, 40)
                and min_positive <= n_pos < sub.height
            ):
                X = _design_matrix(sub, feats)
                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(max_iter=500, random_state=random_seed),
                        ),
                    ]
                )
                pipe.fit(X, y)
                cell = EventBaselineCell(
                    event=event,
                    position=position,
                    kind="depth_status_logistic",
                    train_seasons=train_seasons,
                    n_train=sub.height,
                    prevalence=prevalence,
                    params={"feature_cols": feats, "_pipeline": pipe},
                )
            elif event == "active_label" and "play_prob" in sub.columns:
                cell = EventBaselineCell(
                    event=event,
                    position=position,
                    kind="play_prob_heuristic",
                    train_seasons=train_seasons,
                    n_train=sub.height,
                    prevalence=prevalence,
                    params={"note": "replayable live policy without test peek"},
                )
            cell.content_hash = _cell_hash(cell)
            bundle.cells[key] = cell
    return bundle


def save_baseline_bundle(bundle: EventBaselineBundle, path: Any) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": bundle.schema_version,
        "cells": {k: v.to_dict() for k, v in bundle.cells.items()},
    }
    for cell in payload["cells"].values():
        cell.get("params", {}).pop("_pipeline", None)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
