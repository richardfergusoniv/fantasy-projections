"""Accuracy-first ensemble helpers for the 2026 draft board.

The replacement gate asks whether v3 should own the point engine.  This module
answers a different question: whether a bounded contribution from v3 and/or
preseason ADP improves the already-shipped v1/v2 draft ensemble.

All functions here are dataframe transforms.  The orchestration script owns
filesystem access and the strict chronological split.
"""
from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.projection.evaluation.v3_means_score import score_predictions


POSITIONS = ("QB", "RB", "WR", "TE")
TOP_ADP = 120.0
WEIGHT_STEP = 0.05


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_consensus_snapshot(path: str | Path, *, expected_season: int) -> tuple[pd.DataFrame, dict]:
    """Load one frozen consensus snapshot and fail on temporal ambiguity."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = dict(payload.get("meta") or {})
    season = int(meta.get("season", -1))
    if season != int(expected_season):
        raise ValueError(f"Consensus snapshot season {season} != target {expected_season}")
    as_of_source = "meta.as_of"
    as_of = str(meta.get("as_of") or "")
    if not as_of:
        as_of = str((meta.get("adp") or {}).get("end_date") or "")
        as_of_source = "meta.adp.end_date"
    # These are preseason snapshots.  September 7 safely covers the latest
    # historical file in this project while remaining before the first full
    # NFL Sunday.  Missing provenance fails closed.
    if not as_of or as_of > f"{expected_season}-09-07":
        raise ValueError(f"Consensus snapshot is not demonstrably preseason: {as_of!r}")
    meta["as_of"] = as_of
    meta["as_of_source"] = as_of_source
    rows = pd.DataFrame(payload.get("rows") or [])
    if rows.empty:
        return rows, meta
    rows["player_id"] = rows["player_id"].astype(str)
    rows["position"] = rows["position"].astype(str)
    rows["adp"] = pd.to_numeric(rows.get("adp"), errors="coerce")
    rows["ecr"] = pd.to_numeric(rows.get("ecr"), errors="coerce")
    return rows, meta


def fit_market_curves(history: pd.DataFrame) -> dict[str, IsotonicRegression]:
    """Fit decreasing ADP -> half-PPR points curves by position."""
    curves: dict[str, IsotonicRegression] = {}
    for position in POSITIONS:
        sub = history[history["position"].eq(position)].dropna(
            subset=["adp", "actual_points"]
        )
        if len(sub) < 8:
            raise ValueError(f"Need at least 8 market calibration rows for {position}; got {len(sub)}")
        # Average duplicate ADPs before isotonic fitting.  This makes the fit
        # deterministic even if input row order changes.
        grouped = (
            sub.groupby("adp", as_index=False, observed=True)["actual_points"]
            .mean()
            .sort_values("adp")
        )
        curves[position] = IsotonicRegression(
            increasing=False,
            out_of_bounds="clip",
            y_min=0.0,
        ).fit(grouped["adp"].to_numpy(), grouped["actual_points"].to_numpy())
    return curves


def apply_market_curves(frame: pd.DataFrame, curves: dict[str, IsotonicRegression]) -> pd.Series:
    """Return market-implied points, leaving unmatched rows null."""
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for position, curve in curves.items():
        mask = frame["position"].eq(position) & pd.to_numeric(
            frame["adp"], errors="coerce"
        ).notna()
        if mask.any():
            out.loc[mask] = curve.predict(
                pd.to_numeric(frame.loc[mask, "adp"], errors="coerce").to_numpy()
            )
    return out


def simplex_weights(n_models: int, *, step: float = WEIGHT_STEP) -> Iterable[tuple[float, ...]]:
    """Enumerate a deterministic nonnegative simplex on a fixed grid."""
    units = int(round(1.0 / step))
    if n_models < 1 or not np.isclose(units * step, 1.0):
        raise ValueError("step must divide 1.0 and n_models must be positive")
    for candidate in product(range(units + 1), repeat=n_models):
        if sum(candidate) == units:
            yield tuple(value / units for value in candidate)


def _spearman(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) < 3:
        return float("nan")
    return float(pd.Series(actual).corr(pd.Series(pred), method="spearman"))


def fit_position_weights(
    frame: pd.DataFrame,
    model_cols: tuple[str, ...],
    *,
    actual_col: str = "actual_points",
    step: float = WEIGHT_STEP,
) -> dict[str, dict[str, float]]:
    """Fit per-position convex weights by MAE, with Spearman as tie-break."""
    fitted: dict[str, dict[str, float]] = {}
    for position in POSITIONS:
        sub = frame[frame["position"].eq(position)].dropna(
            subset=[actual_col, *model_cols]
        )
        if len(sub) < 8:
            raise ValueError(f"Need at least 8 fit rows for {position}; got {len(sub)}")
        actual = sub[actual_col].to_numpy(dtype=float)
        models = sub.loc[:, model_cols].to_numpy(dtype=float)
        best: tuple[float, float, tuple[float, ...]] | None = None
        for weights in simplex_weights(len(model_cols), step=step):
            pred = models @ np.asarray(weights, dtype=float)
            mae = float(np.mean(np.abs(pred - actual)))
            rho = _spearman(actual, pred)
            # Stable lexicographic choice: MAE first, then rank, then weights.
            key = (round(mae, 12), -round(rho, 12), weights)
            if best is None or key < best:
                best = key
        assert best is not None
        fitted[position] = {
            col: float(weight) for col, weight in zip(model_cols, best[2], strict=True)
        }
    return fitted


def apply_position_weights(
    frame: pd.DataFrame,
    weights: dict[str, dict[str, float]],
    *,
    out_col: str,
) -> pd.DataFrame:
    """Apply weights after callers have resolved missing-model fallbacks."""
    out = frame.copy()
    values = pd.Series(np.nan, index=out.index, dtype=float)
    for position, position_weights in weights.items():
        mask = out["position"].eq(position)
        if not mask.any():
            continue
        total = pd.Series(0.0, index=out.index[mask], dtype=float)
        for column, weight in position_weights.items():
            if column not in out.columns:
                raise KeyError(f"Missing ensemble input column {column}")
            if out.loc[mask, column].isna().any():
                raise ValueError(f"Unresolved missing values in {column} for {position}")
            total = total + float(weight) * out.loc[mask, column].astype(float)
        values.loc[mask] = total
    out[out_col] = values
    return out


def incumbent_points(frame: pd.DataFrame, weights: dict[str, dict[str, float]]) -> pd.Series:
    """Apply shipped v1/v2 weights; missing v2 explicitly falls back to v1."""
    work = frame.copy()
    work["v2_pred"] = pd.to_numeric(work["v2_pred"], errors="coerce").fillna(
        pd.to_numeric(work["v1_pred"], errors="coerce")
    )
    return apply_position_weights(work, weights, out_col="incumbent_pred")["incumbent_pred"]


def resolve_candidate_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the production fallback rules before fitting or scoring."""
    out = frame.copy()
    out["v1_pred"] = pd.to_numeric(out["v1_pred"], errors="coerce")
    for column in ("v2_pred", "v3_p50"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(out["v1_pred"])
    if "adp_points" not in out.columns:
        out["adp_points"] = np.nan
    out["adp_points"] = pd.to_numeric(out["adp_points"], errors="coerce")
    return out


def metric_block(frame: pd.DataFrame, pred_col: str) -> dict:
    score_frame = frame.rename(columns={"position": "preseason_position"})
    return score_predictions(score_frame, pred_col)


def passes_metrics(candidate: dict, incumbent: dict) -> bool:
    return bool(
        candidate.get("points_mae", float("inf")) <= incumbent.get("points_mae", float("-inf"))
        and candidate.get("spearman", float("-inf")) >= incumbent.get("spearman", float("inf"))
    )


def choose_position_arms(
    frame: pd.DataFrame,
    candidate_cols: dict[str, str],
    *,
    incumbent_col: str = "incumbent_pred",
    arm_complexity: dict[str, int] | None = None,
) -> tuple[dict[str, str], dict[str, dict]]:
    """Select only candidate arms that dominate the incumbent on holdout."""
    selections: dict[str, str] = {}
    evidence: dict[str, dict] = {}
    for position in POSITIONS:
        sub = frame[frame["position"].eq(position)]
        incumbent = metric_block(sub, incumbent_col)
        passing: list[tuple[float, float, int, str, dict]] = []
        arms: dict[str, dict] = {"incumbent": incumbent}
        for arm, column in candidate_cols.items():
            metrics = metric_block(sub, column)
            arms[arm] = metrics
            if passes_metrics(metrics, incumbent):
                complexity = int((arm_complexity or {}).get(arm, 0))
                passing.append((
                    metrics["points_mae"], -metrics["spearman"], complexity, arm, metrics,
                ))
        if passing:
            # Equal forecasts should never be attributed to a larger model.
            passing.sort(
                key=lambda row: (
                    round(row[0], 12), round(row[1], 12), row[2], row[3]
                )
            )
            selected = passing[0][3]
        else:
            selected = "incumbent"
        selections[position] = selected
        evidence[position] = {"selected": selected, "arms": arms}
    return selections, evidence


def assemble_selected(
    frame: pd.DataFrame,
    selections: dict[str, str],
    candidate_cols: dict[str, str],
    *,
    incumbent_col: str = "incumbent_pred",
) -> pd.DataFrame:
    out = frame.copy()
    out["selected_arm"] = "incumbent"
    out["selected_pred"] = out[incumbent_col]
    for position, arm in selections.items():
        if arm == "incumbent":
            continue
        mask = out["position"].eq(position)
        out.loc[mask, "selected_arm"] = arm
        out.loc[mask, "selected_pred"] = out.loc[mask, candidate_cols[arm]]
    return out


def paired_error_block(frame: pd.DataFrame, candidate_col: str) -> dict:
    valid = frame.dropna(subset=["actual_points", candidate_col, "incumbent_pred"])
    incumbent_error = np.abs(valid["incumbent_pred"] - valid["actual_points"])
    candidate_error = np.abs(valid[candidate_col] - valid["actual_points"])
    delta = candidate_error - incumbent_error
    return {
        "n": int(len(valid)),
        "mean_delta_mae": float(delta.mean()) if len(valid) else None,
        "median_delta_absolute_error": float(delta.median()) if len(valid) else None,
        "fraction_improved": float((delta < 0).mean()) if len(valid) else None,
        "fraction_tied": float((delta == 0).mean()) if len(valid) else None,
    }


def bootstrap_deltas(
    frame: pd.DataFrame,
    candidate_col: str,
    *,
    n_boot: int = 2000,
    seed: int = 2026,
) -> dict:
    """Stratified paired bootstrap for MAE and Spearman deltas."""
    valid = frame.dropna(subset=["actual_points", candidate_col, "incumbent_pred"])
    groups = [grp.index.to_numpy() for _, grp in valid.groupby("position", observed=True)]
    rng = np.random.default_rng(seed)
    mae_delta: list[float] = []
    rho_delta: list[float] = []
    for _ in range(n_boot):
        indexes = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        sample = valid.loc[indexes]
        actual = sample["actual_points"].to_numpy(dtype=float)
        candidate = sample[candidate_col].to_numpy(dtype=float)
        incumbent = sample["incumbent_pred"].to_numpy(dtype=float)
        mae_delta.append(float(np.mean(np.abs(candidate - actual) - np.abs(incumbent - actual))))
        rho_delta.append(_spearman(actual, candidate) - _spearman(actual, incumbent))

    def interval(values: list[float]) -> dict:
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(np.nanmean(arr)),
            "p025": float(np.nanquantile(arr, 0.025)),
            "p975": float(np.nanquantile(arr, 0.975)),
        }

    return {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "mae_delta_candidate_minus_incumbent": interval(mae_delta),
        "spearman_delta_candidate_minus_incumbent": interval(rho_delta),
    }
