"""Repair-candidate gates and single-label diagnosis for shadow v1 RB/WR."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

DIAGNOSIS_LABELS = (
    "parity_or_data_defect",
    "raw_rate_model_defect",
    "availability_model_defect",
    "composition_defect",
    "no_isolated_actionable_defect",
)

SPEARMAN_REGRESSION_TOL = 0.01


def _mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual)))


def _spearman(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) < 3:
        return float("nan")
    return float(pd.Series(actual).corr(pd.Series(pred), method="spearman"))


def paired_bootstrap_mae_delta(
    actual: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 2026,
) -> dict[str, float]:
    """Pooled paired bootstrap for MAE(candidate) - MAE(baseline)."""
    rng = np.random.default_rng(seed)
    n = len(actual)
    if n == 0:
        return {"mean": float("nan"), "p025": float("nan"), "p975": float("nan")}
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        base_mae = np.mean(np.abs(baseline[idx] - actual[idx]))
        cand_mae = np.mean(np.abs(candidate[idx] - actual[idx]))
        deltas.append(float(cand_mae - base_mae))
    arr = np.asarray(deltas, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "p025": float(np.quantile(arr, 0.025)),
        "p975": float(np.quantile(arr, 0.975)),
    }


def stage_worsens_top120(
    fold_metrics: list[dict[str, Any]],
    *,
    stage: str,
    position: str,
) -> dict[str, Any]:
    """Count folds where a compose stage increases top-120 MAE for a position."""
    worsens = []
    for fold in fold_metrics:
        if fold.get("position") != position:
            continue
        cell = (fold.get("stage_mae") or {}).get(stage)
        raw = (fold.get("stage_mae") or {}).get("raw_forecast")
        if cell is None or raw is None:
            continue
        if float(cell) > float(raw):
            worsens.append(int(fold["target_season"]))
    return {
        "stage": stage,
        "position": position,
        "worsening_folds": worsens,
        "n_worsening": len(worsens),
        "meets_two_fold_rule": len(worsens) >= 2,
    }


def flag_repair_candidate(
    *,
    stage: str,
    position: str,
    fold_metrics: list[dict[str, Any]],
    pooled_actual: np.ndarray,
    pooled_baseline: np.ndarray,
    pooled_stage: np.ndarray,
    all_eligible_spearman_baseline: float,
    all_eligible_spearman_without_stage: float,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Apply the bounded repair-candidate rules (realized outcomes only)."""
    worsen = stage_worsens_top120(fold_metrics, stage=stage, position=position)
    boot = paired_bootstrap_mae_delta(
        pooled_actual, pooled_baseline, pooled_stage, n_boot=n_boot
    )
    # Stage "harm" means higher MAE than baseline; interval must exclude 0.
    interval_excludes_zero = boot["p025"] > 0 or boot["p975"] < 0
    direction_positive = boot["mean"] > 0
    spearman_ok = (
        all_eligible_spearman_without_stage
        >= all_eligible_spearman_baseline - SPEARMAN_REGRESSION_TOL
        or (
            np.isnan(all_eligible_spearman_baseline)
            and np.isnan(all_eligible_spearman_without_stage)
        )
    )
    flagged = bool(
        worsen["meets_two_fold_rule"]
        and interval_excludes_zero
        and direction_positive
        and spearman_ok
    )
    return {
        "stage": stage,
        "position": position,
        "flagged": flagged,
        "worsening": worsen,
        "bootstrap_mae_delta": boot,
        "interval_excludes_zero": interval_excludes_zero,
        "direction_consistent": direction_positive,
        "spearman_guard_ok": spearman_ok,
        "evidence_source": "realized_nfl_outcomes",
        "sleeper_agreement_used": False,
    }


def classify_diagnosis(
    *,
    parity_defects: Iterable[str] | None = None,
    component_dominance: dict[str, float] | None = None,
    flagged_stages: list[dict[str, Any]] | None = None,
) -> str:
    """Return exactly one diagnosis label."""
    if parity_defects:
        return "parity_or_data_defect"
    flagged = [f for f in (flagged_stages or []) if f.get("flagged")]
    if flagged:
        return "composition_defect"
    dominance = component_dominance or {}
    if not dominance:
        return "no_isolated_actionable_defect"
    # Largest absolute pooled contribution among the four components.
    winner = max(dominance.items(), key=lambda item: abs(float(item[1])))
    name, value = winner
    if abs(float(value)) < 1e-9:
        return "no_isolated_actionable_defect"
    if name == "raw_rate_error":
        return "raw_rate_model_defect"
    if name == "availability_effect":
        return "availability_model_defect"
    if name in {"composition_rate_effect", "finalization_remainder"}:
        return "composition_defect"
    return "no_isolated_actionable_defect"


def repair_gate(
    *,
    fold_mae_deltas: list[float],
    pooled_top120_spearman_baseline: float,
    pooled_top120_spearman_candidate: float,
    all_eligible_ok: bool,
    coverage_unchanged: bool,
    team_identity_unchanged: bool,
) -> dict[str, Any]:
    """Gate a single shadow-only modeling candidate."""
    improves = sum(1 for d in fold_mae_deltas if d < 0)
    no_targeted_loss = all(d <= 0.01 * abs(d) + 1e-12 or d < 0.01 for d in fold_mae_deltas)
    # "No targeted fold loses more than 1% MAE" → delta_mae / baseline <= 0.01
    # Callers pass relative MAE deltas (candidate/baseline - 1).
    relative_ok = all(d <= 0.01 for d in fold_mae_deltas)
    spearman_ok = pooled_top120_spearman_candidate >= pooled_top120_spearman_baseline
    passed = bool(
        improves >= 2
        and relative_ok
        and spearman_ok
        and all_eligible_ok
        and coverage_unchanged
        and team_identity_unchanged
    )
    return {
        "passed": passed,
        "improves_folds": improves,
        "relative_mae_ok": relative_ok,
        "spearman_ok": spearman_ok,
        "all_eligible_ok": all_eligible_ok,
        "coverage_unchanged": coverage_unchanged,
        "team_identity_unchanged": team_identity_unchanged,
        "verdict": "freeze_shadow_candidate" if passed else "hold_v1_structural_role",
        "unused": no_targeted_loss,
    }
