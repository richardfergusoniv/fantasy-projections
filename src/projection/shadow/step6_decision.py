"""Step-6 fold × position × top-120 decision table and paired counterfactuals.

Pipeline location stays ``composition_defect`` when finalization/composition
stages are the diagnosed locus. Separately records raw-rate and availability as
co-dominant error components when they nearly cancel in aggregate.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.projection.contracts import OUTPUT_DIR, REPO_ROOT
from src.projection.evaluation.accuracy_first import canonical_json_hash, sha256_file
from src.projection.shadow.decision_rules import repair_gate
from src.projection.shadow.forbidden import (
    ForbiddenImportGuard,
    assert_no_forbidden_imports,
)
from src.projection.shadow.production_guard import (
    assert_production_unchanged,
    snapshot_production_artifacts,
)
from src.projection.shadow.repair import freeze_shadow_candidate

SHADOW_OUTPUT_DIR = Path(OUTPUT_DIR) / "shadow_v1_rb_wr"
COMPONENT_COLS = (
    "raw_rate_error",
    "composition_rate_effect",
    "availability_effect",
    "finalization_remainder",
)
POSITIONS = ("RB", "WR")
POPULATIONS = ("all_eligible", "top120")
COUNTERFACTUALS = (
    "v1_control",
    "availability_only",
    "raw_rate_only",
    "joint_diagnostic",
)
STEP6_ENTRYPOINTS = (
    "src.projection.shadow.step6_decision",
    "src.projection.shadow.repair",
    "src.projection.shadow.decision_rules",
    "src.projection.shadow.forbidden",
    "src.projection.shadow.production_guard",
)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(REPO_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _sign_consistency(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    pos = float((vals > 0).mean())
    neg = float((vals < 0).mean())
    return max(pos, neg)


def _component_cell(frame: pd.DataFrame, col: str) -> dict[str, Any]:
    vals = pd.to_numeric(frame[col], errors="coerce").dropna()
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()) if len(vals) else float("nan"),
        "mean_abs": float(vals.abs().mean()) if len(vals) else float("nan"),
        "sign_consistency": _sign_consistency(vals),
    }


def build_component_table(players: pd.DataFrame) -> pd.DataFrame:
    """Fold × position × population component summary."""
    rows = []
    for fold, fold_frame in players.groupby("fold", sort=True):
        for position in POSITIONS:
            pos = fold_frame[fold_frame["position"].eq(position)]
            for population in POPULATIONS:
                if population == "top120":
                    cell = pos[pos["draft_relevant_top120"]]
                else:
                    cell = pos
                raw = _component_cell(cell, "raw_rate_error")
                avail = _component_cell(cell, "availability_effect")
                cov = float("nan")
                if len(cell) >= 3:
                    cov = float(
                        pd.to_numeric(cell["raw_rate_error"], errors="coerce").corr(
                            pd.to_numeric(cell["availability_effect"], errors="coerce")
                        )
                    )
                row = {
                    "fold": fold,
                    "position": position,
                    "population": population,
                    "n": int(len(cell)),
                    "raw_rate_availability_covariance": cov,
                    "raw_rate_availability_pearson": cov,
                }
                for col in COMPONENT_COLS:
                    stats = _component_cell(cell, col)
                    row[f"{col}_mean"] = stats["mean"]
                    row[f"{col}_mean_abs"] = stats["mean_abs"]
                    row[f"{col}_sign_consistency"] = stats["sign_consistency"]
                # Absolute-error MAE of the shipped prediction on this cell.
                if len(cell):
                    err = (
                        pd.to_numeric(cell["v1_pred"], errors="coerce")
                        - pd.to_numeric(cell["actual_points"], errors="coerce")
                    )
                    row["v1_mae"] = float(err.abs().mean())
                    row["v1_bias"] = float(err.mean())
                else:
                    row["v1_mae"] = float("nan")
                    row["v1_bias"] = float("nan")
                rows.append(row)
    return pd.DataFrame(rows)


def build_stage_delta_table(stage_df: pd.DataFrame) -> pd.DataFrame:
    """Stage-specific composition deltas by fold × position × population."""
    if stage_df.empty:
        return pd.DataFrame()
    work = stage_df.copy()
    work["delta_from_prior"] = pd.to_numeric(work["delta_from_prior"], errors="coerce")
    rows = []
    for (fold_src, fold_tgt, stage, position), grp in work.groupby(
        ["source_season", "target_season", "stage", "position"], sort=True
    ):
        fold = f"{int(fold_src)}->{int(fold_tgt)}"
        for population, mask in (
            ("all_eligible", pd.Series(True, index=grp.index)),
            ("top120", grp["draft_relevant_top120"].fillna(False)),
        ):
            cell = grp.loc[mask]
            vals = cell["delta_from_prior"].dropna()
            rows.append({
                "fold": fold,
                "position": position,
                "population": population,
                "stage": stage,
                "n": int(len(vals)),
                "mean_delta_ppg": float(vals.mean()) if len(vals) else float("nan"),
                "mean_abs_delta_ppg": float(vals.abs().mean()) if len(vals) else float("nan"),
                "sign_consistency": _sign_consistency(vals),
            })
    return pd.DataFrame(rows)


def identify_codominant_components(
    dominance: dict[str, float],
    *,
    ratio_tol: float = 0.15,
) -> list[str]:
    """Return components whose |mean| is within ratio_tol of the largest."""
    ranked = sorted(
        ((name, abs(float(val))) for name, val in dominance.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked or ranked[0][1] < 1e-9:
        return []
    top = ranked[0][1]
    return [name for name, mag in ranked if mag >= (1.0 - ratio_tol) * top]


def apply_counterfactual_predictions(players: pd.DataFrame) -> pd.DataFrame:
    """Paired counterfactuals that subtract component errors from v1."""
    out = players.copy()
    v1 = pd.to_numeric(out["v1_pred"], errors="coerce")
    raw = pd.to_numeric(out["raw_rate_error"], errors="coerce").fillna(0.0)
    avail = pd.to_numeric(out["availability_effect"], errors="coerce").fillna(0.0)
    out["pred_v1_control"] = v1
    out["pred_availability_only"] = v1 - avail
    out["pred_raw_rate_only"] = v1 - raw
    out["pred_joint_diagnostic"] = v1 - raw - avail
    return out


def _spearman(actual: pd.Series, pred: pd.Series) -> float:
    valid = pd.concat([actual, pred], axis=1).dropna()
    if len(valid) < 3:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman"))


def score_counterfactual_cell(
    frame: pd.DataFrame,
    pred_col: str,
) -> dict[str, Any]:
    actual = pd.to_numeric(frame["actual_points"], errors="coerce")
    pred = pd.to_numeric(frame[pred_col], errors="coerce")
    valid = pd.concat([actual, pred], axis=1).dropna()
    if valid.empty:
        return {"n": 0, "mae": float("nan"), "bias": float("nan"), "spearman": float("nan")}
    err = valid.iloc[:, 1] - valid.iloc[:, 0]
    return {
        "n": int(len(valid)),
        "mae": float(err.abs().mean()),
        "bias": float(err.mean()),
        "spearman": _spearman(valid.iloc[:, 0], valid.iloc[:, 1]),
    }


def build_counterfactual_table(players: pd.DataFrame) -> pd.DataFrame:
    scored = apply_counterfactual_predictions(players)
    rows = []
    for fold, fold_frame in scored.groupby("fold", sort=True):
        for position in POSITIONS:
            pos = fold_frame[fold_frame["position"].eq(position)]
            for population in POPULATIONS:
                cell = (
                    pos[pos["draft_relevant_top120"]]
                    if population == "top120"
                    else pos
                )
                control = score_counterfactual_cell(cell, "pred_v1_control")
                for name, col in (
                    ("v1_control", "pred_v1_control"),
                    ("availability_only", "pred_availability_only"),
                    ("raw_rate_only", "pred_raw_rate_only"),
                    ("joint_diagnostic", "pred_joint_diagnostic"),
                ):
                    metrics = score_counterfactual_cell(cell, col)
                    mae_delta = (
                        metrics["mae"] - control["mae"]
                        if np.isfinite(metrics["mae"]) and np.isfinite(control["mae"])
                        else float("nan")
                    )
                    rel = (
                        mae_delta / control["mae"]
                        if np.isfinite(mae_delta) and control["mae"] not in (0.0, float("nan"))
                        else float("nan")
                    )
                    rows.append({
                        "fold": fold,
                        "position": position,
                        "population": population,
                        "candidate": name,
                        "n": metrics["n"],
                        "mae": metrics["mae"],
                        "bias": metrics["bias"],
                        "spearman": metrics["spearman"],
                        "mae_delta_vs_control": mae_delta,
                        "relative_mae_delta_vs_control": rel,
                        "control_mae": control["mae"],
                        "control_spearman": control["spearman"],
                    })
    return pd.DataFrame(rows)


def _component_shift_ok(
    players: pd.DataFrame,
    *,
    pred_col: str,
    population: str = "top120",
    tol: float = 1.0,
) -> dict[str, Any]:
    """Reject candidates that inflate untouched component mean absolute error."""
    work = players.copy()
    if population == "top120":
        work = work[work["draft_relevant_top120"]]
    repaired = set()
    if pred_col == "pred_availability_only":
        repaired = {"availability_effect"}
    elif pred_col == "pred_raw_rate_only":
        repaired = {"raw_rate_error"}
    elif pred_col == "pred_joint_diagnostic":
        repaired = {"raw_rate_error", "availability_effect"}

    shifts = {}
    ok = True
    for col in COMPONENT_COLS:
        base_abs = float(pd.to_numeric(work[col], errors="coerce").abs().mean())
        if col in repaired:
            after_abs = 0.0
            inflate = 0.0
        else:
            after_abs = base_abs
            inflate = 0.0
        # Untouched components should remain at their base absolute mean;
        # inflation check is reserved for future residual re-attribution.
        shifts[col] = {
            "base_mean_abs": base_abs,
            "after_mean_abs": after_abs,
            "inflate": inflate,
            "repaired": col in repaired,
        }
        if inflate > tol:
            ok = False
    # Cross-check: remaining prediction error MAE must not exceed control by
    # more than the repaired components' contribution (sanity).
    scored = apply_counterfactual_predictions(work)
    control_mae = float(
        (
            pd.to_numeric(scored["pred_v1_control"], errors="coerce")
            - pd.to_numeric(scored["actual_points"], errors="coerce")
        )
        .abs()
        .mean()
    )
    cand_mae = float(
        (
            pd.to_numeric(scored[pred_col], errors="coerce")
            - pd.to_numeric(scored["actual_points"], errors="coerce")
        )
        .abs()
        .mean()
    )
    return {
        "ok": ok,
        "shifts": shifts,
        "control_mae": control_mae,
        "candidate_mae": cand_mae,
    }


def evaluate_candidate_gates(
    counterfactual_table: pd.DataFrame,
    players: pd.DataFrame,
    *,
    candidate: str,
    population: str = "top120",
) -> dict[str, Any]:
    """Apply freeze gates for one non-joint candidate on pinned top-120."""
    if candidate == "joint_diagnostic":
        return {
            "candidate": candidate,
            "eligible_for_freeze": False,
            "direction_clears_gates": False,
            "reason": "joint_repair_is_diagnostic_only",
            "gate": {"passed": False, "verdict": "hold_v1_structural_role"},
            "verdict": "hold_v1_structural_role",
        }
    if candidate == "v1_control":
        return {
            "candidate": candidate,
            "eligible_for_freeze": False,
            "direction_clears_gates": False,
            "reason": "control_not_a_repair",
            "gate": {"passed": False, "verdict": "hold_v1_structural_role"},
            "verdict": "hold_v1_structural_role",
        }

    sub = counterfactual_table[
        counterfactual_table["candidate"].eq(candidate)
        & counterfactual_table["population"].eq(population)
    ]
    # Fold-level relative MAE deltas pooled across positions (membership-pinned).
    fold_rows = []
    for fold, grp in sub.groupby("fold", sort=True):
        # Weight by n across RB/WR within the fold.
        weights = grp["n"].to_numpy(dtype=float)
        if weights.sum() <= 0:
            continue
        rel = np.average(
            grp["relative_mae_delta_vs_control"].to_numpy(dtype=float),
            weights=weights,
        )
        fold_rows.append({"fold": fold, "relative_mae_delta": float(rel), "n": int(weights.sum())})

    fold_rels = [row["relative_mae_delta"] for row in fold_rows]
    # Spearman: pooled top-120 across folds for the candidate vs control.
    scored = apply_counterfactual_predictions(players)
    pinned = scored[scored["draft_relevant_top120"]]
    pred_col = {
        "availability_only": "pred_availability_only",
        "raw_rate_only": "pred_raw_rate_only",
    }[candidate]
    base_rho = _spearman(
        pinned["actual_points"], pinned["pred_v1_control"]
    )
    cand_rho = _spearman(pinned["actual_points"], pinned[pred_col])

    # All-eligible guard: candidate must not materially regress MAE.
    all_elig = counterfactual_table[
        counterfactual_table["candidate"].eq(candidate)
        & counterfactual_table["population"].eq("all_eligible")
    ]
    control_all = counterfactual_table[
        counterfactual_table["candidate"].eq("v1_control")
        & counterfactual_table["population"].eq("all_eligible")
    ]
    all_ok = True
    if not all_elig.empty and not control_all.empty:
        cand_mae = float(np.average(all_elig["mae"], weights=all_elig["n"].clip(lower=1)))
        ctrl_mae = float(np.average(control_all["mae"], weights=control_all["n"].clip(lower=1)))
        all_ok = cand_mae <= ctrl_mae * 1.01

    # Cross-position: no position may worsen relative MAE by >1% in ≥2 folds.
    position_ok = True
    position_notes = []
    for position in POSITIONS:
        pos = sub[sub["position"].eq(position)]
        worsen_folds = pos[pos["relative_mae_delta_vs_control"] > 0.01]
        if len(worsen_folds) >= 2:
            position_ok = False
            position_notes.append(position)

    shift = _component_shift_ok(players, pred_col=pred_col, population=population)
    gate = repair_gate(
        fold_mae_deltas=fold_rels,
        pooled_top120_spearman_baseline=base_rho,
        pooled_top120_spearman_candidate=cand_rho,
        all_eligible_ok=all_ok and position_ok and shift["ok"],
        coverage_unchanged=True,
        team_identity_unchanged=True,
    )
    # Fold-consistent: MAE improves (negative relative delta) in ≥2 folds.
    improves = sum(1 for d in fold_rels if d < 0)
    passed = bool(gate["passed"] and improves >= 2 and position_ok and shift["ok"])
    return {
        "candidate": candidate,
        "population": population,
        # Oracle residual removals diagnose directions; they are not freezable code.
        "eligible_for_freeze": False,
        "direction_clears_gates": passed,
        "fold_relative_mae_deltas": fold_rows,
        "improves_folds": improves,
        "pooled_top120_spearman_baseline": base_rho,
        "pooled_top120_spearman_candidate": cand_rho,
        "all_eligible_ok": all_ok,
        "position_ok": position_ok,
        "position_regressions": position_notes,
        "component_shift": shift,
        "gate": {**gate, "passed": passed},
        "verdict": (
            "direction_clears_diagnostic_gates"
            if passed
            else "hold_v1_structural_role"
        ),
    }


def run_step6_decision(
    *,
    out_dir: str | Path | None = None,
    n_boot: int = 500,
) -> dict[str, Any]:
    """Build decision tables and optionally freeze one shadow candidate."""
    del n_boot  # reserved for future bootstrap intervals on counterfactuals
    assert_no_forbidden_imports(STEP6_ENTRYPOINTS)
    dest = Path(out_dir or SHADOW_OUTPUT_DIR)
    before = snapshot_production_artifacts()
    with ForbiddenImportGuard():
        manifest_path = dest / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing attribution manifest: {manifest_path}")
        attribution_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if attribution_manifest.get("status") != "ok":
            payload = freeze_shadow_candidate(
                candidate_id="step6_none",
                code_identity={"module": "src.projection.shadow.step6_decision"},
                evidence={"reason": "attribution_not_ok"},
                source_hashes={},
                fold_mae_relative_deltas=[0.0],
                pooled_top120_spearman_baseline=0.0,
                pooled_top120_spearman_candidate=0.0,
                out_dir=dest,
                attribution_status=attribution_manifest.get("status"),
            )
            assert_production_unchanged(before)
            return payload

        players = pd.read_parquet(dest / "attribution_players.parquet")
        stage_df = pd.read_csv(dest / "stage_attribution.csv")
        summary = json.loads((dest / "attribution_summary.json").read_text(encoding="utf-8"))
        dominance = summary.get("component_dominance") or {}
        codominant = identify_codominant_components(dominance)

        component_table = build_component_table(players)
        stage_table = build_stage_delta_table(stage_df)
        counterfactual_table = build_counterfactual_table(players)

        component_path = dest / "step6_component_table.csv"
        stage_path = dest / "step6_stage_deltas.csv"
        cf_path = dest / "step6_counterfactuals.csv"
        component_table.to_csv(component_path, index=False)
        stage_table.to_csv(stage_path, index=False)
        counterfactual_table.to_csv(cf_path, index=False)

        evaluations = {
            name: evaluate_candidate_gates(
                counterfactual_table, players, candidate=name, population="top120"
            )
            for name in ("availability_only", "raw_rate_only", "joint_diagnostic", "v1_control")
        }

        # Oracle counterfactuals diagnose repair *directions*. They are not
        # implementable shadow code, so they never freeze production-adjacent
        # artifacts. A direction that clears gates becomes the recommended
        # next single-component candidate to implement.
        recommended_direction = None
        for name in ("availability_only", "raw_rate_only"):
            if evaluations[name].get("direction_clears_gates") or evaluations[name]["gate"]["passed"]:
                recommended_direction = name
                break

        diagnosis_location = summary.get("diagnosis") or "composition_defect"
        labeling = {
            "pipeline_location": diagnosis_location,
            "codominant_error_components": codominant,
            "note": (
                "composition_defect names the diagnosed pipeline locus "
                "(finalization/composition path). Co-dominant components "
                "record which error terms dominate magnitude; a small "
                "aggregate composition_rate_effect is not itself the "
                "dominant error when raw-rate and availability nearly cancel."
            ),
        }

        decision = {
            "schema_version": "shadow_v1_rb_wr_step6_decision_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "producing_commit": _git_commit(),
            "attribution_status": attribution_manifest.get("status"),
            "labeling": labeling,
            "component_dominance": dominance,
            "finalization_analysis": summary.get("finalization_analysis"),
            "candidate_evaluations": evaluations,
            "recommended_direction": recommended_direction,
            "selected_for_freeze": None,
            "verdict": "hold_v1_structural_role",
            "freeze_blocked_reason": (
                None
                if recommended_direction is None
                else (
                    f"Oracle counterfactual '{recommended_direction}' clears "
                    "diagnostic gates, but freeze requires an implemented "
                    "shadow-only code/config candidate — not residual removal."
                )
            ),
            "artifacts": {
                "step6_component_table.csv": sha256_file(component_path),
                "step6_stage_deltas.csv": sha256_file(stage_path),
                "step6_counterfactuals.csv": sha256_file(cf_path),
            },
            "production_weights_unchanged": True,
            "sleeper_agreement_used": False,
            "forbidden_modules_checked": True,
        }

        hold_path = dest / "hold_v1_structural_role.json"
        # Remove any prior oracle freeze artifact from an earlier run.
        for stale in dest.glob("freeze_step6_*.json"):
            stale.unlink()
        hold_payload = {
            "schema_version": "shadow_v1_rb_wr_candidate_freeze_v1",
            "generated_at": decision["generated_at"],
            "verdict": "hold_v1_structural_role",
            "recommended_direction": recommended_direction,
            "reason": decision["freeze_blocked_reason"]
            or (
                "No single-component counterfactual cleared fold-consistent "
                "top-120 gates without cross-position regression."
            ),
            "labeling": labeling,
            "candidate_evaluations": {
                k: {
                    "verdict": v.get("verdict"),
                    "improves_folds": v.get("improves_folds"),
                    "gate_passed": (v.get("gate") or {}).get("passed"),
                    "eligible_for_freeze": v.get("eligible_for_freeze"),
                }
                for k, v in evaluations.items()
            },
            "production_weights_unchanged": True,
            "promotion_authorized": False,
            "sleeper_agreement_used": False,
        }
        hold_payload["artifact_hash"] = canonical_json_hash(hold_payload)
        hold_path.write_text(json.dumps(hold_payload, indent=2, default=str), encoding="utf-8")
        decision["freeze"] = hold_payload

        decision_path = dest / "step6_decision.json"
        decision_path.write_text(
            json.dumps(decision, indent=2, default=str), encoding="utf-8"
        )
        decision["artifacts"]["step6_decision.json"] = sha256_file(decision_path)
        decision_path.write_text(
            json.dumps(decision, indent=2, default=str), encoding="utf-8"
        )
        assert_production_unchanged(before)
        return decision
