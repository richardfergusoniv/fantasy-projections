"""Fail-closed gate for E2 QB-context retrain evaluation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.projection.contracts import OUTPUT_DIR
from src.projection.evaluation.calibration_segments import MINIMUM_N_FOR_GATE
from src.projection.evaluation.decision_quality_gate import derive_thresholds_from_baseline
from src.projection.evaluation.qb_context_evaluation import (
    FROZEN_BASELINE_ID,
    QB_CONTEXT_EVAL_DIR,
)
from src.projection.qb_context import QB_CONTEXT_FEATURES, model_artifact_manifest

QB_CONTEXT_GATE_PATH = Path(OUTPUT_DIR) / "evaluation" / "qb_context_gate.json"
VERDICT_REVIEW = "qb_context_review_ready"
VERDICT_HOLD = "hold_qb_context"


def build_qb_context_gate(
    *,
    evidence_manifest: dict,
    evaluation_payload: dict,
    frozen_baseline_manifest: dict | None = None,
    required_folds: tuple[int, ...] = (2023, 2024, 2025),
) -> dict[str, Any]:
    reasons: list[str] = []
    fold_targets = {
        int(f.get("target_season", -1))
        for f in evidence_manifest.get("folds", [])
    }
    for target in required_folds:
        if target not in fold_targets:
            reasons.append(f"missing_fold:{target}")

    if frozen_baseline_manifest is None:
        reasons.append("missing_frozen_baseline")
    elif frozen_baseline_manifest.get("bundle_id") != FROZEN_BASELINE_ID:
        reasons.append("frozen_baseline_id_mismatch")

    if not evidence_manifest.get("sha256"):
        reasons.append("missing_evidence_hashes")

    cohort = evaluation_payload.get("cohort_metrics")
    if isinstance(cohort, pd.DataFrame):
        eligible = cohort[cohort["eligible_for_gate"].fillna(False)]
        if eligible.empty:
            reasons.append("missing_eligible_cohort_reports")
    else:
        reasons.append("missing_cohort_reports")

    for fold in evaluation_payload.get("folds", []):
        base_meta = fold.get("baseline_metadata") or {}
        cand_meta = fold.get("candidate_metadata") or {}
        if not base_meta.get("training_pairs"):
            reasons.append("missing_baseline_retrain_provenance")
        if not cand_meta.get("training_pairs"):
            reasons.append("missing_candidate_retrain_provenance")
        if not cand_meta.get("use_qb_context"):
            reasons.append("candidate_missing_qb_context_flag")

    thresholds: list[dict[str, Any]] = []
    if frozen_baseline_manifest is not None:
        baseline_path = Path(frozen_baseline_manifest.get("_root", "")) / "cohort_metrics.parquet"
        if baseline_path.exists():
            baseline_metrics = pd.read_parquet(baseline_path)
            if "candidate_points_mae" in baseline_metrics.columns:
                thresholds = derive_thresholds_from_baseline(
                    baseline_metrics,
                    metric_col="candidate_points_mae",
                    group_cols=("cohort_type", "cohort_value"),
                )

    passes = not reasons
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": VERDICT_REVIEW if passes else VERDICT_HOLD,
        "verdict": VERDICT_REVIEW if passes else VERDICT_HOLD,
        "passes": passes,
        "publication_verdict": "review" if passes else "hold",
        "reasons": reasons,
        "minimum_segment_n": MINIMUM_N_FOR_GATE,
        "required_folds": list(required_folds),
        "frozen_baseline_id": FROZEN_BASELINE_ID,
        "thresholds": thresholds,
        "threshold_derivation": "paired_rolling_origin_baseline_candidate_points_mae",
        "provenance": {
            "evidence_manifest_hash": evidence_manifest.get("sha256", {}).get("manifest.json"),
            "frozen_baseline_manifest_hash": (
                frozen_baseline_manifest.get("sha256", {}).get("manifest.json")
                if frozen_baseline_manifest
                else None
            ),
            "qb_context_features": list(QB_CONTEXT_FEATURES),
            "model_manifest": model_artifact_manifest(consumes_qb_context=True),
        },
        "manifest": evidence_manifest,
        "mandatory_stop": True,
        "authorizes_production_integration": False,
    }


def write_qb_context_gate(payload: dict, path: Path | None = None) -> Path:
    out = Path(path or QB_CONTEXT_GATE_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def read_qb_context_gate(path: Path | None = None) -> dict | None:
    out = Path(path or QB_CONTEXT_GATE_PATH)
    if not out.exists():
        return None
    return json.loads(out.read_text(encoding="utf-8"))


def load_frozen_qb_baseline_manifest(
    baseline_id: str = FROZEN_BASELINE_ID,
    *,
    root: Path | None = None,
) -> dict | None:
    base = Path(root or QB_CONTEXT_EVAL_DIR) / "frozen" / baseline_id
    manifest_path = base / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["_root"] = str(base)
    return manifest
