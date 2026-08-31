"""Fail-closed gate for decision-quality evaluation (E1).

Thresholds are derived from frozen baseline evidence rather than hard-coded
constants.  Missing folds, contract hashes, market snapshots, or baseline
references always yield ``hold``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.projection.contracts import OUTPUT_DIR
from src.projection.evaluation.accuracy_first import canonical_json_hash, sha256_file
from src.projection.evaluation.calibration_segments import MINIMUM_N_FOR_GATE
from src.projection.evaluation.decision_quality import (
    DECISION_QUALITY_DIR,
    DEFAULT_FOLDS,
    FROZEN_BASELINE_ID,
    vorp_tier_contract_hashes,
)

DECISION_QUALITY_GATE_PATH = Path(OUTPUT_DIR) / "evaluation" / "decision_quality_gate.json"
VERDICT_COMPLETE = "decision_quality_complete"
VERDICT_HOLD = "hold_decision_quality"


def _round_metric(value: float | None, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def derive_thresholds_from_baseline(
    baseline_metrics: pd.DataFrame,
    *,
    metric_col: str = "precision",
    group_cols: tuple[str, ...] = ("position", "scope", "forecast_family", "top_n"),
    z: float = 1.0,
) -> list[dict[str, Any]]:
    """Build measured tolerances from frozen baseline means and fold dispersion."""
    if baseline_metrics.empty or metric_col not in baseline_metrics.columns:
        return []
    thresholds: list[dict[str, Any]] = []
    for keys, grp in baseline_metrics.groupby(list(group_cols), observed=True):
        values = pd.to_numeric(grp[metric_col], errors="coerce").dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=0)) if len(values) > 1 else 0.0
        floor = max(0.0, mean - z * std)
        entry = {
            "metric": metric_col,
            "floor": _round_metric(floor),
            "baseline_mean": _round_metric(mean),
            "baseline_std": _round_metric(std),
            "baseline_n_folds": int(len(values)),
            "derivation": f"baseline_mean - {z} * baseline_std",
        }
        if isinstance(keys, tuple):
            for col, val in zip(group_cols, keys, strict=False):
                entry[col] = val
        else:
            entry[group_cols[0]] = keys
        thresholds.append(entry)
    return thresholds


def build_decision_quality_gate(
    *,
    evidence_manifest: dict,
    evaluation_payload: dict,
    frozen_baseline_manifest: dict | None = None,
    required_folds: tuple[int, ...] = DEFAULT_FOLDS,
) -> dict[str, Any]:
    """Assemble the E1 gate artifact."""
    reasons: list[str] = []
    contract = evidence_manifest.get("contract_hashes") or {}
    expected_contract = vorp_tier_contract_hashes()
    for key in ("vorp_module_sha256", "tiers_module_sha256"):
        if contract.get(key) != expected_contract.get(key):
            reasons.append(f"contract_hash_mismatch:{key}")

    fold_targets = {int(f["target_season"]) for f in evidence_manifest.get("folds") or []}
    for target in required_folds:
        if target not in fold_targets:
            reasons.append(f"missing_fold:{target}")

    if not evidence_manifest.get("sha256"):
        reasons.append("missing_evidence_hashes")
    if frozen_baseline_manifest is None:
        reasons.append("missing_frozen_baseline_reference")
    elif frozen_baseline_manifest.get("bundle_id") != FROZEN_BASELINE_ID:
        reasons.append("frozen_baseline_id_mismatch")

    market_timestamps = [
        f.get("market_snapshot_timestamp")
        for f in evidence_manifest.get("folds") or []
    ]
    if any(not ts for ts in market_timestamps):
        reasons.append("missing_market_snapshot_timestamp")

    segment_metrics = evaluation_payload.get("segment_metrics")
    if isinstance(segment_metrics, pd.DataFrame) and not segment_metrics.empty:
        ineligible_gate_segments = segment_metrics[
            segment_metrics["eligible_for_gate"].fillna(False)
            & segment_metrics.get("gate_verdict_applied", False)
        ]
        if not ineligible_gate_segments.empty:
            reasons.append("sub_minimum_segment_affected_gate")

    thresholds: list[dict[str, Any]] = []
    if frozen_baseline_manifest is not None:
        baseline_path = Path(frozen_baseline_manifest.get("_root", "")) / "top_n_metrics.parquet"
        if baseline_path.exists():
            baseline_metrics = pd.read_parquet(baseline_path)
            thresholds = derive_thresholds_from_baseline(baseline_metrics)

    passes = not reasons
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": VERDICT_COMPLETE if passes else VERDICT_HOLD,
        "verdict": VERDICT_COMPLETE if passes else VERDICT_HOLD,
        "passes": passes,
        "publication_verdict": "pass" if passes else "hold",
        "reasons": reasons,
        "minimum_segment_n": MINIMUM_N_FOR_GATE,
        "required_folds": list(required_folds),
        "frozen_baseline_id": FROZEN_BASELINE_ID,
        "thresholds": thresholds,
        "provenance": {
            "evidence_manifest_hash": evidence_manifest.get("sha256", {}).get("manifest.json"),
            "frozen_baseline_manifest_hash": (
                frozen_baseline_manifest.get("sha256", {}).get("manifest.json")
                if frozen_baseline_manifest
                else None
            ),
            "contract_hashes": expected_contract,
        },
        "manifest": evidence_manifest,
    }


def write_decision_quality_gate(payload: dict, path: Path | None = None) -> Path:
    out = Path(path or DECISION_QUALITY_GATE_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def read_decision_quality_gate(path: Path | None = None) -> dict | None:
    out = Path(path or DECISION_QUALITY_GATE_PATH)
    if not out.exists():
        return None
    return json.loads(out.read_text(encoding="utf-8"))


def validate_decision_quality_gate(
    gate: dict,
    *,
    evidence_manifest: dict | None = None,
    frozen_baseline_manifest: dict | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Fail-closed provenance checks for E1 completion."""
    failures: list[str] = []
    if gate.get("verdict") != VERDICT_COMPLETE:
        failures.append("gate_verdict_not_complete")
    if gate.get("publication_verdict") != "pass":
        failures.append("publication_verdict_not_pass")

    prov = gate.get("provenance") or {}
    expected = vorp_tier_contract_hashes()
    for key in ("vorp_module_sha256", "tiers_module_sha256"):
        stored = (prov.get("contract_hashes") or {}).get(key)
        if stored != expected.get(key):
            failures.append(f"contract_hash_mismatch:{key}")

    if evidence_manifest is not None:
        manifest_hash = evidence_manifest.get("sha256", {}).get("manifest.json")
        if manifest_hash != prov.get("evidence_manifest_hash"):
            failures.append("evidence_manifest_hash_mismatch")

    if frozen_baseline_manifest is not None:
        baseline_hash = frozen_baseline_manifest.get("sha256", {}).get("manifest.json")
        if baseline_hash != prov.get("frozen_baseline_manifest_hash"):
            failures.append("frozen_baseline_manifest_hash_mismatch")

    ok = not failures
    return ok, {"passed": ok, "failures": failures}


def load_frozen_baseline_manifest(
    baseline_id: str = FROZEN_BASELINE_ID,
    *,
    root: Path | None = None,
) -> dict | None:
    base = Path(root or DECISION_QUALITY_DIR) / "frozen" / baseline_id
    manifest_path = base / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["_root"] = str(base)
    return manifest
