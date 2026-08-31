"""Promotion gate for finish probabilities and simulated draft-value overlays."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.draft_assistant.replacement_contract import default_selected_board_path
from src.projection.contracts import BACKTEST_DIR, OUTPUT_DIR
from src.projection.evaluation.accuracy_first import TOP_ADP, sha256_file
from src.projection.evaluation.finish_probability_calibration import (
    DEFAULT_EVAL_CUTOFFS,
    evaluate_finish_probability_report,
)
from src.projection.inference.recenter import TRANSFORM_VERSION as RECENTER_VERSION
from src.projection.inference.wr_calibration import (
    ARTIFACT_PATH as WR_CALIBRATION_PATH,
    TRANSFORM_VERSION as WR_CALIBRATION_VERSION,
)

FINISH_GATE_PATH = Path(OUTPUT_DIR) / "model_v3" / "finish_probability_gate.json"
VERDICT_READY = "finish_probability_ready"
VERDICT_HOLD = "hold_finish_probabilities"
STATE_DISTRIBUTION_PASS = "holdout_distribution_pass"


def _round_metric(value: float | None, digits: int = 3) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _summarize_finish_metrics(finish_calibration: dict) -> dict:
    """Compact metrics block for publication gate artifact."""
    metrics: dict[str, dict] = {}
    for check in finish_calibration.get("checks") or []:
        cutoff = int(check.get("cutoff", 0))
        key = f"top{cutoff}"
        metrics[key] = {
            "candidate_brier": _round_metric(check.get("brier")),
            "baseline_brier": _round_metric(check.get("baseline_brier")),
            "brier_improvement": _round_metric(check.get("brier_improvement")),
            "calibration_intercept": _round_metric(check.get("calibration_intercept")),
            "calibration_slope": _round_metric(check.get("calibration_slope")),
            "mean_predicted": _round_metric(check.get("mean_predicted")),
            "mean_observed": _round_metric(check.get("mean_observed")),
            "n": int(check.get("n", 0)),
            "passes": bool(check.get("passes", False)),
        }
        wr = (check.get("by_position") or {}).get("WR")
        if cutoff == 12 and wr:
            metrics["wr_top12"] = {
                "n": int(wr.get("n", 0)),
                "candidate_brier": _round_metric(wr.get("brier")),
                "baseline_brier": _round_metric(wr.get("baseline_brier")),
                "brier_improvement": _round_metric(wr.get("brier_improvement")),
                "calibration_intercept": _round_metric(wr.get("calibration_intercept")),
                "calibration_slope": _round_metric(wr.get("calibration_slope")),
                "mean_predicted": _round_metric(wr.get("mean_predicted")),
                "mean_observed": _round_metric(wr.get("mean_observed")),
                "passes": bool(wr.get("passes", False)),
            }
    return metrics


def build_finish_gate_artifact(
    *,
    acceptance: dict,
    finish_calibration: dict,
    provenance: dict | None = None,
    segment_summary: dict | None = None,
    holdout_season: int = 2025,
    holdout_population: str = "top_120_adp",
    n_scored: int = 0,
    baseline_description: str = "2024_preseason_rank_to_finish_rate",
) -> dict:
    """Build the publication gate artifact from holdout evaluation outputs."""
    reasons: list[str] = []
    if not acceptance.get("passes", False):
        reasons.append("recentered_holdout_failed")
    if segment_summary is not None and not segment_summary.get("passes_segment_gate", False):
        reasons.append("segment_calibration_failed")
    if not finish_calibration.get("passes", False):
        reasons.append("finish_probability_calibration_failed")
    if not (finish_calibration.get("checks") or []):
        reasons.append("missing_finish_probability_calibration")

    distribution_pass = acceptance.get("passes", False) and not any(
        r in reasons for r in ("recentered_holdout_failed", "segment_calibration_failed")
    )
    finish_pass = bool(finish_calibration.get("passes", False)) and bool(
        finish_calibration.get("checks")
    )
    passes = not reasons
    prov = dict(provenance or {})
    if segment_summary and segment_summary.get("summary_hash"):
        prov.setdefault("segment_summary_hash", segment_summary["summary_hash"])
    if WR_CALIBRATION_PATH.exists():
        prov.setdefault("wr_calibration_sha256", sha256_file(WR_CALIBRATION_PATH))
    prov.setdefault("recenter_transform_version", RECENTER_VERSION)
    prov.setdefault("wr_calibration_version", WR_CALIBRATION_VERSION)
    calibration_path = Path(BACKTEST_DIR) / "v3_fantasy_interval_calibration.json"
    if calibration_path.exists():
        prov.setdefault("calibration_hash", sha256_file(calibration_path))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": VERDICT_READY if passes else VERDICT_HOLD,
        "verdict": VERDICT_READY if passes else VERDICT_HOLD,
        "passes": passes,
        "publication_verdict": "pass" if passes else "hold",
        "reasons": reasons,
        "distribution_acceptance": {
            "verdict": "pass" if distribution_pass else "hold",
            "source_state": STATE_DISTRIBUTION_PASS if distribution_pass else "wr_calibration_candidate",
            "acceptance": acceptance,
        },
        "finish_calibration": {
            "verdict": "pass" if finish_pass else "hold",
            "holdout_season": int(holdout_season),
            "holdout_population": holdout_population,
            "n_scored": int(n_scored),
            "baseline": baseline_description,
            "metrics": _summarize_finish_metrics(finish_calibration),
            "checks": finish_calibration.get("checks") or [],
            "wr_summary": finish_calibration.get("wr_summary"),
        },
        "provenance": prov,
        "segment_summary_hash": (segment_summary or {}).get("summary_hash"),
        "top120_n": int(n_scored),
        # Legacy compatibility for callers still reading holdout.acceptance fields.
        "holdout": acceptance,
    }


def evaluate_finish_gate(
    scored_top120: pd.DataFrame,
    *,
    segment_summary: dict | None = None,
    recentered_holdout: dict | None = None,
    finish_calibration: dict | None = None,
    provenance: dict | None = None,
    holdout_season: int = 2025,
) -> dict:
    """Decide whether finish/VORP overlays may publish."""
    if finish_calibration is None and not scored_top120.empty:
        finish_calibration = evaluate_finish_probability_report(scored_top120)
    return build_finish_gate_artifact(
        acceptance=recentered_holdout or {},
        finish_calibration=finish_calibration or {"passes": False, "checks": []},
        provenance=provenance,
        segment_summary=segment_summary,
        holdout_season=holdout_season,
        n_scored=int(len(scored_top120)),
    )


def write_finish_probability_gate(payload: dict, path: Path | None = None) -> Path:
    out = Path(path or FINISH_GATE_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def read_finish_probability_gate(path: Path | None = None) -> dict | None:
    out = Path(path or FINISH_GATE_PATH)
    if not out.exists():
        return None
    return json.loads(out.read_text(encoding="utf-8"))


def filter_top120(frame: pd.DataFrame, *, adp_col: str = "adp") -> pd.DataFrame:
    adp = pd.to_numeric(frame[adp_col], errors="coerce")
    return frame.loc[adp.notna() & adp.le(TOP_ADP)].copy()


def _hash_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    return sha256_file(path)


def validate_draw_partitions(manifest: dict) -> tuple[bool, dict]:
    """Verify partitioned draw artifacts exist and match manifest hashes."""
    partition_dir = manifest.get("partition_dir")
    expected_hashes = manifest.get("partition_hashes") or []
    if not partition_dir or not expected_hashes:
        return False, {"reason": "missing_draw_partitions"}
    root = Path(partition_dir)
    if not root.exists():
        return False, {"reason": "partition_dir_missing", "partition_dir": partition_dir}
    parts = sorted(root.glob("part-*.parquet"))
    if len(parts) != len(expected_hashes):
        return False, {
            "reason": "partition_count_mismatch",
            "expected": len(expected_hashes),
            "found": len(parts),
        }
    actual_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in parts]
    if actual_hashes != list(expected_hashes):
        return False, {"reason": "partition_hash_mismatch"}
    return True, {"partition_count": len(parts)}


def validate_finish_probability_publication(
    *,
    season: int,
    board: pd.DataFrame,
    manifest: dict,
    gate: dict,
    fantasy_path: str | None = None,
) -> tuple[bool, dict]:
    """Fail-closed provenance checks before attaching p_finish_* fields."""
    failures: list[str] = []
    details: dict = {}

    state = gate.get("state") or gate.get("verdict")
    if state != VERDICT_READY:
        failures.append("gate_state_not_ready")
    if gate.get("publication_verdict") != "pass":
        failures.append("publication_verdict_not_pass")
    dist = gate.get("distribution_acceptance") or {}
    if dist.get("verdict") != "pass":
        failures.append("distribution_acceptance_not_pass")
    finish = gate.get("finish_calibration") or {}
    if finish.get("verdict") != "pass":
        failures.append("finish_calibration_not_pass")

    board_path = Path(fantasy_path) if fantasy_path else default_selected_board_path(season)
    if not board_path.exists():
        board_path = Path(OUTPUT_DIR) / f"fantasy_points_{season}.csv"
    board_hash = _hash_file(board_path)
    manifest_board_hash = manifest.get("selected_board_hash")
    if not board_hash or not manifest_board_hash or board_hash != manifest_board_hash:
        failures.append("selected_board_hash_mismatch")
        details["board_hash"] = board_hash
        details["manifest_board_hash"] = manifest_board_hash

    expected_model_id = gate.get("provenance", {}).get("board_model_id") or "accuracy_first_ensemble"
    if manifest.get("selected_board_model_id") != expected_model_id:
        failures.append("selected_board_model_id_mismatch")

    if "projection_run_id" in board.columns:
        board_ids = board["projection_run_id"].dropna().astype(str).unique()
        if len(board_ids) == 1:
            run_id = str(board_ids[0])
            manifest_run = manifest.get("canonical_projection_run_id") or manifest.get(
                "source_projection_run_id"
            )
            if not manifest_run or str(manifest_run) != run_id:
                failures.append("projection_run_id_mismatch")
                details["board_run_id"] = run_id
                details["manifest_run_id"] = manifest_run

    gate_prov = gate.get("provenance") or {}
    if manifest.get("transform_version") and gate_prov.get("recenter_transform_version"):
        if manifest["transform_version"] != gate_prov["recenter_transform_version"]:
            failures.append("recenter_transform_version_mismatch")
    if manifest.get("calibration_hash") and gate_prov.get("calibration_hash"):
        if manifest["calibration_hash"] != gate_prov["calibration_hash"]:
            failures.append("calibration_hash_mismatch")
    if manifest.get("wr_calibration_version") and gate_prov.get("wr_calibration_version"):
        if manifest["wr_calibration_version"] != gate_prov["wr_calibration_version"]:
            failures.append("wr_calibration_version_mismatch")
    if gate_prov.get("wr_calibration_sha256"):
        actual_wr_hash = _hash_file(WR_CALIBRATION_PATH)
        if not actual_wr_hash or actual_wr_hash != gate_prov["wr_calibration_sha256"]:
            failures.append("wr_calibration_hash_mismatch")

    segment_hash = gate.get("segment_summary_hash")
    if segment_hash and gate_prov.get("segment_summary_hash"):
        if segment_hash != gate_prov["segment_summary_hash"]:
            failures.append("segment_summary_hash_mismatch")

    partitions_ok, partition_meta = validate_draw_partitions(manifest)
    if not partitions_ok:
        failures.append(partition_meta.get("reason", "partition_validation_failed"))
        details["partitions"] = partition_meta
    else:
        details["partitions"] = partition_meta

    ok = not failures
    return ok, {
        "passed": ok,
        "failures": failures,
        "details": details,
        "gate_state": state,
        "publication_verdict": gate.get("publication_verdict"),
    }
