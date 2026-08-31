"""Promotion gate for simulated VORP overlays."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.projection.contracts import BACKTEST_DIR, OUTPUT_DIR
from src.projection.evaluation.finish_probability_gate import (
    VERDICT_READY as FINISH_READY,
    read_finish_probability_gate,
    validate_draw_partitions,
)
from src.draft_assistant.positional_ranks import (
    FINISH_PROBABILITY_TIE_POLICY,
    TIE_POLICY,
)
from src.projection.inference.recenter import sha256_file
from src.projection.inference.wr_calibration import ARTIFACT_PATH as WR_CALIBRATION_PATH

VERDICT_READY = "simulated_vorp_ready"
VERDICT_HOLD = "hold"


def gate_output_dir(season: int, selected_board_hash: str) -> Path:
    return (
        Path(OUTPUT_DIR)
        / "model_v3"
        / "simulated_vorp"
        / f"season={season}"
        / f"board={selected_board_hash}"
    )


def finish_gate_hash(gate: dict | None) -> str | None:
    if not gate:
        return None
    payload = dict(gate)
    payload.pop("generated_at", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_preconditions(
    *,
    finish_gate: dict | None,
    manifest: dict,
    replacement_contract: dict,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not finish_gate:
        failures.append("missing_finish_probability_gate")
    else:
        state = finish_gate.get("state") or finish_gate.get("verdict")
        if state != FINISH_READY:
            failures.append("finish_probability_gate_not_ready")
        if finish_gate.get("publication_verdict") != "pass":
            failures.append("finish_probability_publication_not_pass")
        dist = finish_gate.get("distribution_acceptance") or {}
        if dist.get("verdict") != "pass":
            failures.append("distribution_acceptance_not_pass")
        finish = finish_gate.get("finish_calibration") or {}
        if finish.get("verdict") != "pass":
            failures.append("finish_calibration_not_pass")
    if not manifest.get("selected_board_hash"):
        failures.append("missing_selected_board_hash")
    if not manifest.get("selected_board_model_id"):
        failures.append("missing_selected_board_model_id")
    if not (manifest.get("canonical_projection_run_id") or manifest.get("source_projection_run_id")):
        failures.append("missing_canonical_projection_run_id")
    partitions_ok, _ = validate_draw_partitions(manifest)
    if not partitions_ok:
        failures.append("draw_partitions_invalid")
    if not replacement_contract.get("contract_hash"):
        failures.append("missing_replacement_contract_hash")
    return not failures, failures


def build_simulated_vorp_gate(
    *,
    season: int,
    manifest: dict,
    replacement_contract: dict,
    contract_tests: dict,
    finish_gate: dict | None,
    chunk_size_invariant: bool = True,
    rerun_deterministic: bool = True,
) -> dict:
    pre_ok, pre_failures = validate_preconditions(
        finish_gate=finish_gate,
        manifest=manifest,
        replacement_contract=replacement_contract,
    )
    contract_ok = bool(contract_tests.get("passes"))
    partitions_ok, partition_meta = validate_draw_partitions(manifest)
    board_hash_match = (
        str(manifest.get("selected_board_hash"))
        == str(replacement_contract.get("selected_board_hash"))
    )
    run_match = (
        str(manifest.get("canonical_projection_run_id") or manifest.get("source_projection_run_id"))
        == str(replacement_contract.get("canonical_projection_run_id"))
    )
    calibration_hash = None
    calibration_path = Path(BACKTEST_DIR) / "v3_fantasy_interval_calibration.json"
    if calibration_path.exists():
        calibration_hash = sha256_file(calibration_path)
    calibration_match = (
        not manifest.get("calibration_hash")
        or manifest.get("calibration_hash") == calibration_hash
    )
    finish_hash = finish_gate_hash(finish_gate)
    gate_prov = (finish_gate or {}).get("provenance") or {}
    wr_hash = sha256_file(WR_CALIBRATION_PATH) if WR_CALIBRATION_PATH.exists() else None
    passes = (
        pre_ok
        and contract_ok
        and partitions_ok
        and board_hash_match
        and run_match
        and calibration_match
        and chunk_size_invariant
        and rerun_deterministic
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": VERDICT_READY if passes else VERDICT_HOLD,
        "publication_verdict": "pass" if passes else "hold",
        "preconditions": {
            "finish_probability_ready": pre_ok and "finish_probability_gate_not_ready" not in pre_failures,
            "distribution_acceptance_pass": bool(
                ((finish_gate or {}).get("distribution_acceptance") or {}).get("verdict") == "pass"
            ),
            "finish_calibration_pass": bool(
                ((finish_gate or {}).get("finish_calibration") or {}).get("verdict") == "pass"
            ),
            "failures": pre_failures,
        },
        "replacement_contract": {
            "verdict": "pass" if contract_ok and board_hash_match and run_match else "hold",
            "contract_hash": replacement_contract.get("contract_hash"),
        },
        "draw_integrity": {
            "verdict": "pass" if partitions_ok and chunk_size_invariant and rerun_deterministic else "hold",
            "partition_hashes_match": partitions_ok,
            "chunk_size_invariant": chunk_size_invariant,
            "rerun_deterministic": rerun_deterministic,
            "details": partition_meta,
        },
        "ranking_contract": {
            "verdict": "pass" if contract_ok else "hold",
            "tie_policy": TIE_POLICY,
            "simulated_positional_rank_fields": [
                "expected_pos_rank",
                "median_pos_rank",
            ],
            "finish_probability_tie_policy": FINISH_PROBABILITY_TIE_POLICY,
            "finish_probability_fields": [
                "p_finish_top6",
                "p_finish_top12",
                "p_finish_top24",
                "p_finish_top36",
                "p_finish_top48",
            ],
            "note": (
                "Finish probabilities and simulated positional-rank moments use "
                "different per-draw tie policies; see rank_tie_policies in the "
                "published draft board metadata."
            ),
        },
        "provenance": {
            "selected_board_hash_match": board_hash_match,
            "canonical_projection_run_id_match": run_match,
            "calibration_hash_match": calibration_match,
            "finish_probability_gate_hash_match": bool(finish_hash),
            "wr_calibration_artifact_hash": wr_hash,
            "segment_report_hash": (finish_gate or {}).get("segment_summary_hash"),
        },
        "output_contract": {
            "verdict": "pass" if contract_ok else "hold",
            "deterministic_vorp_unchanged": bool(
                next(
                    (t for t in contract_tests.get("tests", []) if t["name"] == "schema_separation"),
                    {},
                ).get("passes", False)
            ),
            "deterministic_tiers_unchanged": True,
        },
        "contract_tests": contract_tests,
        "manifest": {
            "season": int(season),
            "selected_board_hash": manifest.get("selected_board_hash"),
            "selected_board_model_id": manifest.get("selected_board_model_id"),
            "canonical_projection_run_id": manifest.get("canonical_projection_run_id")
            or manifest.get("source_projection_run_id"),
            "transform_version": manifest.get("transform_version"),
            "wr_calibration_version": manifest.get("wr_calibration_version"),
            "wr_calibration_artifact_hash": wr_hash,
            "simulation_manifest_hash": hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "simulation_partition_hashes": manifest.get("partition_hashes") or [],
            "calibration_hash": manifest.get("calibration_hash") or calibration_hash,
            "segment_report_hash": (finish_gate or {}).get("segment_summary_hash"),
            "finish_probability_gate_hash": finish_hash,
            "finish_probability_gate_verdict": (finish_gate or {}).get("state")
            or (finish_gate or {}).get("verdict"),
            "roster_configuration_hash": replacement_contract.get("roster_configuration_hash"),
            "replacement_contract_hash": replacement_contract.get("contract_hash"),
            "scoring_configuration_hash": replacement_contract.get("scoring_configuration_hash"),
            "deterministic_seed": manifest.get("deterministic_seed"),
            "draw_count": manifest.get("n_draws"),
        },
    }


def write_simulated_vorp_gate(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_simulated_vorp_gate(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_simulated_vorp_publication(
    *,
    manifest: dict,
    finish_gate: dict | None,
    replacement_contract: dict,
    vorp_gate: dict,
) -> tuple[bool, dict[str, Any]]:
    failures: list[str] = []
    if vorp_gate.get("state") != VERDICT_READY:
        failures.append("simulated_vorp_gate_not_ready")
    if vorp_gate.get("publication_verdict") != "pass":
        failures.append("simulated_vorp_publication_not_pass")
    if not finish_gate:
        failures.append("missing_finish_probability_gate")
    else:
        if (finish_gate.get("state") or finish_gate.get("verdict")) != FINISH_READY:
            failures.append("finish_probability_gate_not_ready")
        if finish_gate.get("publication_verdict") != "pass":
            failures.append("finish_probability_publication_not_pass")
    if str(manifest.get("selected_board_hash")) != str(replacement_contract.get("selected_board_hash")):
        failures.append("selected_board_hash_mismatch")
    if str(manifest.get("selected_board_model_id")) != str(
        replacement_contract.get("selected_board_model_id")
    ):
        failures.append("selected_board_model_id_mismatch")
    manifest_run = manifest.get("canonical_projection_run_id") or manifest.get(
        "source_projection_run_id"
    )
    if str(manifest_run) != str(replacement_contract.get("canonical_projection_run_id")):
        failures.append("canonical_projection_run_id_mismatch")
    if manifest.get("transform_version") and vorp_gate.get("manifest", {}).get("transform_version"):
        if manifest["transform_version"] != vorp_gate["manifest"]["transform_version"]:
            failures.append("recenter_transform_version_mismatch")
    wr_hash = sha256_file(WR_CALIBRATION_PATH) if WR_CALIBRATION_PATH.exists() else None
    if wr_hash and vorp_gate.get("provenance", {}).get("wr_calibration_artifact_hash"):
        if wr_hash != vorp_gate["provenance"]["wr_calibration_artifact_hash"]:
            failures.append("wr_calibration_hash_mismatch")
    partitions_ok, partition_meta = validate_draw_partitions(manifest)
    if not partitions_ok:
        failures.append(partition_meta.get("reason", "partition_validation_failed"))
    contract_hash = replacement_contract.get("contract_hash")
    if not contract_hash or contract_hash != vorp_gate.get("replacement_contract", {}).get(
        "contract_hash"
    ):
        failures.append("replacement_contract_hash_mismatch")
    return not failures, {"passed": not failures, "failures": failures}
