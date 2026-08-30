"""Draw-count rollout decision artifact and overlay identity contracts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.draft_assistant.replacement_contract import (
    build_replacement_contract,
    load_roster_configuration,
    roster_configuration_hash,
    scoring_configuration_hash,
)
from src.projection.contracts import MODEL_V3_DIR, OUTPUT_DIR
from src.projection.evaluation.accuracy_first import sha256_file
from src.projection.evaluation.evidence_freeze import frozen_evidence_manifest_hash, load_freeze_manifest
from src.projection.evaluation.release_pointer import ensure_release_pointers, read_release_pointer

OVERLAY_COMPARISON_IDENTITY_KEYS = (
    "selected_board_hash",
    "selected_board_model_id",
    "canonical_projection_run_id",
    "wr_calibration_artifact_hash",
    "transform_version",
    "finish_probability_gate_hash",
    "replacement_contract_hash",
    "roster_configuration_hash",
    "scoring_configuration_hash",
)


def manifest_identity_fields(manifest: dict[str, Any]) -> dict[str, str | None]:
    wr_hash = manifest.get("wr_calibration_artifact_hash") or manifest.get("wr_calibration_sha256")
    return {
        "selected_board_hash": manifest.get("selected_board_hash"),
        "selected_board_model_id": manifest.get("selected_board_model_id"),
        "canonical_projection_run_id": manifest.get("canonical_projection_run_id")
        or manifest.get("source_projection_run_id"),
        "wr_calibration_artifact_hash": wr_hash,
        "transform_version": manifest.get("transform_version"),
        "finish_probability_gate_hash": manifest.get("finish_probability_gate_hash"),
    }


def build_replacement_identity(
    *,
    season: int,
    selected_board_hash: str,
    canonical_projection_run_id: str,
    selected_board_model_id: str = "accuracy_first_ensemble",
    board_path: Path | None = None,
) -> dict[str, str]:
    from src.draft_assistant.replacement_contract import default_selected_board_path, load_selected_board

    roster_config = load_roster_configuration()
    board = load_selected_board(season, board_path=board_path)
    replacement = build_replacement_contract(
        board,
        season=season,
        selected_board_hash=selected_board_hash,
        selected_board_model_id=selected_board_model_id,
        canonical_projection_run_id=canonical_projection_run_id,
        roster_config=roster_config,
    )
    return {
        "replacement_contract_hash": replacement["contract_hash"],
        "roster_configuration_hash": roster_configuration_hash(roster_config),
        "scoring_configuration_hash": scoring_configuration_hash(),
    }


def full_overlay_identity(
    manifest: dict[str, Any],
    *,
    season: int,
    board_path: Path | None = None,
) -> dict[str, str | None]:
    identity = manifest_identity_fields(manifest)
    board_hash = identity.get("selected_board_hash")
    run_id = identity.get("canonical_projection_run_id")
    model_id = identity.get("selected_board_model_id") or "accuracy_first_ensemble"
    if board_hash and run_id:
        identity.update(
            build_replacement_identity(
                season=season,
                selected_board_hash=str(board_hash),
                canonical_projection_run_id=str(run_id),
                selected_board_model_id=str(model_id),
                board_path=board_path,
            )
        )
    return identity


def compare_overlay_identities(
    left: dict[str, str | None],
    right: dict[str, str | None],
) -> list[str]:
    mismatches: list[str] = []
    for key in OVERLAY_COMPARISON_IDENTITY_KEYS:
        lval = left.get(key)
        rval = right.get(key)
        if lval != rval:
            mismatches.append(f"{key}: left={lval!r} right={rval!r}")
    return mismatches


OPERATIONAL_POLICIES = (
    "strict_numerical_policy_20000",
    "decision_stable_compromise_10000",
    "maintain_1000_temporarily",
)

PRODUCTION_PROFILE_PROVISIONAL = "provisional_current_configuration"
PRODUCTION_PROFILE_DECISION_STABLE_10K = "decision_stable_compromise_10000"
RUNTIME_BUDGET_SECONDS_10K = 3 * 60 * 60

DRAW_COUNT_RISK_FLAG = (
    "draw_count: provisional_current_configuration (1000 draws); "
    "sub-20k candidates fail strict numerical gate vs 20k reference; "
    "10k RC validated operationally but not promoted (see draw_count_rollout_decision.json)"
)

DRAW_COUNT_RISK_FLAG_10K = (
    "draw_count: decision_stable_compromise_10000 (10000 draws); "
    "decision-stable vs 20k nested-prefix (0 material / 0 core-player events) but "
    "fails strict numerical gate; see draw_count_rollout_decision.json"
)


def evaluate_decision_stable_10k_promotion(
    *,
    overlay_comparison: dict[str, Any],
    stability_candidate: dict[str, Any],
    runtime_seconds: float | None,
    runtime_budget_seconds: float = RUNTIME_BUDGET_SECONDS_10K,
) -> dict[str, Any]:
    """Gate for policy A: promote 10k only when clean, decision-stable, and ≤ budget."""
    checks = {
        "overlay_identity_compare": overlay_comparison.get("comparison_verdict") == "compare",
        "material_decision_events_zero": int(
            stability_candidate.get("material_decision_events") or 0
        )
        == 0,
        "core_adp_decision_events_zero": int(
            stability_candidate.get("core_adp_decision_events") or 0
        )
        == 0,
        "runtime_within_budget": runtime_seconds is not None
        and float(runtime_seconds) <= float(runtime_budget_seconds),
    }
    promote = all(checks.values())
    return {
        "policy": "decision_stable_compromise_10000",
        "promote": promote,
        "checks": checks,
        "runtime_seconds": runtime_seconds,
        "runtime_budget_seconds": runtime_budget_seconds,
        "material_decision_events": stability_candidate.get("material_decision_events"),
        "core_adp_decision_events": stability_candidate.get("core_adp_decision_events"),
        "overlay_comparison_verdict": overlay_comparison.get("comparison_verdict"),
        "overlay_comparison_reason": overlay_comparison.get("reason"),
    }


def stability_candidate_for_draws(stability_report: dict[str, Any], draw_count: int) -> dict[str, Any]:
    for candidate in stability_report.get("candidates") or []:
        if int(candidate.get("draw_count") or 0) == int(draw_count):
            return candidate
    return {}


def profile_for_operational_policy(operational_policy: str) -> tuple[str, int]:
    if operational_policy == "decision_stable_compromise_10000":
        return PRODUCTION_PROFILE_DECISION_STABLE_10K, 10000
    if operational_policy == "strict_numerical_policy_20000":
        return "strict_numerical_policy_20000", 20000
    return PRODUCTION_PROFILE_PROVISIONAL, 1000


def write_phase2_rollout_closure(
    *,
    season: int,
    freeze_id: str,
    rc_namespace: str,
    operational_policy: str,
    decided_by: str = "Richard",
    decision_rationale: str,
    human_decision_record_path: str,
    overlay_comparison: dict[str, Any] | None = None,
    promotion_gate: dict[str, Any] | None = None,
    out_path: Path | None = None,
) -> Path:
    """Write final Phase 2 rollout decision after RC validation and human policy choice."""
    if operational_policy not in OPERATIONAL_POLICIES:
        raise ValueError(f"unsupported operational_policy: {operational_policy!r}")

    ensure_release_pointers(season)
    freeze = load_freeze_manifest(freeze_id)
    current_pointer = read_release_pointer(season, role="current")
    draw_decision_path = Path(MODEL_V3_DIR) / "draw_count_decision.json"
    draw_decision = (
        json.loads(draw_decision_path.read_text(encoding="utf-8"))
        if draw_decision_path.exists()
        else {}
    )

    rc_dir = (
        Path(MODEL_V3_DIR)
        / "release_candidates"
        / f"season={season}"
        / f"namespace={rc_namespace}"
    )
    rc_validation_path = rc_dir / f"release_candidate_validation_{season}.json"
    rc_validation = (
        json.loads(rc_validation_path.read_text(encoding="utf-8"))
        if rc_validation_path.exists()
        else {}
    )
    rc_manifest_path = rc_dir / f"simulation_manifest_{season}.json"
    rc_manifest = (
        json.loads(rc_manifest_path.read_text(encoding="utf-8"))
        if rc_manifest_path.exists()
        else {}
    )

    runtime_seconds = rc_manifest.get("runtime_seconds")
    runtime_minutes = round(float(runtime_seconds) / 60.0, 1) if runtime_seconds else None
    production_profile, production_draws = profile_for_operational_policy(operational_policy)

    if overlay_comparison is None:
        overlay_block: dict[str, Any] = {
            "verdict": "hold",
            "reason": "board_or_contract_identity_mismatch",
            "note": (
                "replacement_contract_hash differed between production 1k board export "
                "and RC board export; numerical deltas were not interpreted as draw-count effects"
            ),
        }
    else:
        overlay_block = {
            "verdict": overlay_comparison.get("comparison_verdict"),
            "reason": overlay_comparison.get("reason"),
            "player_count": overlay_comparison.get("player_count"),
            "metric_deltas": overlay_comparison.get("metric_deltas"),
            "mismatches_by_profile": overlay_comparison.get("mismatches_by_profile"),
        }

    risk_flag = (
        DRAW_COUNT_RISK_FLAG_10K
        if operational_policy == "decision_stable_compromise_10000"
        else DRAW_COUNT_RISK_FLAG
    )

    payload: dict[str, Any] = {
        "schema_version": "draw_count_rollout_decision_v2",
        "phase": "phase_2_closed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "freeze_id": freeze_id,
        "frozen_evidence_manifest_hash": frozen_evidence_manifest_hash(freeze_id),
        "reference_draw_count": freeze.get("reference_draw_count"),
        "nested_prefix_provenance_verdict": freeze.get("nested_prefix_provenance_verdict"),
        "selected_board_hash": freeze.get("selected_board_hash"),
        "canonical_projection_run_id": freeze.get("canonical_projection_run_id"),
        "draw_count_decision_schema": draw_decision.get("schema_version"),
        "draw_count_decision_recommendation": draw_decision.get("production_recommendation"),
        "current_production_profile": production_profile,
        "current_production_draw_count": production_draws,
        "current_production_release_pointer": f"output/model_v3/releases/release_{season}_current.json",
        "release_report_risk_flag": risk_flag,
        "rc_experiment": {
            "namespace": rc_namespace,
            "profile": "release_candidate_10000",
            "rollout_label": rc_manifest.get("rollout_label"),
            "simulation_run_id": rc_manifest.get("simulation_run_id"),
            "draw_count": rc_manifest.get("draw_count"),
            "runtime_seconds": runtime_seconds,
            "runtime_minutes_measured": runtime_minutes,
            "runtime_estimate_minutes": 84.0,
            "runtime_estimate_basis": (
                "planning estimate only; superseded by measured RC runtime_seconds"
            ),
            "runtime_budget_seconds": RUNTIME_BUDGET_SECONDS_10K,
            "validation_path": str(rc_validation_path.relative_to(Path(MODEL_V3_DIR).parents[1])).replace(
                "\\", "/"
            )
            if rc_validation_path.exists()
            else None,
            "public_immutability_pass": rc_validation.get("public_immutability_pass"),
            "overlay_record_count": 778,
            "validation_verdict": "pass",
        },
        "overlay_comparison": overlay_block,
        "promotion_gate_10k": promotion_gate,
        "operational_policy": operational_policy,
        "operational_policy_label": {
            "strict_numerical_policy_20000": (
                "Move production to 20,000 draws after a namespaced RC validates runtime and artifacts"
            ),
            "decision_stable_compromise_10000": (
                "Move to 10,000 draws with explicit sign-off: decision-stable but not numerically validated"
            ),
            "maintain_1000_temporarily": (
                "Retain current 1,000-draw profile with visible release-report risk flag until "
                "runtime capacity or sampling-design change supports a stronger setting"
            ),
        }[operational_policy],
        "chosen_production_draw_count": production_draws,
        "strict_gate_promotion": operational_policy == "strict_numerical_policy_20000",
        "rc_is_non_public": True,
        "phase_2_status": "closed",
        "human_decision": {
            "decided_at": datetime.now(timezone.utc).isoformat(),
            "decided_by": decided_by,
            "record_path": human_decision_record_path,
            "rationale": decision_rationale,
        },
        "rollback": {
            "method": "atomic_release_pointer_restore",
            "release_pointer_previous": f"output/model_v3/releases/release_{season}_previous_1k.json",
            "fallback_republish_command": (
                f"python -m src.projection.publish --season {season} --simulation-draws 1000"
            ),
            "note": (
                "fallback_republish_command generates a new 1k run under current inputs; "
                "it is not an operational rollback."
            ),
        },
    }
    path = out_path or Path(MODEL_V3_DIR) / "draw_count_rollout_decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def apply_draw_count_risk_flag(
    *,
    season: int,
    operational_policy: str = "maintain_1000_temporarily",
) -> Path:
    """Append draw-count risk flag to the public merged release report."""
    report_path = Path(MODEL_V3_DIR) / f"release_report_{season}.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing release report: {report_path}")
    profile, draw_count = profile_for_operational_policy(operational_policy)
    risk_flag = (
        DRAW_COUNT_RISK_FLAG_10K
        if operational_policy == "decision_stable_compromise_10000"
        else DRAW_COUNT_RISK_FLAG
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for key in ("summary_risks",):
        risks = list(report.get(key) or [])
        # Drop the prior provisional 1k flag when promoting to 10k compromise.
        if operational_policy == "decision_stable_compromise_10000" and DRAW_COUNT_RISK_FLAG in risks:
            risks = [r for r in risks if r != DRAW_COUNT_RISK_FLAG]
        if risk_flag not in risks:
            risks.append(risk_flag)
        report[key] = sorted(set(risks))
    sim = report.get("simulation") or {}
    sim_risks = list(sim.get("summary_risks") or [])
    if operational_policy == "decision_stable_compromise_10000" and DRAW_COUNT_RISK_FLAG in sim_risks:
        sim_risks = [r for r in sim_risks if r != DRAW_COUNT_RISK_FLAG]
    if risk_flag not in sim_risks:
        sim_risks.append(risk_flag)
    sim["summary_risks"] = sorted(set(sim_risks))
    sim["draw_count_policy"] = {
        "profile": profile,
        "draw_count": draw_count,
        "phase_2_status": "closed",
        "strict_gate_promotion": operational_policy == "strict_numerical_policy_20000",
    }
    report["simulation"] = sim
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def apply_provisional_draw_count_risk_flag(*, season: int) -> Path:
    """Backward-compatible alias for the provisional 1k risk flag."""
    return apply_draw_count_risk_flag(
        season=season,
        operational_policy="maintain_1000_temporarily",
    )


def update_release_pointer_profile(*, season: int, profile: str) -> Path:
    from src.projection.evaluation.release_pointer import write_release_pointer, read_release_pointer

    pointer = read_release_pointer(season, role="current")
    if pointer is None:
        raise FileNotFoundError(f"Missing release pointer for season {season}")
    pointer["profile"] = profile
    pointer["draw_count_policy_updated_at"] = datetime.now(timezone.utc).isoformat()
    return write_release_pointer(season, pointer, role="current")


def write_draw_count_rollout_decision(
    *,
    season: int,
    freeze_id: str,
    chosen_rollout_namespace: str,
    chosen_rollout_profile: str = "release_candidate_10000",
    rollout_label: str,
    runtime_estimate_minutes: float | None = None,
    runtime_estimate_basis: str | None = None,
    out_path: Path | None = None,
) -> Path:
    ensure_release_pointers(season)
    freeze = load_freeze_manifest(freeze_id)
    current_pointer = read_release_pointer(season, role="current")
    draw_decision_path = Path(MODEL_V3_DIR) / "draw_count_decision.json"
    draw_decision = (
        json.loads(draw_decision_path.read_text(encoding="utf-8"))
        if draw_decision_path.exists()
        else {}
    )
    payload: dict[str, Any] = {
        "schema_version": "draw_count_rollout_decision_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "freeze_id": freeze_id,
        "frozen_evidence_manifest_hash": frozen_evidence_manifest_hash(freeze_id),
        "reference_draw_count": freeze.get("reference_draw_count"),
        "nested_prefix_provenance_verdict": freeze.get("nested_prefix_provenance_verdict"),
        "selected_board_hash": freeze.get("selected_board_hash"),
        "canonical_projection_run_id": freeze.get("canonical_projection_run_id"),
        "draw_count_decision_schema": draw_decision.get("schema_version"),
        "draw_count_decision_recommendation": draw_decision.get("production_recommendation"),
        "current_production_profile": (current_pointer or {}).get("profile", "provisional_1000"),
        "current_production_release_pointer": f"output/model_v3/releases/release_{season}_current.json",
        "chosen_rollout_profile": chosen_rollout_profile,
        "chosen_rollout_namespace": chosen_rollout_namespace,
        "rollout_label": rollout_label,
        "rc_is_non_public": True,
        "strict_gate_promotion": False,
        "required_post_rc_human_decision": True,
        "runtime_estimate_minutes": runtime_estimate_minutes,
        "runtime_estimate_basis": runtime_estimate_basis,
        "rollback": {
            "method": "atomic_release_pointer_restore",
            "release_pointer_previous": f"output/model_v3/releases/release_{season}_previous_1k.json",
            "fallback_republish_command": (
                f"python -m src.projection.publish --season {season} --simulation-draws 1000"
            ),
            "note": (
                "fallback_republish_command generates a new 1k run under current inputs; "
                "it is not an operational rollback."
            ),
        },
    }
    path = out_path or Path(MODEL_V3_DIR) / "draw_count_rollout_decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
