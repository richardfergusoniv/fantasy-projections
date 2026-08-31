"""Bounded shadow-only repair candidates for v1 RB/WR (never mutates production)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.projection.evaluation.accuracy_first import canonical_json_hash, sha256_file
from src.projection.shadow.contracts import (
    CLOSEOUT_SCHEMA,
    SHADOW_V1_RB_WR_DIR,
    SHADOW_V1_REPAIR_TRACK_CLOSED,
)
from src.projection.shadow.decision_rules import repair_gate
from src.projection.shadow.forbidden import assert_no_forbidden_imports
from src.projection.shadow.production_guard import (
    assert_production_unchanged,
    snapshot_production_artifacts,
)

SHADOW_OUTPUT_DIR = SHADOW_V1_RB_WR_DIR
REPAIR_ENTRYPOINTS = (
    "src.projection.shadow.repair",
    "src.projection.shadow.rb_wr_attribution",
    "src.projection.shadow.decision_rules",
)

FREEZE_SCHEMA = "shadow_v1_rb_wr_candidate_freeze_v1"


def evaluate_repair_track_authorization(
    policy_path: str | Path | None = None,
) -> dict[str, Any]:
    """Authorize freeze only via an explicit, hash-valid reopen record.

    Absent, malformed, hash-mismatched, or closed seals are fail-closed
    (not authorized). Deleting the seal does not reopen the track.
    """
    path = Path(policy_path or SHADOW_V1_REPAIR_TRACK_CLOSED)
    if not path.is_file():
        return {
            "authorized": False,
            "reason": "repair_track_closed",
            "policy_detail": "policy_absent",
            "policy_path": str(path).replace("\\", "/"),
            "policy": None,
        }
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "authorized": False,
            "reason": "repair_track_closed",
            "policy_detail": "policy_malformed",
            "policy_path": str(path).replace("\\", "/"),
            "policy": None,
            "error": str(exc),
        }
    if not isinstance(raw, dict):
        return {
            "authorized": False,
            "reason": "repair_track_closed",
            "policy_detail": "policy_malformed",
            "policy_path": str(path).replace("\\", "/"),
            "policy": None,
        }

    recorded = raw.get("artifact_hash")
    body = {k: v for k, v in raw.items() if k != "artifact_hash"}
    expected = canonical_json_hash(body)
    if not isinstance(recorded, str) or recorded != expected:
        return {
            "authorized": False,
            "reason": "repair_track_closed",
            "policy_detail": "policy_hash_mismatch",
            "policy_path": str(path).replace("\\", "/"),
            "policy": raw,
        }
    if raw.get("schema_version") != CLOSEOUT_SCHEMA:
        return {
            "authorized": False,
            "reason": "repair_track_closed",
            "policy_detail": "policy_schema_invalid",
            "policy_path": str(path).replace("\\", "/"),
            "policy": raw,
        }
    if raw.get("further_repair_authorized") is not True:
        return {
            "authorized": False,
            "reason": "repair_track_closed",
            "policy_detail": "further_repair_not_authorized",
            "policy_path": str(path).replace("\\", "/"),
            "policy": raw,
        }

    reopen = raw.get("reopen")
    if not isinstance(reopen, dict):
        return {
            "authorized": False,
            "reason": "repair_track_closed",
            "policy_detail": "reopen_evidence_missing",
            "policy_path": str(path).replace("\\", "/"),
            "policy": raw,
        }
    defect = str(reopen.get("cutoff_available_defect") or "").strip()
    refs = reopen.get("evidence_refs")
    if not defect or not isinstance(refs, list) or not refs:
        return {
            "authorized": False,
            "reason": "repair_track_closed",
            "policy_detail": "reopen_evidence_incomplete",
            "policy_path": str(path).replace("\\", "/"),
            "policy": raw,
        }

    return {
        "authorized": True,
        "reason": "repair_authorized",
        "policy_detail": "explicit_reopen",
        "policy_path": str(path).replace("\\", "/"),
        "policy": raw,
    }


def _hold_payload(
    *,
    candidate_id: str,
    reason: str,
    attribution_status: str | None,
    policy: dict[str, Any],
    note: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": FREEZE_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "verdict": "hold_v1_structural_role",
        "gate": {"passed": False, "reason": reason, "verdict": "hold_v1_structural_role"},
        "attribution_status": attribution_status,
        "repair_track_policy": {
            "reason": policy.get("reason"),
            "policy_detail": policy.get("policy_detail"),
            "policy_path": policy.get("policy_path"),
            "authorized": bool(policy.get("authorized")),
        },
        "production_weights_unchanged": True,
        "promotion_authorized": False,
        "note": note,
    }
    if extra:
        payload.update(extra)
    payload["artifact_hash"] = canonical_json_hash(payload)
    return payload


def freeze_shadow_candidate(
    *,
    candidate_id: str,
    code_identity: dict[str, Any],
    evidence: dict[str, Any],
    source_hashes: dict[str, str],
    board_2026_path: str | Path | None = None,
    fold_mae_relative_deltas: list[float],
    pooled_top120_spearman_baseline: float,
    pooled_top120_spearman_candidate: float,
    all_eligible_ok: bool = True,
    coverage_unchanged: bool = True,
    team_identity_unchanged: bool = True,
    out_dir: str | Path | None = None,
    attribution_status: str | None = None,
    policy_path: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze one shadow candidate if authorized and it clears the repair gate.

    Fail-closed when the canonical repair-track policy is absent, invalid, or
    closed. Candidate freezing is also prohibited when attribution is incomplete.
    Production artifacts are snapshotted and must be unchanged afterward.
    """
    assert_no_forbidden_imports(REPAIR_ENTRYPOINTS)
    before = snapshot_production_artifacts()
    dest = Path(out_dir or SHADOW_OUTPUT_DIR)
    dest.mkdir(parents=True, exist_ok=True)

    policy = evaluate_repair_track_authorization(policy_path)
    if not policy["authorized"]:
        payload = _hold_payload(
            candidate_id=candidate_id,
            reason="repair_track_closed",
            attribution_status=attribution_status,
            policy=policy,
            note=(
                "Shadow RB/WR repair track is closed or policy is invalid; "
                "candidate freezing requires an explicit hash-valid authorization "
                "record backed by cutoff-available defect evidence."
            ),
        )
        path = dest / "hold_v1_structural_role.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        assert_production_unchanged(before)
        return payload

    if attribution_status is None:
        manifest_path = dest / "manifest.json"
        if manifest_path.is_file():
            attribution_status = json.loads(
                manifest_path.read_text(encoding="utf-8")
            ).get("status")
    if attribution_status != "ok":
        payload = _hold_payload(
            candidate_id=candidate_id,
            reason="attribution_incomplete",
            attribution_status=attribution_status,
            policy=policy,
            note="Candidate freezing prohibited until attribution status is ok.",
        )
        path = dest / "hold_v1_structural_role.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        assert_production_unchanged(before)
        return payload

    gate = repair_gate(
        fold_mae_deltas=fold_mae_relative_deltas,
        pooled_top120_spearman_baseline=pooled_top120_spearman_baseline,
        pooled_top120_spearman_candidate=pooled_top120_spearman_candidate,
        all_eligible_ok=all_eligible_ok,
        coverage_unchanged=coverage_unchanged,
        team_identity_unchanged=team_identity_unchanged,
    )
    board_hash = None
    if board_2026_path is not None and Path(board_2026_path).is_file():
        board_hash = sha256_file(board_2026_path)

    payload: dict[str, Any] = {
        "schema_version": FREEZE_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "code_identity": code_identity,
        "evidence": evidence,
        "source_hashes": source_hashes,
        "board_2026_sha256": board_hash,
        "gate": gate,
        "attribution_status": attribution_status,
        "repair_track_policy": {
            "reason": policy.get("reason"),
            "policy_detail": policy.get("policy_detail"),
            "policy_path": policy.get("policy_path"),
            "authorized": True,
        },
        "production_weights_unchanged": True,
        "promotion_authorized": False,
        "note": (
            "Production RB/WR weights remain under accuracy-first snapshot "
            "equality; shadow freeze never mutates production artifacts."
        ),
    }
    if gate["passed"]:
        path = dest / f"freeze_{candidate_id}.json"
    else:
        path = dest / "hold_v1_structural_role.json"
        payload["verdict"] = "hold_v1_structural_role"
    payload["artifact_hash"] = canonical_json_hash(payload)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert_production_unchanged(before)
    return payload
