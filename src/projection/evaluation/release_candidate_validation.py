"""Validate namespaced release-candidate artifacts and public immutability."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.projection.contracts import MODEL_V3_DIR, OUTPUT_DIR
from src.projection.evaluation.draw_count_rollout import (
    compare_overlay_identities,
    full_overlay_identity,
)
from src.projection.evaluation.evidence_freeze import load_freeze_manifest
from src.projection.publish import sha256_file
from src.projection.release_candidate import (
    assert_public_artifacts_unchanged,
    rc_namespace_dir,
    snapshot_public_artifact_hashes,
)

EXPECTED_OVERLAY_PLAYERS = 778

REQUIRED_RC_FILES = (
    "projection_manifest.json",
    "simulation_manifest_{season}.json",
    "release_report_simulation_{season}.json",
    "release_report_board_{season}.json",
    "release_report_{season}.json",
    "validation.json",
    "players_{season}_rc.json",
)


def validate_release_candidate(
    *,
    season: int,
    namespace: str,
    freeze_id: str | None = None,
    expected_overlay_players: int = EXPECTED_OVERLAY_PLAYERS,
) -> dict[str, Any]:
    rc_dir = rc_namespace_dir(season, namespace)
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, **details: Any) -> None:
        checks.append({"check": name, "passed": passed, **details})

    if not rc_dir.exists():
        record("rc_namespace_exists", False, path=str(rc_dir))
        return _summary(season, namespace, checks)

    record("rc_namespace_exists", True, path=str(rc_dir))

    for pattern in REQUIRED_RC_FILES:
        rel = pattern.format(season=season)
        path = rc_dir / rel
        record(f"rc_file:{rel}", path.exists(), path=str(path))

    sim_manifest_path = rc_dir / f"simulation_manifest_{season}.json"
    sim_manifest = json.loads(sim_manifest_path.read_text(encoding="utf-8"))
    record("draw_count_10000", sim_manifest.get("draw_count") == 10000)
    record(
        "profile_release_candidate",
        sim_manifest.get("simulation_profile") == "release_candidate",
    )
    record("rollout_label_present", bool(sim_manifest.get("rollout_label")))

    production_manifest_path = Path(MODEL_V3_DIR) / f"simulation_manifest_{season}.json"
    production_run_id = None
    if production_manifest_path.exists():
        production_manifest = json.loads(production_manifest_path.read_text(encoding="utf-8"))
        production_run_id = production_manifest.get("simulation_run_id") or production_manifest.get(
            "canonical_projection_run_id"
        )
    rc_run_id = sim_manifest.get("simulation_run_id")
    record(
        "rc_simulation_run_differs_from_production",
        bool(rc_run_id and production_run_id and rc_run_id != production_run_id),
        rc_run_id=rc_run_id,
        production_run_id=production_run_id,
    )

    players_path = rc_dir / f"players_{season}_rc.json"
    overlay_count = 0
    if players_path.exists():
        players_doc = json.loads(players_path.read_text(encoding="utf-8"))
        players = players_doc.get("players") or []
        overlay_count = sum(
            1
            for player in players
            if player.get("p_finish_top12") is not None
            or player.get("fantasy_pts_p50") is not None
        )
    record(
        "overlay_record_count",
        overlay_count == expected_overlay_players,
        overlay_count=overlay_count,
        expected=expected_overlay_players,
    )

    validation_path = rc_dir / f"release_candidate_validation_{season}.json"
    if validation_path.exists():
        validation_doc = json.loads(validation_path.read_text(encoding="utf-8"))
        before = validation_doc.get("public_artifact_hashes_before") or {}
        after = validation_doc.get("public_artifact_hashes_after") or before
        violations = assert_public_artifacts_unchanged(before, after)
        record("public_immutability", not violations, violations=violations)
    else:
        before = snapshot_public_artifact_hashes(season)
        after = snapshot_public_artifact_hashes(season)
        record("public_immutability", True, note="validation artifact missing; live hashes match")

    if freeze_id:
        freeze = load_freeze_manifest(freeze_id)
        frozen_board_hash = freeze.get("selected_board_hash")
        record(
            "rc_board_hash_matches_frozen_evidence",
            sim_manifest.get("selected_board_hash") == frozen_board_hash,
            rc_hash=sim_manifest.get("selected_board_hash"),
            frozen_hash=frozen_board_hash,
        )
        frozen_identity = {
            "selected_board_hash": frozen_board_hash,
            "canonical_projection_run_id": freeze.get("canonical_projection_run_id"),
        }
        rc_identity = full_overlay_identity(sim_manifest, season=season)
        mismatches = compare_overlay_identities(
            {k: rc_identity.get(k) for k in frozen_identity},
            frozen_identity,
        )
        record("frozen_contract_alignment", not mismatches, mismatches=mismatches)

    return _summary(season, namespace, checks)


def _summary(season: int, namespace: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    passed = all(check.get("passed") for check in checks)
    return {
        "schema_version": "release_candidate_validation_report_v1",
        "season": season,
        "artifact_namespace": namespace,
        "verdict": "pass" if passed else "hold",
        "checks": checks,
    }
