"""Release-bundle manifest schema, path confinement, hashing, and enumeration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.projection.release_bundle import (
    BUNDLE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ReleaseBundleError,
    artifact_record,
    build_manifest,
    canonical_dumps,
    canonical_hash,
    confine_namespace_path,
    enumerate_namespace_files,
    load_sealed_manifest,
    player_id_set_hash,
    seal_manifest,
    selected_points_vector_hash,
    treatment_block,
    validate_artifact_enumeration,
    validate_manifest_schema,
    verify_artifact_hashes,
    verify_provenance_identities,
)


def _empty_treatment():
    return treatment_block([])


def _minimal_artifacts(tmp_path: Path) -> list[dict]:
    selected = tmp_path / "selected_board.csv"
    report = tmp_path / "release_report.json"
    selected.write_text("player_id,fantasy_pts_season\na,100\n", encoding="utf-8")
    report.write_text("{}", encoding="utf-8")
    return [
        artifact_record(
            role="selected_board",
            path="selected_board.csv",
            file_path=selected,
            required=True,
            browser_consumed=False,
        ),
        artifact_record(
            role="release_report",
            path="release_report.json",
            file_path=report,
            required=True,
            browser_consumed=False,
            media_type="application/json",
        ),
    ]


def _minimal_payload(tmp_path: Path, artifacts: list[dict] | None = None) -> dict:
    artifacts = artifacts or _minimal_artifacts(tmp_path)
    board_hash = next(entry["sha256"] for entry in artifacts if entry["role"] == "selected_board")
    return {
        "schema_version": SCHEMA_VERSION,
        "bundle": {
            "season": 2026,
            "namespace": "test_ns",
            "release_id": "rel-1",
            "created_at": "2026-08-29T00:00:00+00:00",
            "model_id": "accuracy_first_ensemble",
            "schema_version": BUNDLE_SCHEMA_VERSION,
        },
        "application": {
            "contract_version": "accuracy_first_2026_v1",
            "contract_hash": "a" * 64,
        },
        "runs": {
            "projection_run_id": "proj-1",
            "simulation_run_id": "sim-1",
        },
        "board": {
            "selected_board_file_hash": board_hash,
            "selected_points_vector_hash": selected_points_vector_hash({"a": 100.0}),
        },
        "simulation": {
            "profile": "publish",
            "draw_count": 10000,
            "configuration_hash": "b" * 64,
            "calibration_hashes": {"v3_interval": "c" * 64},
            "joint_donor_hash": "d" * 64,
        },
        "overlay": {
            "simulated_player_population_hash": player_id_set_hash(["a"]),
            "simulated_player_count": 1,
        },
        "artifacts": artifacts,
        "contract_treatments": {
            "selected": treatment_block(["a"]),
            "incumbent": _empty_treatment(),
            "new_player_v1_only": _empty_treatment(),
        },
    }


def test_confine_rejects_parent_absolute_and_backslash():
    assert confine_namespace_path("players_2026.json") == "players_2026.json"
    assert confine_namespace_path("simulations/part-00000.parquet") == "simulations/part-00000.parquet"
    with pytest.raises(ReleaseBundleError, match="posix"):
        confine_namespace_path(r"foo\\bar.json")
    with pytest.raises(ReleaseBundleError, match="unsafe"):
        confine_namespace_path("../secret.json")
    with pytest.raises(ReleaseBundleError, match="unsafe"):
        confine_namespace_path("foo/../bar.json")
    with pytest.raises(ReleaseBundleError, match="namespace-relative"):
        confine_namespace_path("/etc/passwd")
    with pytest.raises(ReleaseBundleError, match="unsafe"):
        confine_namespace_path("foo//bar.json")
    with pytest.raises(ReleaseBundleError, match="non-empty"):
        confine_namespace_path("")


def test_canonical_dumps_is_byte_stable_across_key_order():
    left = {"b": 1, "a": {"z": 2, "y": 3}}
    right = {"a": {"y": 3, "z": 2}, "b": 1}
    assert canonical_dumps(left) == canonical_dumps(right)
    assert canonical_dumps(left) == b'{"a":{"y":3,"z":2},"b":1}'
    assert canonical_hash(left) == hashlib.sha256(canonical_dumps(left)).hexdigest()


def test_schema_rejects_mutable_status_and_missing_sections(tmp_path: Path):
    payload = _minimal_payload(tmp_path)
    validate_manifest_schema(payload)
    payload["status"] = "active"
    with pytest.raises(ReleaseBundleError, match="mutable status"):
        validate_manifest_schema(payload)
    del payload["status"]
    del payload["overlay"]
    with pytest.raises(ReleaseBundleError, match="missing section"):
        validate_manifest_schema(payload)


def test_duplicate_roles_and_paths_are_rejected(tmp_path: Path):
    selected = tmp_path / "a.csv"
    selected.write_text("x", encoding="utf-8")
    first = artifact_record(role="selected_board", path="a.csv", file_path=selected)
    second = dict(first)
    second["role"] = "selected_board"
    second["path"] = "b.csv"
    with pytest.raises(ReleaseBundleError, match="duplicate artifact roles"):
        validate_artifact_enumeration([first, second])
    second["role"] = "release_report"
    second["path"] = "a.csv"
    with pytest.raises(ReleaseBundleError, match="duplicate artifact paths"):
        validate_artifact_enumeration([first, second])


def test_enumeration_rejects_missing_and_unlisted_files(tmp_path: Path):
    artifacts = _minimal_artifacts(tmp_path)
    validate_artifact_enumeration(artifacts, root=tmp_path)
    (tmp_path / "extra.txt").write_text("nope", encoding="utf-8")
    with pytest.raises(ReleaseBundleError, match="unlisted"):
        validate_artifact_enumeration(artifacts, root=tmp_path)
    (tmp_path / "extra.txt").unlink()
    (tmp_path / "release_report.json").unlink()
    with pytest.raises(ReleaseBundleError, match="missing"):
        validate_artifact_enumeration(artifacts, root=tmp_path)


def test_sidecars_are_excluded_from_artifact_enumeration(tmp_path: Path):
    _minimal_artifacts(tmp_path)
    (tmp_path / "release_bundle_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "release_bundle_validation.json").write_text("{}", encoding="utf-8")
    listed = enumerate_namespace_files(tmp_path)
    assert "release_bundle_manifest.json" not in listed
    assert "release_bundle_validation.json" not in listed
    assert "selected_board.csv" in listed


def test_seal_is_canonical_and_hash_excludes_self(tmp_path: Path):
    payload = _minimal_payload(tmp_path)
    manifest, digest = seal_manifest(payload, root=tmp_path)
    raw = (tmp_path / "release_bundle_manifest.json").read_bytes()
    assert raw == canonical_dumps(manifest)
    assert digest == hashlib.sha256(raw).hexdigest()
    assert "sha256" not in manifest or manifest.get("manifest_sha256") is None
    loaded, loaded_digest = load_sealed_manifest(tmp_path)
    assert loaded == manifest
    assert loaded_digest == digest
    pretty = json.dumps(manifest, indent=2).encode("utf-8")
    (tmp_path / "release_bundle_manifest.json").write_bytes(pretty)
    with pytest.raises(ReleaseBundleError, match="canonical"):
        load_sealed_manifest(tmp_path)


def test_hash_mismatch_and_provenance_inconsistency(tmp_path: Path):
    payload = _minimal_payload(tmp_path)
    manifest, _ = seal_manifest(payload, root=tmp_path)
    selected = tmp_path / "selected_board.csv"
    selected.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ReleaseBundleError, match="hash mismatch"):
        verify_artifact_hashes(manifest, root=tmp_path)
    selected.write_text("player_id,fantasy_pts_season\na,100\n", encoding="utf-8")
    with pytest.raises(ReleaseBundleError, match="inconsistent provenance"):
        verify_provenance_identities(manifest, expected={"projection_run_id": "other"})
    verify_provenance_identities(
        manifest,
        expected={"projection_run_id": "proj-1", "simulation_run_id": "sim-1"},
    )


def test_build_manifest_round_trip(tmp_path: Path):
    artifacts = _minimal_artifacts(tmp_path)
    board_hash = artifacts[0]["sha256"]
    built = build_manifest(
        season=2026,
        namespace="test_ns",
        release_id="rel-1",
        model_id="accuracy_first_ensemble",
        created_at="2026-08-29T00:00:00+00:00",
        application={"contract_version": "accuracy_first_2026_v1", "contract_hash": "a" * 64},
        runs={"projection_run_id": "proj-1", "simulation_run_id": "sim-1"},
        board={
            "selected_board_file_hash": board_hash,
            "selected_points_vector_hash": selected_points_vector_hash({"a": 100.0}),
        },
        simulation={
            "profile": "publish",
            "draw_count": 10000,
            "configuration_hash": "b" * 64,
            "calibration_hashes": {"v3_interval": "c" * 64},
            "joint_donor_hash": "d" * 64,
        },
        overlay={
            "simulated_player_population_hash": player_id_set_hash(["a"]),
            "simulated_player_count": 1,
        },
        artifacts=artifacts,
        contract_treatments={
            "selected": treatment_block(["a"]),
            "incumbent": treatment_block([]),
            "new_player_v1_only": treatment_block([]),
        },
    )
    assert built["schema_version"] == SCHEMA_VERSION
    assert "status" not in built
