"""Tamper tests for the six mandatory promotion invariants."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.projection.active_release import build_active_pointer, pointer_path, read_active_pointer, write_active_pointer
from src.projection.evaluation.promotion_invariants import validate_promotion_invariants
from src.projection.promote_release import PromoteReleaseError, promote_release
from src.projection.release_bundle import bundle_root, canonical_dumps, load_sealed_manifest, public_release_dir
from tests.fixtures.release_bundle_v2 import seal_v2_bundle


SOURCE_COMMIT = "abc123def4567890abcdef1234567890abcdef12"


def _patch_roots(tmp_path: Path, monkeypatch, *, source_commit: str = SOURCE_COMMIT) -> None:
    model_v3 = tmp_path / "model_v3"
    model_v3.mkdir()
    monkeypatch.setattr("src.projection.release_bundle.MODEL_V3_DIR", str(model_v3))
    monkeypatch.setattr("src.projection.release_bundle.REPO_ROOT", str(tmp_path))
    monkeypatch.setattr("src.projection.active_release.REPO_ROOT", str(tmp_path))
    monkeypatch.setattr("src.projection.git_provenance.REPO_ROOT", str(tmp_path))
    monkeypatch.setattr("src.projection.git_provenance.working_tree_dirty", lambda **_: False)
    monkeypatch.setattr("src.projection.git_provenance.current_head_commit", lambda **_: source_commit)


def test_v1_bundle_is_not_promotion_eligible(tmp_path, monkeypatch):
    from src.projection.release_bundle import SCHEMA_VERSION, player_id_set_hash, selected_points_vector_hash, treatment_block
    from src.projection.release_bundle_publish import seal_staged_bundle

    _patch_roots(tmp_path, monkeypatch)
    root = bundle_root(2026, "legacy_ns")
    root.mkdir(parents=True, exist_ok=True)
    selected = "player_id,fantasy_pts_season\na,100\n"
    (root / "selected_board.csv").write_text(selected, encoding="utf-8")
    (root / "players_2026.json").write_text(
        json.dumps({"meta": {"model_id": "accuracy_first_ensemble"}, "players": [{"player_id": "a"}]}),
        encoding="utf-8",
    )
    for name in (
        "team_stats_2026.json",
        "comparison_2026.json",
        "release_report_2026.json",
        "release_report_simulation_2026.json",
        "release_report_board_2026.json",
    ):
        (root / name).write_text("{}", encoding="utf-8")
    (root / "application_contract.json").write_text(json.dumps({"contract_hash": "a" * 64}), encoding="utf-8")
    (root / "simulation_manifest_2026.json").write_text(
        json.dumps({"draw_count": 10000, "simulation_run_id": "sim-1"}), encoding="utf-8"
    )
    board_hash = hashlib.sha256(selected.encode()).hexdigest()
    seal_staged_bundle(
        season=2026,
        namespace="legacy_ns",
        root=root,
        release_id="rel-legacy",
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
        contract_treatments={
            "selected": treatment_block(["a"]),
            "incumbent": treatment_block([]),
            "new_player_v1_only": treatment_block([]),
        },
        artifact_specs=[
            ("selected_board", "selected_board.csv", True, False),
            ("players", "players_2026.json", True, True),
            ("team_stats", "team_stats_2026.json", True, True),
            ("comparison", "comparison_2026.json", True, True),
            ("release_report", "release_report_2026.json", True, False),
            ("release_report_simulation", "release_report_simulation_2026.json", True, False),
            ("release_report_board", "release_report_board_2026.json", True, False),
            ("application_contract", "application_contract.json", True, False),
            ("simulation_manifest", "simulation_manifest_2026.json", True, False),
        ],
        schema_version=SCHEMA_VERSION,
    )
    report = validate_promotion_invariants(season=2026, namespace="legacy_ns")
    assert report["verdict"] == "fail"


def test_all_invariants_pass_for_v2_bundle(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch)
    seal_v2_bundle(tmp_path, "good_ns")
    report = validate_promotion_invariants(season=2026, namespace="good_ns")
    assert report["verdict"] == "pass"


@pytest.mark.parametrize(
    "tamper,expected_check",
    [
        ("overlay_manifest", "overlay_coverage_alignment"),
        ("overlay_report", "overlay_coverage_alignment"),
        ("board_manifest", "selected_board_hash_alignment"),
        ("board_players_meta", "selected_board_hash_alignment"),
        ("board_sim_manifest", "selected_board_hash_alignment"),
        ("board_report", "selected_board_hash_alignment"),
        ("profile_manifest", "simulation_profile_identity"),
        ("profile_sim_manifest", "simulation_profile_identity"),
        ("profile_report", "simulation_profile_identity"),
        ("policy_tamper", "simulation_profile_identity"),
        ("config_tamper", "simulation_profile_identity"),
        ("v2_hash", "ensemble_source_provenance"),
        ("adp_hash", "ensemble_source_provenance"),
        ("missing_v2_key", "ensemble_source_provenance"),
        ("browser_public", "browser_artifact_completeness"),
        ("git_dirty", "git_provenance"),
        ("commit_mismatch", "git_provenance"),
    ],
)
def test_invariant_tamper_blocks_promotion(tmp_path, monkeypatch, tamper, expected_check):
    _patch_roots(tmp_path, monkeypatch, source_commit=SOURCE_COMMIT)
    seal_v2_bundle(tmp_path, "tamper_ns", source_commit=SOURCE_COMMIT)
    root = bundle_root(2026, "tamper_ns")

    if tamper == "overlay_manifest":
        manifest, _ = load_sealed_manifest(root)
        manifest["overlay_coverage"]["total_players"] = 999
        (root / "release_bundle_manifest.json").write_bytes(canonical_dumps(manifest))
    elif tamper == "overlay_report":
        report = json.loads((root / "release_report_2026.json").read_text(encoding="utf-8"))
        report["overlay_coverage"]["total_players"] = 999
        (root / "release_report_2026.json").write_text(json.dumps(report), encoding="utf-8")
    elif tamper == "board_manifest":
        manifest, _ = load_sealed_manifest(root)
        del manifest["board"]["selected_board_sha256"]
        (root / "release_bundle_manifest.json").write_bytes(canonical_dumps(manifest))
    elif tamper == "board_players_meta":
        players = json.loads((root / "players_2026.json").read_text(encoding="utf-8"))
        del players["meta"]["selected_board_sha256"]
        (root / "players_2026.json").write_text(json.dumps(players), encoding="utf-8")
    elif tamper == "board_sim_manifest":
        sim = json.loads((root / "simulation_manifest_2026.json").read_text(encoding="utf-8"))
        del sim["selected_board_sha256"]
        (root / "simulation_manifest_2026.json").write_text(json.dumps(sim), encoding="utf-8")
    elif tamper == "board_report":
        report = json.loads((root / "release_report_2026.json").read_text(encoding="utf-8"))
        del report["board"]["selected_board_sha256"]
        (root / "release_report_2026.json").write_text(json.dumps(report), encoding="utf-8")
    elif tamper == "profile_manifest":
        manifest, _ = load_sealed_manifest(root)
        del manifest["simulation"]["policy_hash"]
        (root / "release_bundle_manifest.json").write_bytes(canonical_dumps(manifest))
    elif tamper == "profile_sim_manifest":
        sim = json.loads((root / "simulation_manifest_2026.json").read_text(encoding="utf-8"))
        del sim["chunk_size"]
        (root / "simulation_manifest_2026.json").write_text(json.dumps(sim), encoding="utf-8")
    elif tamper == "profile_report":
        report = json.loads((root / "release_report_2026.json").read_text(encoding="utf-8"))
        del report["simulation"]["configuration_hash"]
        (root / "release_report_2026.json").write_text(json.dumps(report), encoding="utf-8")
    elif tamper == "policy_tamper":
        (root / "draw_count_rollout_decision.json").write_text("{}", encoding="utf-8")
    elif tamper == "config_tamper":
        (root / "simulation_config.json").write_text('{"profiles":{"publish":{"draws":1,"chunk_size":1}}}', encoding="utf-8")
    elif tamper == "v2_hash":
        (root / "model_v2_fantasy_points_2026.csv").write_text("tampered", encoding="utf-8")
    elif tamper == "adp_hash":
        (root / "consensus_2026.json").write_text("{}", encoding="utf-8")
    elif tamper == "missing_v2_key":
        contract = json.loads((root / "application_contract.json").read_text(encoding="utf-8"))
        contract["source_hashes"].pop("v2_points_2026")
        (root / "application_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    elif tamper == "browser_public":
        public = public_release_dir("tamper_ns")
        (public / "players_2026.json").write_text("{}", encoding="utf-8")
    elif tamper == "git_dirty":
        monkeypatch.setattr("src.projection.git_provenance.working_tree_dirty", lambda **_: True)
    elif tamper == "commit_mismatch":
        monkeypatch.setattr("src.projection.git_provenance.current_head_commit", lambda **_: "deadbeef" * 5)

    report = validate_promotion_invariants(season=2026, namespace="tamper_ns")
    assert report["verdict"] == "fail"
    direct_checks = {
        "overlay_manifest",
        "git_dirty",
        "commit_mismatch",
        "browser_public",
    }
    if tamper in direct_checks:
        failed = [check for check in report["checks"] if check["check"] == expected_check]
        assert failed and not failed[0]["passed"]
    if tamper != "browser_public":
        with pytest.raises(PromoteReleaseError):
            promote_release(2026, "tamper_ns")


def test_promotion_rehashes_copied_browser_files(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch, source_commit=SOURCE_COMMIT)
    seal_v2_bundle(tmp_path, "rehash_ns", source_commit=SOURCE_COMMIT)
    public = public_release_dir("rehash_ns")
    if public.exists():
        import shutil

        shutil.rmtree(public)
    promote_release(2026, "rehash_ns")
    manifest, _ = load_sealed_manifest(bundle_root(2026, "rehash_ns"))
    players_entry = next(entry for entry in manifest["artifacts"] if entry["role"] == "players")
    copied = public / players_entry["path"]
    assert copied.is_file()
    from src.projection.evaluation.accuracy_first import sha256_file

    assert sha256_file(copied) == players_entry["sha256"]


def test_browser_copy_failure_restores_public_namespace_and_pointer(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch, source_commit=SOURCE_COMMIT)
    seal_v2_bundle(tmp_path, "copy_fail_ns", source_commit=SOURCE_COMMIT)
    pointer = build_active_pointer(
        season=2026,
        namespace="live_ns",
        release_id="live",
        manifest_sha256="f" * 64,
    )
    write_active_pointer(pointer)
    public = public_release_dir("copy_fail_ns")
    public.mkdir(parents=True, exist_ok=True)
    marker = public / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    before_pointer = pointer_path(2026).read_bytes()

    with patch(
        "src.projection.evaluation.promotion_invariants.validate_browser_artifacts_in_directory",
        return_value={"check": "browser_artifact_completeness", "passed": False, "mismatches": ["forced"]},
    ):
        with pytest.raises(PromoteReleaseError):
            promote_release(2026, "copy_fail_ns")

    assert marker.read_text(encoding="utf-8") == "keep"
    assert pointer_path(2026).read_bytes() == before_pointer
    assert read_active_pointer(2026)["namespace"] == "live_ns"


def test_simulation_profile_resolution_failure_is_not_swallowed(monkeypatch, tmp_path):
    from src.projection.simulation_profile_resolver import resolve_simulation_profile_identity

    with pytest.raises(FileNotFoundError):
        resolve_simulation_profile_identity(
            rollout_path=tmp_path / "missing_rollout.json",
            simulation_config_path_arg=tmp_path / "missing_config.json",
        )


def test_public_promotion_api_has_no_skip_git():
    from src.projection import promote_release as promote_module
    from src.projection.evaluation import promotion_invariants as invariants_module

    assert "skip_git" not in inspect.signature(promote_module.promote_release).parameters
    assert "skip_git" not in inspect.signature(promote_module.rollback_release).parameters
    assert "skip_git" not in inspect.signature(invariants_module.validate_promotion_invariants).parameters
