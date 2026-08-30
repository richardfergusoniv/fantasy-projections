"""The sealed bundle must pin the external v2 board it blends into the mean.

v2 is produced by a SEPARATE repository and synced in as a CSV. It carries
0.55 of the published WR mean and 0.30 of RB, yet it was the one input the
seal did not cover: swap the file, republish, and every other hash in the
chain still validated. These tests cover the pin and its backward
compatibility with bundles sealed before the pin existed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.projection.evaluation.release_bundle_validation import validate_release_bundle
from src.projection.release_bundle import (
    player_id_set_hash,
    selected_points_vector_hash,
    treatment_block,
)
from src.projection.release_bundle_publish import seal_staged_bundle

# Written as bytes: Path.write_text translates newlines on Windows, so a hash
# taken over the in-memory string would not match the file on disk.
V2_CSV = b"player_id,fantasy_pts_season\na,111\n"
SEASON = 2026


def _patch_roots(tmp_path: Path, monkeypatch) -> None:
    model_v3 = tmp_path / "model_v3"
    model_v3.mkdir()
    monkeypatch.setattr("src.projection.release_bundle.MODEL_V3_DIR", str(model_v3))
    monkeypatch.setattr("src.projection.release_bundle.REPO_ROOT", str(tmp_path))
    monkeypatch.setattr("src.projection.active_release.REPO_ROOT", str(tmp_path))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _seal(
    tmp_path: Path,
    namespace: str,
    *,
    v2_in_bundle: bool,
    v2_in_contract: bool,
) -> Path:
    from src.projection.release_bundle import bundle_root

    root = bundle_root(SEASON, namespace)
    root.mkdir(parents=True, exist_ok=True)
    _write(root / "selected_board.csv", "player_id,fantasy_pts_season\na,100\n")
    _write(
        root / f"players_{SEASON}.json",
        json.dumps({"meta": {"model_id": "accuracy_first_ensemble"}, "players": [{"player_id": "a"}]}),
    )
    _write(root / f"team_stats_{SEASON}.json", json.dumps({"players": []}))
    _write(root / f"comparison_{SEASON}.json", json.dumps({"players": []}))
    _write(root / f"release_report_{SEASON}.json", "{}")
    _write(root / f"release_report_simulation_{SEASON}.json", "{}")
    _write(root / f"release_report_board_{SEASON}.json", "{}")

    v2_rel = f"model_v2_fantasy_points_{SEASON}.csv"
    source_hashes = {}
    if v2_in_contract:
        source_hashes[f"v2_points_{SEASON}"] = hashlib.sha256(V2_CSV).hexdigest()
    _write(
        root / "application_contract.json",
        json.dumps({"contract_hash": "a" * 64, "source_hashes": source_hashes}),
    )
    _write(
        root / f"simulation_manifest_{SEASON}.json",
        json.dumps(
            {
                "draw_count": 10000,
                "simulation_run_id": "sim-1",
                "canonical_projection_run_id": "proj-1",
            }
        ),
    )

    specs = [
        ("selected_board", "selected_board.csv", True, False),
        ("players", f"players_{SEASON}.json", True, True),
        ("team_stats", f"team_stats_{SEASON}.json", True, True),
        ("comparison", f"comparison_{SEASON}.json", True, True),
        ("release_report", f"release_report_{SEASON}.json", True, False),
        ("release_report_simulation", f"release_report_simulation_{SEASON}.json", True, False),
        ("release_report_board", f"release_report_board_{SEASON}.json", True, False),
        ("application_contract", "application_contract.json", True, False),
        ("simulation_manifest", f"simulation_manifest_{SEASON}.json", True, False),
    ]
    if v2_in_bundle:
        (root / v2_rel).write_bytes(V2_CSV)
        specs.insert(1, ("v2_points", v2_rel, True, False))

    board_hash = hashlib.sha256((root / "selected_board.csv").read_bytes()).hexdigest()
    seal_staged_bundle(
        season=SEASON,
        namespace=namespace,
        root=root,
        release_id="rel-1",
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
        artifact_specs=specs,
    )
    return root


def _v2_check(report: dict) -> dict | None:
    return next(
        (c for c in report["checks"] if c["check"] == "v2_points_source_hash"),
        None,
    )


def test_pinned_v2_matching_the_sealed_copy_passes(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch)
    _seal(tmp_path, "ns_ok", v2_in_bundle=True, v2_in_contract=True)
    report = validate_release_bundle(season=SEASON, namespace="ns_ok")
    check = _v2_check(report)
    assert check is not None and check["passed"], check


def test_swapping_the_v2_board_is_detected(tmp_path, monkeypatch):
    """The whole point: this swap used to leave every hash in the chain valid."""
    _patch_roots(tmp_path, monkeypatch)
    root = _seal(tmp_path, "ns_swap", v2_in_bundle=True, v2_in_contract=True)

    (root / f"model_v2_fantasy_points_{SEASON}.csv").write_text(
        "player_id,fantasy_pts_season\na,999\n", encoding="utf-8"
    )

    report = validate_release_bundle(season=SEASON, namespace="ns_swap")
    check = _v2_check(report)
    assert check is not None and not check["passed"], check


def test_contract_pinning_v2_without_a_sealed_copy_fails(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch)
    _seal(tmp_path, "ns_unsealed", v2_in_bundle=False, v2_in_contract=True)
    report = validate_release_bundle(season=SEASON, namespace="ns_unsealed")
    check = _v2_check(report)
    assert check is not None and not check["passed"], check


def test_sealed_v2_the_contract_does_not_pin_fails(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch)
    _seal(tmp_path, "ns_unpinned", v2_in_bundle=True, v2_in_contract=False)
    report = validate_release_bundle(season=SEASON, namespace="ns_unpinned")
    check = _v2_check(report)
    assert check is not None and not check["passed"], check


def test_bundles_sealed_before_the_pin_still_validate(tmp_path, monkeypatch):
    """Backward compatibility: the currently active bundle predates the pin."""
    _patch_roots(tmp_path, monkeypatch)
    _seal(tmp_path, "ns_legacy", v2_in_bundle=False, v2_in_contract=False)
    report = validate_release_bundle(season=SEASON, namespace="ns_legacy")
    assert _v2_check(report) is None
    assert [c for c in report["checks"] if not c["passed"]] == []
