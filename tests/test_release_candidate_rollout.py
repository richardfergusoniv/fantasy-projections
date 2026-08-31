"""Tests for draw-count rollout, freeze, RC non-publication, and overlay comparison."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.projection.evaluation.draw_count_rollout import (
    compare_overlay_identities,
    write_draw_count_rollout_decision,
)
from src.projection.evaluation.draw_profile_comparison import compare_draw_profile_overlays
from src.projection.evaluation.evidence_freeze import freeze_draw_stability_evidence
from src.projection.evaluation.release_pointer import snapshot_production_bundle
from src.projection.release_candidate import (
    assert_public_artifacts_unchanged,
    snapshot_public_artifact_hashes,
)


def test_assert_public_artifacts_unchanged_detects_mutation():
    before = {"players": "abc", "simulation_manifest": "def"}
    after = {"players": "abc", "simulation_manifest": "changed"}
    violations = assert_public_artifacts_unchanged(before, after)
    assert len(violations) == 1
    assert "simulation_manifest" in violations[0]


def test_compare_overlay_identities_hold_on_board_mismatch():
    left = {"selected_board_hash": "aaa", "canonical_projection_run_id": "run1"}
    right = {"selected_board_hash": "bbb", "canonical_projection_run_id": "run1"}
    mismatches = compare_overlay_identities(left, right)
    assert any("selected_board_hash" in item for item in mismatches)


def test_compare_draw_profile_overlays_hold_on_identity_mismatch(tmp_path: Path):
    profiles = {
        "a": {
            "manifest": tmp_path / "a_manifest.json",
            "players": tmp_path / "a_players.json",
        },
        "b": {
            "manifest": tmp_path / "b_manifest.json",
            "players": tmp_path / "b_players.json",
        },
    }
    profiles["a"]["manifest"].write_text(
        json.dumps(
            {
                "selected_board_hash": "hash_a",
                "selected_board_model_id": "accuracy_first_ensemble",
                "canonical_projection_run_id": "run1",
                "wr_calibration_artifact_hash": "wr1",
                "transform_version": "v1_median_correction",
                "finish_probability_gate_hash": "fp1",
            }
        ),
        encoding="utf-8",
    )
    profiles["b"]["manifest"].write_text(
        json.dumps(
            {
                "selected_board_hash": "hash_b",
                "selected_board_model_id": "accuracy_first_ensemble",
                "canonical_projection_run_id": "run1",
                "wr_calibration_artifact_hash": "wr1",
                "transform_version": "v1_median_correction",
                "finish_probability_gate_hash": "fp1",
            }
        ),
        encoding="utf-8",
    )
    for label in ("a", "b"):
        profiles[label]["players"].write_text(
            json.dumps({"players": [{"player_id": "p1", "fantasy_pts_p50": 100.0}]}),
            encoding="utf-8",
        )
    report = compare_draw_profile_overlays(season=2026, profiles=profiles)
    assert report["comparison_verdict"] == "hold"
    assert report["reason"] == "board_or_contract_identity_mismatch"


def test_write_phase2_rollout_closure(tmp_path: Path, monkeypatch):
    from src.projection.evaluation import draw_count_rollout as rollout

    model_v3 = tmp_path / "model_v3"
    model_v3.mkdir()
    frozen = model_v3 / "frozen" / "test_freeze"
    frozen.mkdir(parents=True)
    (frozen / "freeze_manifest.json").write_text(
        json.dumps(
            {
                "freeze_id": "test_freeze",
                "selected_board_hash": "hash",
                "canonical_projection_run_id": "run",
                "reference_draw_count": 20000,
                "nested_prefix_provenance_verdict": "ok",
                "sha256": {"freeze_manifest.json": "abc"},
            }
        ),
        encoding="utf-8",
    )
    (model_v3 / "draw_count_decision.json").write_text(
        json.dumps({"schema_version": "draw_count_decision_v2", "production_recommendation": "x"}),
        encoding="utf-8",
    )
    releases = model_v3 / "releases"
    releases.mkdir()
    (releases / "release_2026_current.json").write_text(
        json.dumps({"profile": "provisional_1000"}), encoding="utf-8"
    )
    rc_dir = (
        model_v3
        / "release_candidates"
        / "season=2026"
        / "namespace=rc_test"
    )
    rc_dir.mkdir(parents=True)
    (rc_dir / "simulation_manifest_2026.json").write_text(
        json.dumps({"runtime_seconds": 600, "draw_count": 10000, "simulation_run_id": "rc-run"}),
        encoding="utf-8",
    )
    (rc_dir / "release_candidate_validation_2026.json").write_text(
        json.dumps({"public_immutability_pass": True}), encoding="utf-8"
    )

    monkeypatch.setattr(rollout, "MODEL_V3_DIR", str(model_v3))
    monkeypatch.setattr(
        "src.projection.evaluation.evidence_freeze.FROZEN_ROOT",
        model_v3 / "frozen",
    )
    monkeypatch.setattr(
        rollout,
        "frozen_evidence_manifest_hash",
        lambda freeze_id: "test_manifest_hash",
    )
    monkeypatch.setattr(
        "src.projection.evaluation.release_pointer.MODEL_V3_DIR",
        str(model_v3),
    )

    path = rollout.write_phase2_rollout_closure(
        season=2026,
        freeze_id="test_freeze",
        rc_namespace="rc_test",
        operational_policy="maintain_1000_temporarily",
        decision_rationale="test",
        human_decision_record_path="docs/test.md",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["current_production_profile"] == "provisional_current_configuration"
    assert payload["operational_policy"] == "maintain_1000_temporarily"
    assert payload["phase_2_status"] == "closed"


def test_evaluate_decision_stable_10k_promotion_requires_all_checks():
    from src.projection.evaluation.draw_count_rollout import evaluate_decision_stable_10k_promotion

    gate = evaluate_decision_stable_10k_promotion(
        overlay_comparison={"comparison_verdict": "compare"},
        stability_candidate={"material_decision_events": 0, "core_adp_decision_events": 0},
        runtime_seconds=9656.0,
    )
    assert gate["promote"] is True

    hold = evaluate_decision_stable_10k_promotion(
        overlay_comparison={"comparison_verdict": "hold", "reason": "board_or_contract_identity_mismatch"},
        stability_candidate={"material_decision_events": 0, "core_adp_decision_events": 0},
        runtime_seconds=9656.0,
    )
    assert hold["promote"] is False
    assert hold["checks"]["overlay_identity_compare"] is False


def test_freeze_draw_stability_evidence_copies_and_hashes(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for name in (
        "draw_stability_intermediate_v20k_2026.json",
        "draw_count_decision.json",
        "decision_change_diagnostics_2026.json",
        "player_stability_diagnostics_2026.parquet",
    ):
        path = source / name
        if name.endswith(".parquet"):
            pd.DataFrame({"player_id": ["p1"], "flag": [1]}).to_parquet(path, index=False)
        else:
            path.write_text(json.dumps({"artifact": name}), encoding="utf-8")

    frozen_root = tmp_path / "frozen"
    monkeypatch.setattr(
        "src.projection.evaluation.evidence_freeze.FROZEN_ROOT",
        frozen_root,
    )
    manifest_path = freeze_draw_stability_evidence(
        freeze_id="test_freeze",
        season=2026,
        source_dir=source,
        selected_board_hash="board_hash",
        canonical_projection_run_id="run_id",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["freeze_id"] == "test_freeze"
    assert manifest["selected_board_hash"] == "board_hash"
    assert len(manifest["sha256"]) >= 4
    assert (frozen_root / "test_freeze" / "draw_count_decision.json").exists()


def test_snapshot_production_bundle_structure(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "src.projection.evaluation.release_pointer.MODEL_V3_DIR",
        str(tmp_path / "model_v3"),
    )
    monkeypatch.setattr(
        "src.projection.evaluation.release_pointer.OUTPUT_DIR",
        str(tmp_path / "output"),
    )
    monkeypatch.setattr(
        "src.projection.evaluation.release_pointer.DRAFT_DATA_DIR",
        str(tmp_path / "draft_assistant" / "data"),
    )
    model_v3 = tmp_path / "model_v3"
    model_v3.mkdir(parents=True)
    manifest = {
        "draw_count": 1000,
        "canonical_projection_run_id": "run-1",
        "selected_board_hash": "hash-1",
    }
    (model_v3 / "simulation_manifest_2026.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    bundle = snapshot_production_bundle(2026, label="test", profile="provisional_1000")
    assert bundle["draw_count"] == 1000
    assert bundle["profile"] == "provisional_1000"
    assert bundle["files"]["simulation_manifest"]["exists"] is True
