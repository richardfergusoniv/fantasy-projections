"""Namespaced release-candidate publish that never mutates public production artifacts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.draft_assistant.prepare import ACCURACY_FIRST_DIR, export_draft_data
from src.projection.contracts import MODEL_V3_DIR, OUTPUT_DIR, REPO_ROOT
from src.projection.evaluation.draw_count_rollout import full_overlay_identity
from src.projection.evaluation.release_pointer import ensure_release_pointers
from src.projection.evaluation.release_report import (
    build_release_report_simulation,
    write_release_report_simulation,
)
from src.projection.inference.simulate import write_simulation_outputs
from src.projection.publish import sha256_file, validate_projection_contract


def rc_namespace_dir(season: int, namespace: str) -> Path:
    return (
        Path(MODEL_V3_DIR)
        / "release_candidates"
        / f"season={season}"
        / f"namespace={namespace}"
    )


def snapshot_public_artifact_hashes(season: int) -> dict[str, str | None]:
    from src.draft_assistant.prepare import DRAFT_DATA_DIR

    paths = {
        "players": Path(DRAFT_DATA_DIR) / f"players_{season}.json",
        "simulation_manifest": Path(MODEL_V3_DIR) / f"simulation_manifest_{season}.json",
        "release_report": Path(MODEL_V3_DIR) / f"release_report_{season}.json",
        "release_pointer_current": Path(MODEL_V3_DIR) / "releases" / f"release_{season}_current.json",
    }
    return {
        key: sha256_file(path) if path.exists() else None for key, path in paths.items()
    }


def assert_public_artifacts_unchanged(
    before: dict[str, str | None],
    after: dict[str, str | None],
) -> list[str]:
    violations: list[str] = []
    for key, prior in before.items():
        current = after.get(key)
        if prior != current:
            violations.append(f"{key}: before={prior!r} after={current!r}")
    return violations


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def publish_release_candidate(
    season: int,
    *,
    artifact_namespace: str,
    simulation_draws: int = 10000,
    simulation_profile: str = "release_candidate",
    rollout_label: str,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Run simulation + RC export under a namespace without touching public artifacts."""
    if not artifact_namespace:
        raise ValueError("artifact_namespace is required for release-candidate publish")
    if simulation_profile != "release_candidate":
        raise ValueError("release-candidate publish requires simulation_profile=release_candidate")

    ensure_release_pointers(season)
    public_before = snapshot_public_artifact_hashes(season)
    pointer_path = Path(MODEL_V3_DIR) / "releases" / f"release_{season}_current.json"
    release_pointer_before = pointer_path.read_bytes() if pointer_path.exists() else None

    rc_dir = rc_namespace_dir(season, artifact_namespace)
    rc_dir.mkdir(parents=True, exist_ok=True)

    projections_path = Path(OUTPUT_DIR) / f"projection_run_{season}.json"
    projections_csv = Path(OUTPUT_DIR) / f"projections_{season}.csv"
    if not projections_csv.exists():
        raise FileNotFoundError(f"Missing frozen projections for RC: {projections_csv}")
    projections = pd.read_csv(projections_csv)
    validate_projection_contract(projections, season)
    projection_run = (
        json.loads(projections_path.read_text(encoding="utf-8"))
        if projections_path.exists()
        else None
    )

    accuracy_board_path = Path(ACCURACY_FIRST_DIR) / f"fantasy_points_{season}.csv"
    selected_board = None
    selected_board_hash = None
    selected_board_model_id = None
    if accuracy_board_path.exists():
        selected_board = pd.read_csv(accuracy_board_path)
        selected_board_hash = sha256_file(accuracy_board_path)
        selected_board_model_id = "accuracy_first_ensemble"

    canonical_run_id = str(projections["projection_run_id"].iloc[0])
    simulation_run_id = f"{canonical_run_id}__rc__{artifact_namespace}"
    partition_root = rc_dir / "simulations"

    simulation_manifest = write_simulation_outputs(
        projections,
        season,
        n_draws=simulation_draws,
        selected_board=selected_board,
        selected_board_hash=selected_board_hash,
        selected_board_model_id=selected_board_model_id,
        simulation_profile=simulation_profile,
        out_dir=rc_dir,
        partition_root=partition_root,
        simulation_run_id=simulation_run_id,
        rollout_label=rollout_label,
        artifact_namespace=artifact_namespace,
    )
    simulation_manifest["simulation_run_id"] = simulation_run_id
    simulation_manifest["artifact_namespace"] = artifact_namespace
    simulation_manifest["rollout_label"] = rollout_label
    simulation_manifest["rc_is_non_public"] = True
    _write_json(rc_dir / f"simulation_manifest_{season}.json", simulation_manifest)
    _write_json(
        rc_dir / "projection_manifest.json",
        {
            "schema_version": 1,
            "season": season,
            "run_id": canonical_run_id,
            "source": str(projections_path.relative_to(REPO_ROOT)).replace("\\", "/")
            if projections_path.exists()
            else None,
            "projection_run": projection_run,
            "as_of": as_of,
            "rc_is_non_public": True,
            "artifact_namespace": artifact_namespace,
        },
    )

    sim_report = build_release_report_simulation(
        season=season,
        projection_run=projection_run,
        simulation_manifest=simulation_manifest,
    )
    sim_report["artifact_namespace"] = artifact_namespace
    sim_report["rollout_label"] = rollout_label
    sim_report["rc_is_non_public"] = True
    write_release_report_simulation(sim_report, season=season, out_dir=rc_dir)

    fantasy_path = Path(OUTPUT_DIR) / f"fantasy_points_{season}.csv"
    players_rc_path = rc_dir / f"players_{season}_rc.json"
    export_draft_data(
        season,
        fantasy_path=str(fantasy_path),
        out_path=str(players_rc_path),
        model_v3_dir=str(rc_dir),
        simulation_manifest_path=str(rc_dir / f"simulation_manifest_{season}.json"),
        skip_public_release_reports=True,
        require_gate=False,
    )

    production_manifest_path = Path(MODEL_V3_DIR) / f"simulation_manifest_{season}.json"
    production_run_id = None
    if production_manifest_path.exists():
        production_manifest = json.loads(production_manifest_path.read_text(encoding="utf-8"))
        production_run_id = production_manifest.get("simulation_run_id") or production_manifest.get(
            "canonical_projection_run_id"
        )

    validation = {
        "schema_version": "release_candidate_validation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "artifact_namespace": artifact_namespace,
        "rollout_label": rollout_label,
        "simulation_draw_count": simulation_draws,
        "simulation_profile": simulation_profile,
        "simulation_run_id": simulation_run_id,
        "production_simulation_run_id": production_run_id,
        "public_artifact_hashes_before": public_before,
        "overlay_identity": full_overlay_identity(simulation_manifest, season=season),
        "rc_root": str(rc_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
    }

    public_after = snapshot_public_artifact_hashes(season)
    validation["public_artifact_hashes_after"] = public_after
    violations = assert_public_artifacts_unchanged(public_before, public_after)
    validation["public_immutability_violations"] = violations
    validation["public_immutability_pass"] = not violations

    release_pointer_after = pointer_path.read_bytes() if pointer_path.exists() else None
    validation["release_pointer_unchanged"] = release_pointer_before == release_pointer_after

    _write_json(rc_dir / f"release_candidate_validation_{season}.json", validation)
    _write_json(rc_dir / "validation.json", validation)

    if violations:
        raise RuntimeError(
            "Release-candidate publish modified public artifacts: "
            + "; ".join(violations)
        )
    if not validation["release_pointer_unchanged"]:
        raise RuntimeError("Release-candidate publish modified the active release pointer")

    return {
        "artifact_namespace": artifact_namespace,
        "rc_dir": str(rc_dir),
        "simulation_manifest": simulation_manifest,
        "validation": validation,
    }
