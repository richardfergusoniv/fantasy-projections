"""Build a namespaced sealed release bundle without touching the active pointer."""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.draft_assistant.compare_prepare import rebase_comparison_payload
from src.draft_assistant.draft_value_simulation import (
    compute_finish_probabilities,
    compute_simulated_vorp_metrics,
)
from src.draft_assistant.prepare import DRAFT_DATA_DIR, export_draft_data
from src.projection.accuracy_application import (
    DEFAULT_CONTRACT_PATH,
    MODEL_ID,
    apply_application_contract,
    freeze_application_contract_from_artifacts,
    inputs_from_frozen_sources,
    load_application_contract,
    validate_application_contract,
)
from src.projection.active_release import pointer_path
from src.projection.contracts import MODEL_V3_DIR, OUTPUT_DIR, REPO_ROOT
from src.projection.evaluation.accuracy_first import canonical_json_hash, sha256_file
from src.projection.evaluation.release_bundle_validation import validate_release_bundle
from src.projection.evaluation.release_report import (
    build_release_report_simulation,
    write_release_report_simulation,
)
from src.projection.inference.recenter import board_points_series
from src.projection.inference.simulate import write_simulation_outputs
from src.projection.inference.simulation_config import load_simulation_config, profile_draws
from src.projection.release_bundle import (
    MANIFEST_FILENAME,
    artifact_record,
    build_manifest,
    bundle_root,
    player_id_set_hash,
    public_release_dir,
    seal_manifest,
    selected_points_vector_hash,
    treatment_block,
    validate_namespace,
)
from src.projection.release_candidate import (
    assert_public_artifacts_unchanged,
    snapshot_public_artifact_hashes,
)
from src.team_stats.prepare import export_team_stats


BROWSER_ROLES = ("players", "team_stats", "comparison", "deep_band_accuracy")


class ReleaseBundlePublishError(RuntimeError):
    """Candidate construction refused."""


def live_release_snapshot(season: int) -> dict[str, str | None]:
    hashes = snapshot_public_artifact_hashes(season)
    path = pointer_path(season)
    hashes["active_pointer"] = sha256_file(path) if path.exists() else None
    return hashes


def copy_browser_consumed(*, root: Path, manifest: Mapping[str, Any], manifest_sha256: str) -> Path:
    namespace = manifest["bundle"]["namespace"]
    public = public_release_dir(namespace)
    if public.exists():
        shutil.rmtree(public)
    public.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / MANIFEST_FILENAME, public / MANIFEST_FILENAME)
    copied_digest = sha256_file(public / MANIFEST_FILENAME)
    if copied_digest != manifest_sha256:
        raise ReleaseBundlePublishError("public manifest copy is not byte-identical to the sealed manifest")
    for entry in manifest["artifacts"]:
        if not entry.get("browser_consumed"):
            continue
        src = root / entry["path"]
        dest = public / entry["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        if sha256_file(dest) != entry["sha256"]:
            raise ReleaseBundlePublishError(f"public copy drifted for {entry['role']}")
    return public


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _simulation_configuration_hash(profile: str) -> str:
    config = load_simulation_config()
    return canonical_json_hash(
        {
            "random_seed": config.get("random_seed"),
            "profile": profile,
            "profile_config": (config.get("profiles") or {}).get(profile),
        }
    )


def _artifact(
    role: str,
    rel: str,
    root: Path,
    *,
    required: bool = True,
    browser_consumed: bool = False,
) -> dict[str, Any]:
    return artifact_record(
        role=role,
        path=rel,
        file_path=root / rel,
        required=required,
        browser_consumed=browser_consumed,
    )


def seal_staged_bundle(
    *,
    season: int,
    namespace: str,
    root: Path,
    release_id: str,
    application: Mapping[str, Any],
    runs: Mapping[str, Any],
    board: Mapping[str, Any],
    simulation: Mapping[str, Any],
    overlay: Mapping[str, Any],
    contract_treatments: Mapping[str, Any],
    artifact_specs: list[tuple[str, str, bool, bool]],
    created_at: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Hash staged files, seal the manifest, copy browser artifacts, write attestation.

    Does not write or modify the active pointer.
    """
    validate_namespace(namespace)
    artifacts = [
        _artifact(role, rel, root, required=required, browser_consumed=browser)
        for role, rel, required, browser in artifact_specs
    ]
    payload = build_manifest(
        season=season,
        namespace=namespace,
        release_id=release_id,
        model_id=MODEL_ID,
        created_at=created_at,
        application=application,
        runs=runs,
        board=board,
        simulation=simulation,
        overlay=overlay,
        artifacts=artifacts,
        contract_treatments=contract_treatments,
    )
    manifest, digest = seal_manifest(payload, root=root)
    copy_browser_consumed(root=root, manifest=manifest, manifest_sha256=digest)
    validate_release_bundle(season=season, namespace=namespace, require_active=False)
    return manifest, digest


def _ensure_application_contract() -> dict[str, Any]:
    if DEFAULT_CONTRACT_PATH.exists():
        return load_application_contract(DEFAULT_CONTRACT_PATH)
    return freeze_application_contract_from_artifacts()


def _attach_overlays(players_path: Path, draws_path: Path, board: pd.DataFrame) -> None:
    if not draws_path.exists():
        raise ReleaseBundlePublishError(f"missing recentered draws: {draws_path}")
    draws = pd.read_parquet(draws_path)
    payload = json.loads(players_path.read_text(encoding="utf-8"))
    finish = compute_finish_probabilities(draws)
    vorp = compute_simulated_vorp_metrics(draws, board)
    by_finish = finish.set_index("player_id") if not finish.empty else pd.DataFrame()
    by_vorp = vorp.set_index("player_id") if not vorp.empty else pd.DataFrame()
    for player in payload.get("players") or []:
        pid = str(player.get("player_id"))
        if pid in by_finish.index:
            for col in by_finish.columns:
                player[col] = float(by_finish.loc[pid, col])
        if pid in by_vorp.index:
            for col in by_vorp.columns:
                value = by_vorp.loc[pid, col]
                player[col] = float(value) if pd.notna(value) else None
    players_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def publish_release_bundle(
    season: int,
    *,
    artifact_namespace: str,
    simulation_profile: str = "publish",
    simulation_draws: int | None = None,
    as_of: str | None = None,
    projections: pd.DataFrame | None = None,
    skip_simulation: bool = False,
) -> dict[str, Any]:
    """Generate artifacts, then seal. Never mutates the active pointer."""
    if simulation_profile != "publish":
        raise ReleaseBundlePublishError("namespaced accuracy publish requires simulation-profile=publish")
    namespace = validate_namespace(artifact_namespace)
    before = live_release_snapshot(season)
    pointer_before = pointer_path(season).read_bytes() if pointer_path(season).exists() else None

    contract = _ensure_application_contract()
    validate_application_contract(contract, require_source_files=False)

    from src.projection.data_prep import get_conn
    from src.projection.fantasy_points import compute_fantasy_points
    from src.projection.predict import project_season, with_display_names
    from src.projection.publish import COMPOSITION_VERSION, validate_projection_contract
    from src.sentiment.snapshot import attach_sentiment

    run_id = str(uuid.uuid4())
    if projections is None:
        conn = get_conn()
        try:
            projections = project_season(conn, season, as_of=as_of)
            projections = with_display_names(conn, projections, season)
        finally:
            conn.close()
        projections = attach_sentiment(projections, season=season, as_of=as_of)
        projections["projection_run_id"] = run_id
        projections["composition_version"] = COMPOSITION_VERSION
        from src.projection.contracts import OUTPUT_COLUMNS

        projections = projections[OUTPUT_COLUMNS].sort_values(
            ["position", "team", "player_id", "stat"]
        )
    else:
        projections = projections.copy()
        if "projection_run_id" not in projections.columns:
            projections["projection_run_id"] = run_id
        run_id = str(projections["projection_run_id"].iloc[0])
    validate_projection_contract(projections, season)
    fantasy = compute_fantasy_points(projections)

    v2_path = Path(OUTPUT_DIR) / "model_v2" / f"fantasy_points_{season}.csv"
    consensus_path = Path(REPO_ROOT) / "data" / "consensus" / f"consensus_{season}.json"
    v3_path = Path(MODEL_V3_DIR) / f"simulation_summary_{season}.csv"
    v2_by_id, adp_by_id, v3_by_id = inputs_from_frozen_sources(
        v2_path=v2_path,
        consensus_path=consensus_path,
        v3_path=v3_path if v3_path.exists() else None,
    )
    selected, treatments = apply_application_contract(
        fantasy,
        contract,
        v2_by_id=v2_by_id,
        adp_by_id=adp_by_id,
        v3_by_id=v3_by_id,
    )

    root = bundle_root(season, namespace)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    selected_rel = f"fantasy_points_{season}.csv"
    projections_rel = f"projections_{season}.csv"
    selected.to_csv(root / selected_rel, index=False)
    projections.to_csv(root / projections_rel, index=False)
    contract_rel = "application_contract.json"
    _write_json(root / contract_rel, contract)

    sim_config = load_simulation_config()
    draws = simulation_draws if simulation_draws is not None else profile_draws(sim_config, "publish")
    draws = int(draws or 10000)
    simulation_run_id = f"{run_id}__publish__{namespace}"
    simulation_manifest = None
    if not skip_simulation:
        simulation_manifest = write_simulation_outputs(
            projections,
            season,
            n_draws=draws,
            selected_board=selected,
            selected_board_hash=sha256_file(root / selected_rel),
            selected_board_model_id=MODEL_ID,
            simulation_profile="publish",
            out_dir=root,
            partition_root=root / "simulations",
            simulation_run_id=simulation_run_id,
            artifact_namespace=namespace,
        )
        _write_json(root / f"simulation_manifest_{season}.json", simulation_manifest)

    projection_run = {
        "schema_version": 1,
        "run_id": run_id,
        "season": int(season),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_namespace": namespace,
        "application_contract_hash": contract["contract_hash"],
    }
    _write_json(root / "projection_run.json", projection_run)

    players_rel = f"players_{season}.json"
    team_rel = f"team_stats_{season}.json"
    sim_report = build_release_report_simulation(
        season=season,
        projection_run=projection_run,
        simulation_manifest=simulation_manifest,
    )
    sim_report["artifact_namespace"] = namespace
    write_release_report_simulation(sim_report, season=season, out_dir=root)
    export_team_stats(
        season,
        projections_path=str(root / projections_rel),
        fantasy_path=str(root / selected_rel),
        out_path=str(root / team_rel),
    )
    export_draft_data(
        season,
        fantasy_path=str(root / selected_rel),
        out_path=str(root / players_rel),
        model_v3_dir=str(root),
        simulation_manifest_path=str(root / f"simulation_manifest_{season}.json")
        if (root / f"simulation_manifest_{season}.json").exists()
        else None,
        skip_public_release_reports=True,
        require_gate=False,
        use_ensemble=False,
        attach_sim_vorp=False,
    )
    recentered_path = root / f"simulations_recentered_{season}.parquet"
    if recentered_path.exists():
        _attach_overlays(root / players_rel, recentered_path, selected)

    comparison_rel = f"comparison_{season}.json"
    legacy_comparison = Path(DRAFT_DATA_DIR) / f"comparison_{season}.json"
    players_doc = json.loads((root / players_rel).read_text(encoding="utf-8"))
    if legacy_comparison.exists():
        comparison = rebase_comparison_payload(
            players_doc,
            json.loads(legacy_comparison.read_text(encoding="utf-8")),
        )
    else:
        comparison = {
            "meta": {
                "season": season,
                "board_model_id": MODEL_ID,
                "market_snapshot_preserved": False,
            },
            "players": [],
        }
    _write_json(root / comparison_rel, comparison)

    board_points = {
        str(pid): float(pts)
        for pid, pts in board_points_series(selected).items()
    }
    overlay_ids = [str(pid) for pid in board_points]
    if players_doc.get("players"):
        overlay_ids = [str(row["player_id"]) for row in players_doc["players"]]

    # The v2 board is an input the bundle previously referenced but never
    # sealed: it lives in a separate repository and is synced in as a CSV, so
    # without a copy inside the namespace the manifest could not pin the file
    # that carries 0.55 of the published WR mean and 0.30 of RB. Copy it in and
    # enumerate it like any other artifact.
    v2_rel = f"model_v2_fantasy_points_{season}.csv"
    shutil.copy2(v2_path, root / v2_rel)

    artifact_specs = [
        ("selected_board", selected_rel, True, False),
        ("v2_points", v2_rel, True, False),
        ("projections", projections_rel, True, False),
        ("application_contract", contract_rel, True, False),
        ("projection_run", "projection_run.json", True, False),
        ("players", players_rel, True, True),
        ("team_stats", team_rel, True, True),
        ("comparison", comparison_rel, True, True),
        ("release_report_simulation", f"release_report_simulation_{season}.json", True, False),
        ("release_report_board", f"release_report_board_{season}.json", True, False),
        ("release_report", f"release_report_{season}.json", True, False),
    ]
    if (root / f"simulation_manifest_{season}.json").exists():
        artifact_specs.append(("simulation_manifest", f"simulation_manifest_{season}.json", True, False))
    if (root / f"simulation_summary_{season}.csv").exists():
        artifact_specs.append(("simulation_summary", f"simulation_summary_{season}.csv", True, False))
    if (root / f"simulation_summary_recentered_{season}.csv").exists():
        artifact_specs.append(
            ("simulation_summary_recentered", f"simulation_summary_recentered_{season}.csv", True, False)
        )
    sim_parquet = root / f"simulations_{season}.parquet"
    if sim_parquet.exists():
        artifact_specs.append(("simulations", sim_parquet.name, True, False))
    if recentered_path.exists():
        artifact_specs.append(("recentered_draws", recentered_path.name, True, False))
    deep_band = Path(DRAFT_DATA_DIR) / "deep_band_accuracy.json"
    if deep_band.exists():
        shutil.copy2(deep_band, root / "deep_band_accuracy.json")
        artifact_specs.append(("deep_band_accuracy", "deep_band_accuracy.json", False, True))

    partition_dir = root / "simulations"
    if partition_dir.exists():
        for path in sorted(partition_dir.rglob("*.parquet")):
            rel = path.relative_to(root).as_posix()
            artifact_specs.append((f"draw_partition:{rel}", rel, False, False))

    cal_hashes = {}
    if simulation_manifest:
        for key in (
            "calibration_hash",
            "wr_calibration_artifact_hash",
            "finish_probability_gate_hash",
            "segment_report_hash",
        ):
            if simulation_manifest.get(key):
                cal_hashes[key] = simulation_manifest[key]
    joint_hash = (simulation_manifest or {}).get("joint_donors_hash") or ("0" * 64)

    manifest, digest = seal_staged_bundle(
        season=season,
        namespace=namespace,
        root=root,
        release_id=str(uuid.uuid4()),
        application={
            "contract_version": contract["contract_version"],
            "contract_hash": contract["contract_hash"],
        },
        runs={
            "projection_run_id": run_id,
            "simulation_run_id": (simulation_manifest or {}).get("simulation_run_id") or simulation_run_id,
        },
        board={
            "selected_board_file_hash": sha256_file(root / selected_rel),
            "selected_points_vector_hash": selected_points_vector_hash(board_points),
        },
        simulation={
            "profile": "publish",
            "draw_count": int((simulation_manifest or {}).get("draw_count") or draws),
            "configuration_hash": _simulation_configuration_hash("publish"),
            "calibration_hashes": cal_hashes or {"placeholder": "0" * 64},
            "joint_donor_hash": joint_hash,
        },
        overlay={
            "simulated_player_population_hash": player_id_set_hash(overlay_ids),
            "simulated_player_count": len(set(overlay_ids)),
        },
        contract_treatments={
            "selected": treatment_block(treatments["player_ids"]["selected"]),
            "incumbent": treatment_block(treatments["player_ids"]["incumbent"]),
            "new_player_v1_only": treatment_block(treatments["player_ids"]["new_player_v1_only"]),
        },
        artifact_specs=artifact_specs,
    )

    after = live_release_snapshot(season)
    violations = assert_public_artifacts_unchanged(before, after)
    pointer_after = pointer_path(season).read_bytes() if pointer_path(season).exists() else None
    if violations:
        raise ReleaseBundlePublishError(
            "candidate construction mutated the live release: " + "; ".join(violations)
        )
    if pointer_before != pointer_after:
        raise ReleaseBundlePublishError("candidate construction mutated the active pointer")

    return {
        "artifact_namespace": namespace,
        "bundle_root": str(root),
        "manifest_sha256": digest,
        "release_id": manifest["bundle"]["release_id"],
        "treatments": {key: treatments[key] for key in ("selected", "incumbent", "new_player_v1_only")},
        "public_immutability_pass": True,
    }
