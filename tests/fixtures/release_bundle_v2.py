"""Shared helpers for v2 promotion-eligible release bundle tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.projection.overlay_coverage import compute_overlay_coverage
from src.projection.release_bundle import (
    SCHEMA_VERSION_V2,
    player_id_set_hash,
    selected_points_vector_hash,
    treatment_block,
)
from src.projection.release_bundle_publish import seal_staged_bundle
from src.projection.simulation_profile_resolver import (
    PROFILE_IDENTITY_FIELDS,
    resolve_simulation_profile_identity,
)


def _write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    return _write_bytes(path, data)


def overlay_player(player_id: str = "a") -> dict[str, Any]:
    return {
        "player_id": player_id,
        "fantasy_pts_p10": 80.0,
        "fantasy_pts_p25": 90.0,
        "fantasy_pts_p50": 100.0,
        "fantasy_pts_p75": 110.0,
        "fantasy_pts_p90": 120.0,
        "volatility_flag": False,
        "p_finish_top6": 0.1,
        "p_finish_top12": 0.2,
        "p_finish_top24": 0.3,
        "p_finish_top36": 0.4,
        "p_finish_top48": 0.5,
        "sim_vorp_p10": 1.0,
        "sim_vorp_p50": 2.0,
        "sim_vorp_p90": 3.0,
        "p_vorp_positive": 0.6,
        "expected_pos_rank": 10.0,
        "median_pos_rank": 11.0,
    }


def _simulation_config_fixture() -> dict[str, Any]:
    return {
        "random_seed": 2026,
        "profiles": {
            "publish": {"draws": 10000, "chunk_size": 250},
            "dev": {"draws": 1000, "chunk_size": 250},
        },
    }


def seal_v2_bundle(
    tmp_path: Path,
    namespace: str,
    *,
    season: int = 2026,
    release_id: str = "rel-1",
    source_commit: str = "abc123def4567890abcdef1234567890abcdef12",
) -> tuple[dict[str, Any], str]:
    from src.projection.release_bundle import bundle_root

    root = bundle_root(season, namespace)
    root.mkdir(parents=True, exist_ok=True)
    selected = b"player_id,fantasy_pts_season\na,100\n"
    board_hash = _write_bytes(root / "selected_board.csv", selected)
    players_doc = {
        "meta": {"model_id": "accuracy_first_ensemble", "selected_board_sha256": board_hash},
        "players": [overlay_player()],
    }
    _write_json(root / "players_2026.json", players_doc)
    _write_json(root / "team_stats_2026.json", {"players": []})
    _write_json(root / "comparison_2026.json", {"players": []})
    overlay_coverage = compute_overlay_coverage(players_doc)
    rollout = {
        "current_production_profile": "decision_stable_compromise_10000",
        "current_production_draw_count": 10000,
        "chosen_production_draw_count": 10000,
    }
    _write_json(root / "draw_count_rollout_decision.json", rollout)
    _write_json(root / "simulation_config.json", _simulation_config_fixture())
    profile_identity = resolve_simulation_profile_identity(
        profile_key="publish",
        rollout_path=root / "draw_count_rollout_decision.json",
        simulation_config_path_arg=root / "simulation_config.json",
    )
    _write_json(
        root / "release_report_2026.json",
        {
            "board": {"selected_board_sha256": board_hash},
            "overlay_coverage": overlay_coverage,
            "simulation": {key: profile_identity[key] for key in PROFILE_IDENTITY_FIELDS},
        },
    )
    _write_json(root / "release_report_simulation_2026.json", {})
    _write_json(root / "release_report_board_2026.json", {})
    weights_hash = _write_bytes(root / "ensemble_weights.json", b'{"weights":{}}')
    adp_hash = _write_bytes(root / f"consensus_{season}.json", b'{"players":[]}')
    v2_hash = _write_bytes(root / f"model_v2_fantasy_points_{season}.csv", b"player_id,fantasy_pts_season\na,100\n")
    _write_json(
        root / "application_contract.json",
        {
            "contract_hash": "0" * 64,
            "source_hashes": {
                "ensemble_weights": weights_hash,
                f"consensus_{season}": adp_hash,
                f"v2_points_{season}": v2_hash,
            },
        },
    )
    contract = json.loads((root / "application_contract.json").read_text(encoding="utf-8"))
    from src.projection.accuracy_application import contract_hash as compute_contract_hash

    contract["contract_hash"] = compute_contract_hash(contract)
    contract_hash = contract["contract_hash"]
    (root / "application_contract.json").write_text(json.dumps(contract), encoding="utf-8")

    _write_json(
        root / "simulation_manifest_2026.json",
        {
            "draw_count": profile_identity["draw_count"],
            "chunk_size": profile_identity["chunk_size"],
            "simulation_run_id": "sim-1",
            "canonical_projection_run_id": "proj-1",
            "simulation_profile": profile_identity["profile_key"],
            "profile_key": profile_identity["profile_key"],
            "profile_label": profile_identity["profile_label"],
            "configuration_hash": profile_identity["configuration_hash"],
            "policy_hash": profile_identity["policy_hash"],
            "selected_board_hash": board_hash,
            "selected_board_sha256": board_hash,
        },
    )
    specs = [
        ("selected_board", "selected_board.csv", True, False),
        ("v2_points", f"model_v2_fantasy_points_{season}.csv", True, False),
        ("adp_source", f"consensus_{season}.json", True, False),
        ("ensemble_weights", "ensemble_weights.json", True, False),
        ("simulation_config", "simulation_config.json", True, False),
        ("draw_count_rollout_decision", "draw_count_rollout_decision.json", True, False),
        ("players", "players_2026.json", True, True),
        ("team_stats", "team_stats_2026.json", True, True),
        ("comparison", "comparison_2026.json", True, True),
        ("release_report", "release_report_2026.json", True, False),
        ("release_report_simulation", "release_report_simulation_2026.json", True, False),
        ("release_report_board", "release_report_board_2026.json", True, False),
        ("application_contract", "application_contract.json", True, False),
        ("simulation_manifest", "simulation_manifest_2026.json", True, False),
    ]
    return seal_staged_bundle(
        season=season,
        namespace=namespace,
        root=root,
        release_id=release_id,
        application={"contract_version": "accuracy_first_2026_v1", "contract_hash": contract_hash},
        runs={"projection_run_id": "proj-1", "simulation_run_id": "sim-1"},
        board={
            "selected_board_sha256": board_hash,
            "selected_board_file_hash": board_hash,
            "selected_points_vector_hash": selected_points_vector_hash({"a": 100.0}),
        },
        simulation={
            "profile": profile_identity["profile_key"],
            "profile_key": profile_identity["profile_key"],
            "profile_label": profile_identity["profile_label"],
            "draw_count": profile_identity["draw_count"],
            "chunk_size": profile_identity["chunk_size"],
            "configuration_hash": profile_identity["configuration_hash"],
            "policy_hash": profile_identity["policy_hash"],
            "calibration_hashes": {"v3_interval": "c" * 64},
            "joint_donor_hash": "d" * 64,
        },
        overlay={
            "simulated_player_population_hash": player_id_set_hash(["a"]),
            "simulated_player_count": 1,
        },
        overlay_coverage=overlay_coverage,
        ensemble={
            "contract_hash": contract_hash,
            "ensemble_weights_hash": weights_hash,
            "v2_points_hash": v2_hash,
            "adp_source_hash": adp_hash,
        },
        git={"source_commit": source_commit, "source_dirty": False},
        contract_treatments={
            "selected": treatment_block(["a"]),
            "incumbent": treatment_block([]),
            "new_player_v1_only": treatment_block([]),
        },
        artifact_specs=specs,
        schema_version=SCHEMA_VERSION_V2,
    )
