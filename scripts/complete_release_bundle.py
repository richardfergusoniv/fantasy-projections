#!/usr/bin/env python3
"""Complete sealing for a staged release bundle after simulation already finished."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.projection.accuracy_application import load_application_contract
from src.projection.evaluation.accuracy_first import sha256_file
from src.projection.inference.recenter import board_points_series
from src.projection.release_bundle import (
    bundle_root,
    player_id_set_hash,
    selected_points_vector_hash,
    treatment_block,
)
from src.projection.release_bundle_publish import (
    _simulation_configuration_hash,
    copy_browser_consumed,
    seal_staged_bundle,
)


def complete_seal(season: int, namespace: str) -> dict:
    root = bundle_root(season, namespace)
    contract = load_application_contract(root / "application_contract.json")
    selected_rel = f"fantasy_points_{season}.csv"
    selected = pd.read_csv(root / selected_rel)
    board_points = {str(pid): float(pts) for pid, pts in board_points_series(selected).items()}
    players_path = root / f"players_{season}.json"
    players_doc = json.loads(players_path.read_text(encoding="utf-8")) if players_path.exists() else {}
    overlay_ids = [str(row["player_id"]) for row in players_doc.get("players") or []] or list(board_points)
    sim_manifest_path = root / f"simulation_manifest_{season}.json"
    simulation_manifest = json.loads(sim_manifest_path.read_text(encoding="utf-8")) if sim_manifest_path.exists() else {}
    projection_run = json.loads((root / "projection_run.json").read_text(encoding="utf-8"))
    recentered_path = root / f"simulations_recentered_{season}.parquet"

    artifact_specs = [
        ("selected_board", selected_rel, True, False),
        ("projections", f"projections_{season}.csv", True, False),
        ("application_contract", "application_contract.json", True, False),
        ("projection_run", "projection_run.json", True, False),
        ("players", f"players_{season}.json", True, True),
        ("team_stats", f"team_stats_{season}.json", True, True),
        ("comparison", f"comparison_{season}.json", True, True),
        ("release_report_simulation", f"release_report_simulation_{season}.json", True, False),
        ("release_report_board", f"release_report_board_{season}.json", True, False),
        ("release_report", f"release_report_{season}.json", True, False),
    ]
    for name, role, req, browser in (
        (f"simulation_manifest_{season}.json", "simulation_manifest", True, False),
        (f"simulation_summary_{season}.csv", "simulation_summary", True, False),
        (f"simulation_summary_recentered_{season}.csv", "simulation_summary_recentered", True, False),
        (f"simulations_{season}.parquet", "simulations", True, False),
    ):
        if (root / name).exists():
            artifact_specs.append((role, name, req, browser))
    if recentered_path.exists():
        artifact_specs.append(("recentered_draws", recentered_path.name, True, False))
    if (root / "deep_band_accuracy.json").exists():
        artifact_specs.append(("deep_band_accuracy", "deep_band_accuracy.json", False, True))
    partition_dir = root / "simulations"
    if partition_dir.exists():
        for path in sorted(partition_dir.rglob("*.parquet")):
            rel = path.relative_to(root).as_posix()
            artifact_specs.append((f"draw_partition:{rel}", rel, False, False))

    cal_hashes = {}
    for key in (
        "calibration_hash",
        "wr_calibration_artifact_hash",
        "finish_probability_gate_hash",
        "segment_report_hash",
    ):
        if simulation_manifest.get(key):
            cal_hashes[key] = simulation_manifest[key]

    treatments_path = root / "contract_treatments.json"
    if treatments_path.exists():
        treatments = json.loads(treatments_path.read_text(encoding="utf-8"))
    else:
        treatments = {
            "selected": treatment_block([]),
            "incumbent": treatment_block([]),
            "new_player_v1_only": treatment_block([]),
        }

    manifest, digest = seal_staged_bundle(
        season=season,
        namespace=namespace,
        root=root,
        release_id=projection_run.get("run_id") or simulation_manifest.get("simulation_run_id"),
        application={
            "contract_version": contract["contract_version"],
            "contract_hash": contract["contract_hash"],
        },
        runs={
            "projection_run_id": projection_run["run_id"],
            "simulation_run_id": simulation_manifest.get("simulation_run_id"),
        },
        board={
            "selected_board_file_hash": sha256_file(root / selected_rel),
            "selected_points_vector_hash": selected_points_vector_hash(board_points),
        },
        simulation={
            "profile": "publish",
            "draw_count": int(simulation_manifest.get("draw_count") or 10000),
            "configuration_hash": _simulation_configuration_hash("publish"),
            "calibration_hashes": cal_hashes or {"placeholder": "0" * 64},
            "joint_donor_hash": simulation_manifest.get("joint_donors_hash") or ("0" * 64),
        },
        overlay={
            "simulated_player_population_hash": player_id_set_hash(overlay_ids),
            "simulated_player_count": len(set(overlay_ids)),
        },
        contract_treatments={
            "selected": treatments.get("selected") or treatment_block([]),
            "incumbent": treatments.get("incumbent") or treatment_block([]),
            "new_player_v1_only": treatments.get("new_player_v1_only") or treatment_block([]),
        },
        artifact_specs=artifact_specs,
    )
    copy_browser_consumed(root=root, manifest=manifest, manifest_sha256=digest)
    return {"manifest_sha256": digest, "release_id": manifest["bundle"]["release_id"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--artifact-namespace", required=True)
    args = parser.parse_args()
    result = complete_seal(args.season, args.artifact_namespace)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
