"""Atomic production release pointers — rollback repoints, not republish."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.draft_assistant.prepare import DRAFT_DATA_DIR
from src.projection.contracts import MODEL_V3_DIR, OUTPUT_DIR, REPO_ROOT
from src.projection.evaluation.accuracy_first import sha256_file

RELEASES_DIR = Path(MODEL_V3_DIR) / "releases"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def production_bundle_paths(season: int) -> dict[str, Path]:
    return {
        "simulation_manifest": Path(MODEL_V3_DIR) / f"simulation_manifest_{season}.json",
        "release_report": Path(MODEL_V3_DIR) / f"release_report_{season}.json",
        "release_report_simulation": Path(MODEL_V3_DIR) / f"release_report_simulation_{season}.json",
        "release_report_board": Path(MODEL_V3_DIR) / f"release_report_board_{season}.json",
        "players": Path(DRAFT_DATA_DIR) / f"players_{season}.json",
        "projection_run": Path(OUTPUT_DIR) / f"projection_run_{season}.json",
    }


def snapshot_production_bundle(season: int, *, label: str, profile: str) -> dict[str, Any]:
    paths = production_bundle_paths(season)
    files: dict[str, Any] = {}
    for key, path in paths.items():
        files[key] = {
            "path": _rel(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
        }
    sim_manifest = None
    if paths["simulation_manifest"].exists():
        sim_manifest = json.loads(paths["simulation_manifest"].read_text(encoding="utf-8"))
    return {
        "label": label,
        "profile": profile,
        "season": season,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "draw_count": (sim_manifest or {}).get("draw_count"),
        "simulation_run_id": (sim_manifest or {}).get("simulation_run_id")
        or (sim_manifest or {}).get("canonical_projection_run_id"),
        "canonical_projection_run_id": (sim_manifest or {}).get("canonical_projection_run_id"),
        "selected_board_hash": (sim_manifest or {}).get("selected_board_hash"),
        "files": files,
    }


def write_release_pointer(season: int, bundle: dict[str, Any], *, role: str) -> Path:
    RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    path = RELEASES_DIR / f"release_{season}_{role}.json"
    path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return path


def read_release_pointer(season: int, *, role: str = "current") -> dict[str, Any] | None:
    path = RELEASES_DIR / f"release_{season}_{role}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_release_pointers(season: int) -> dict[str, Path]:
    """Bootstrap current/previous pointers from live production if missing."""
    current_path = RELEASES_DIR / f"release_{season}_current.json"
    if not current_path.exists():
        bundle = snapshot_production_bundle(
            season,
            label="bootstrap_current",
            profile="provisional_1000",
        )
        write_release_pointer(season, bundle, role="current")
    previous_path = RELEASES_DIR / f"release_{season}_previous_1k.json"
    if not previous_path.exists():
        bundle = snapshot_production_bundle(
            season,
            label="bootstrap_previous_1k",
            profile="provisional_1000",
        )
        write_release_pointer(season, bundle, role="previous_1k")
    return {
        "current": current_path,
        "previous_1k": previous_path,
    }


def restore_release_pointer(season: int, *, from_role: str = "previous_1k") -> dict[str, Any]:
    """Operational rollback: repoint production metadata only (no pipeline rerun)."""
    pointer = read_release_pointer(season, role=from_role)
    if pointer is None:
        raise FileNotFoundError(f"Release pointer missing: release_{season}_{from_role}.json")
    current = read_release_pointer(season, role="current")
    if current is not None:
        write_release_pointer(season, current, role="previous_1k")
    write_release_pointer(season, pointer, role="current")
    return pointer
