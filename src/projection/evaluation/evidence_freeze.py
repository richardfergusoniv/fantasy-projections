"""Copy draw-stability evidence into an immutable frozen bundle with hashes."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.projection.contracts import MODEL_V3_DIR, OUTPUT_DIR, REPO_ROOT
from src.projection.evaluation.accuracy_first import sha256_file

FROZEN_ROOT = Path(MODEL_V3_DIR) / "frozen"

DEFAULT_EVIDENCE_FILES = (
    "draw_stability_intermediate_v20k_2026.json",
    "draw_count_decision.json",
    "decision_change_diagnostics_2026.json",
    "player_stability_diagnostics_2026.parquet",
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def freeze_draw_stability_evidence(
    *,
    freeze_id: str,
    season: int = 2026,
    source_dir: Path | None = None,
    evidence_files: tuple[str, ...] = DEFAULT_EVIDENCE_FILES,
    selected_board_hash: str | None = None,
    canonical_projection_run_id: str | None = None,
    reference_draw_count: int = 20000,
    nested_prefix_provenance_verdict: str | None = None,
) -> Path:
    """Copy evidence artifacts and write ``freeze_manifest.json``."""
    source_dir = source_dir or Path(MODEL_V3_DIR)
    dest_dir = FROZEN_ROOT / freeze_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    source_paths: dict[str, str] = {}
    sha256: dict[str, str] = {}
    for name in evidence_files:
        src = source_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing evidence artifact for freeze: {src}")
        dst = dest_dir / name
        shutil.copy2(src, dst)
        key = name
        source_paths[key] = _rel(src)
        sha256[key] = sha256_file(dst)

    decision_path = source_dir / "draw_count_decision.json"
    if nested_prefix_provenance_verdict is None and decision_path.exists():
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        nested_prefix_provenance_verdict = decision.get("provenance_verdict")

    manifest: dict[str, Any] = {
        "freeze_id": freeze_id,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "source_paths": source_paths,
        "sha256": sha256,
        "selected_board_hash": selected_board_hash,
        "canonical_projection_run_id": canonical_projection_run_id,
        "reference_draw_count": reference_draw_count,
        "nested_prefix_provenance_verdict": nested_prefix_provenance_verdict,
    }
    manifest_path = dest_dir / "freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["sha256"]["freeze_manifest.json"] = sha256_file(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def load_freeze_manifest(freeze_id: str) -> dict[str, Any]:
    path = FROZEN_ROOT / freeze_id / "freeze_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Freeze manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_evidence_manifest_hash(freeze_id: str) -> str:
    return sha256_file(FROZEN_ROOT / freeze_id / "freeze_manifest.json")
