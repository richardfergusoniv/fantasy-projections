"""Hash production contracts before/after every shadow run; fail on drift."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.projection.accuracy_application import (
    DEFAULT_CONTRACT_PATH,
    DEFAULT_FREEZE_PATH,
    DEFAULT_WEIGHTS_PATH,
)
from src.projection.active_release import pointer_path, read_active_pointer
from src.projection.contracts import REPO_ROOT
from src.projection.evaluation.accuracy_first import sha256_file
from src.projection.release_bundle import MANIFEST_FILENAME, public_release_dir
from src.projection.shadow.forbidden import assert_input_path_allowed

PRODUCTION_SEASON = 2026


class ProductionDriftError(RuntimeError):
    """A production artifact changed during a shadow run."""


def _optional_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    assert_input_path_allowed(path)
    return sha256_file(path)


def _active_sealed_manifest_path(season: int = PRODUCTION_SEASON) -> Path | None:
    pointer = read_active_pointer(season)
    if not pointer:
        return None
    rel = str(pointer.get("manifest_path") or "")
    if rel:
        candidate = Path(REPO_ROOT) / rel
        if candidate.is_file():
            return candidate
    namespace = pointer.get("namespace")
    if namespace:
        public = public_release_dir(str(namespace)) / MANIFEST_FILENAME
        if public.is_file():
            return public
    return None


def snapshot_production_artifacts(
    *,
    season: int = PRODUCTION_SEASON,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    freeze_path: Path = DEFAULT_FREEZE_PATH,
) -> dict[str, Any]:
    """Hash production ensemble weights, contract, pointer, and sealed manifest."""
    pointer = pointer_path(season)
    sealed = _active_sealed_manifest_path(season)
    return {
        "ensemble_weights": {
            "path": str(weights_path).replace("\\", "/"),
            "sha256": _optional_hash(Path(weights_path)),
        },
        "application_contract": {
            "path": str(contract_path).replace("\\", "/"),
            "sha256": _optional_hash(Path(contract_path)),
        },
        "active_pointer": {
            "path": str(pointer).replace("\\", "/"),
            "sha256": _optional_hash(pointer),
        },
        "active_sealed_manifest": {
            "path": None if sealed is None else str(sealed).replace("\\", "/"),
            "sha256": None if sealed is None else sha256_file(sealed),
        },
        "accuracy_first_freeze_manifest": {
            "path": str(freeze_path).replace("\\", "/"),
            "sha256": _optional_hash(Path(freeze_path)),
        },
    }


def assert_production_unchanged(
    before: dict[str, Any],
    after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare two snapshots; any hash change fails the run."""
    after = after or snapshot_production_artifacts()
    changed = []
    for key in before:
        b = (before.get(key) or {}).get("sha256")
        a = (after.get(key) or {}).get("sha256")
        if b != a:
            changed.append({"artifact": key, "before": b, "after": a})
    if changed:
        raise ProductionDriftError(
            "Production artifacts changed during shadow run: "
            + json.dumps(changed)
        )
    return {
        "production_weights_unchanged": True,
        "before": before,
        "after": after,
        "changed": [],
    }
