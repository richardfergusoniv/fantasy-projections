#!/usr/bin/env python3
"""Machine-readable inventory of weekly-v2 artifacts, entrypoints, and readiness sites."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SIBLING = REPO.parent / "fantasy-projections-2"

from src.app.projections.weekly_manifest import (  # noqa: E402
    OPTIONAL_ARTIFACTS,
    REQUIRED_MODEL_ARTIFACTS,
    sha256_file,
    validate_manifest,
)
from src.app.projections.weekly_v2_bridge import weekly_v2_readiness  # noqa: E402
from src.projection.weekly.config.paths import MODELS_DIR, OUTPUTS_DIR  # noqa: E402


ENTRYPOINTS = {
    "ingest": "scripts/weekly_v2_ingest.py",
    "build_features": "scripts/weekly_v2_build_features.py",
    "train": "scripts/weekly_v2_train.py",
    "evaluate": "scripts/weekly_v2_evaluate.py",
    "project": "scripts/weekly_v2_project.py",
    "inventory": "scripts/weekly_v2_inventory.py",
    "audit_features": "scripts/audit_weekly_features.py",
    "app_weekly_promote": "src/app/projections/weekly_run.py::WeeklyProjectionService.promote_week",
    "app_inference": "src/app/projections/weekly_inference.py::run_weekly_inference",
    "readiness": "src/app/projections/weekly_v2_bridge.py::weekly_v2_readiness",
}

READINESS_SITES = [
    "src/app/projections/weekly_v2_bridge.py",
    "src/app/api/v1/operations.py",
    "src/app/league/sleeper/shadow_sync.py",
    "src/app/jobs/handlers.py",
    "web/src/screens/Operations.tsx",
]


def _artifact_record(path: Path, *, required: bool) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False, "required": required}
    return {
        "path": str(path),
        "exists": True,
        "required": required,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
    }


def scan_tree(root: Path, label: str) -> dict:
    records: list[dict] = []
    if not root.exists():
        return {"label": label, "root": str(root), "exists": False, "artifacts": records}
    for pattern in ("**/*.joblib", "**/*.json", "**/*.parquet"):
        for path in sorted(root.glob(pattern)):
            name = path.name
            required = name in REQUIRED_MODEL_ARTIFACTS
            optional = name in OPTIONAL_ARTIFACTS or name.endswith(".meta.json")
            records.append(_artifact_record(path, required=required or optional))
    return {"label": label, "root": str(root), "exists": True, "artifacts": records}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory weekly-v2 artifacts and readiness")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, default=1)
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "weekly_v2_inventory.json")
    args = parser.parse_args(argv)

    validation = validate_manifest(args.season)
    readiness = weekly_v2_readiness(args.season, args.week)
    sibling_manifest = SIBLING / "models" / "training_manifest.json"

    inventory = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "target_season": args.season,
        "target_week": args.week,
        "required_model_artifacts": list(REQUIRED_MODEL_ARTIFACTS),
        "optional_artifacts": list(OPTIONAL_ARTIFACTS),
        "repos": {
            "local": scan_tree(MODELS_DIR, "local_models"),
            "local_outputs": scan_tree(OUTPUTS_DIR, "local_outputs"),
            "sibling": scan_tree(SIBLING / "models", "sibling_models"),
        },
        "sibling_training_manifest_exists": sibling_manifest.exists(),
        "manifest_validation": validation.to_dict(),
        "readiness": readiness.to_dict(),
        "missing_reason": (
            f"All {len(REQUIRED_MODEL_ARTIFACTS)} required joblib files absent under {MODELS_DIR}"
            if validation.missing_artifacts == REQUIRED_MODEL_ARTIFACTS
            else list(validation.failures) + list(validation.missing_artifacts)
        ),
        "entrypoints": {
            name: {"path": path, "exists": (REPO / path.split("::")[0]).exists()}
            for name, path in ENTRYPOINTS.items()
        },
        "readiness_sites": READINESS_SITES,
        "rookie_calibration_tuning_required_for_publication": False,
        "rookie_calibration_tuning_notes": (
            "Rookie models, calibration.json, and tuning_selection.json are optional/degraded "
            "for weekly publication; volume/efficiency/team_totals are required."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(json.dumps(inventory, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
