#!/usr/bin/env python3
"""Phase 1 production rehearsal helpers: inspect, verify surfaces, compare hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.active_release import pointer_path, read_active_pointer
from src.projection.release_bundle import (
    MANIFEST_FILENAME,
    VALIDATION_FILENAME,
    bundle_root,
    load_sealed_manifest,
    public_release_dir,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(season: int, namespace: str) -> dict:
    root = bundle_root(season, namespace)
    manifest, digest = load_sealed_manifest(root)
    validation_path = root / VALIDATION_FILENAME
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else None
    public = public_release_dir(namespace)
    browser_roles = [
        entry["role"]
        for entry in manifest["artifacts"]
        if entry.get("browser_consumed")
    ]
    public_files = {}
    for entry in manifest["artifacts"]:
        if not entry.get("browser_consumed"):
            continue
        rel = entry["path"]
        pub = public / rel
        public_files[rel] = {
            "listed_sha256": entry["sha256"],
            "public_exists": pub.is_file(),
            "public_sha256": sha256_file(pub) if pub.is_file() else None,
            "match": pub.is_file() and sha256_file(pub) == entry["sha256"],
        }
    return {
        "bundle_root": str(root),
        "manifest_sha256": digest,
        "release_id": manifest["bundle"]["release_id"],
        "model_id": manifest["bundle"]["model_id"],
        "draw_count": manifest["simulation"]["draw_count"],
        "profile": manifest["simulation"]["profile"],
        "artifact_count": len(manifest["artifacts"]),
        "browser_roles": browser_roles,
        "contract_treatments": manifest["contract_treatments"],
        "validation_verdict": (validation or {}).get("verdict"),
        "validation_derived_status": (validation or {}).get("derived_status"),
        "public_files": public_files,
        "manifest": manifest,
    }


def active_state(season: int) -> dict:
    pointer = read_active_pointer(season)
    return {
        "pointer_path": str(pointer_path(season)),
        "pointer": pointer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--artifact-namespace", required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--active", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        report = inspect(args.season, args.artifact_namespace)
        # omit full manifest from stdout summary; still include key fields
        summary = {k: v for k, v in report.items() if k != "manifest"}
        print(json.dumps(summary, indent=2))
        print("\n--- manifest (bundle identity) ---")
        print(json.dumps(report["manifest"]["bundle"], indent=2))
        print(json.dumps(report["manifest"]["board"], indent=2))
        print(json.dumps(report["manifest"]["overlay"], indent=2))
    if args.active:
        print(json.dumps(active_state(args.season), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
