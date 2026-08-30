#!/usr/bin/env python3
"""Copy a sealed bundle to a new namespace and re-seal (for rollback rehearsal)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.evaluation.release_bundle_validation import validate_release_bundle
from src.projection.release_bundle_publish import copy_browser_consumed
from src.projection.release_bundle import (
    MANIFEST_FILENAME,
    VALIDATION_FILENAME,
    bundle_root,
    load_sealed_manifest,
    seal_manifest,
)


def copy_reseal(*, season: int, source_ns: str, dest_ns: str, release_id: str | None = None) -> dict:
    src = bundle_root(season, source_ns)
    dst = bundle_root(season, dest_ns)
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for sidecar in (MANIFEST_FILENAME, VALIDATION_FILENAME):
        path = dst / sidecar
        if path.exists():
            path.unlink()
    manifest, _ = load_sealed_manifest(src)
    manifest["bundle"]["namespace"] = dest_ns
    manifest["bundle"]["release_id"] = release_id or str(uuid.uuid4())
    manifest["artifacts"] = [
        {**entry, "path": entry["path"]}
        for entry in manifest["artifacts"]
    ]
    sealed, digest = seal_manifest(manifest, root=dst)
    copy_browser_consumed(root=dst, manifest=sealed, manifest_sha256=digest)
    report = validate_release_bundle(season=season, namespace=dest_ns, require_active=False)
    return {
        "source_namespace": source_ns,
        "dest_namespace": dest_ns,
        "manifest_sha256": digest,
        "release_id": sealed["bundle"]["release_id"],
        "validation_verdict": report.get("verdict"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--source-namespace", required=True)
    parser.add_argument("--dest-namespace", required=True)
    parser.add_argument("--release-id", default=None)
    args = parser.parse_args()
    result = copy_reseal(
        season=args.season,
        source_ns=args.source_namespace,
        dest_ns=args.dest_namespace,
        release_id=args.release_id,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["validation_verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
