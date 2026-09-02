#!/usr/bin/env python3
"""Upload a sealed release bundle namespace to Supabase Storage with hash verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.config import get_settings
from src.app.storage.release_bundle import (
    artifact_storage_uri_for_path,
    bundle_storage_uri_for_namespace,
    upload_bytes_to_storage,
)
from src.projection.active_release import build_active_pointer, write_active_pointer
from src.projection.release_bundle import MANIFEST_FILENAME, load_sealed_manifest, validate_namespace


def _content_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".csv":
        return "text/csv"
    if path.suffix == ".parquet":
        return "application/vnd.apache.parquet"
    return "application/octet-stream"


def upload_namespace(
    *,
    bundle_root: Path,
    season: int,
    namespace: str,
    write_pointer: bool,
) -> dict:
    namespace = validate_namespace(namespace)
    manifest, manifest_digest = load_sealed_manifest(bundle_root)
    uploaded: list[dict] = []
    for entry in manifest.get("artifacts", []):
        rel = str(entry.get("path") or "")
        if not rel:
            continue
        source = bundle_root / rel
        if not source.is_file():
            raise FileNotFoundError(f"missing artifact for role {entry.get('role')}: {source}")
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != str(entry.get("sha256", "")).lower():
            raise ValueError(
                f"hash mismatch for {rel}: manifest={entry.get('sha256')} computed={digest}"
            )
        key = f"releases/{namespace}/{rel}"
        uri = upload_bytes_to_storage(content, key=key, content_type=_content_type(source))
        uploaded.append({"role": entry.get("role"), "path": rel, "uri": uri, "sha256": digest})

    manifest_bytes = (bundle_root / MANIFEST_FILENAME).read_bytes()
    manifest_uri = upload_bytes_to_storage(
        manifest_bytes,
        key=f"releases/{namespace}/{MANIFEST_FILENAME}",
        content_type="application/json",
    )
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_digest:
        raise ValueError("manifest digest mismatch after read")

    pointer = build_active_pointer(
        season=season,
        namespace=namespace,
        release_id=str(manifest.get("bundle", {}).get("release_id") or ""),
        manifest_sha256=manifest_digest,
    )
    pointer["manifest_storage_uri"] = manifest_uri
    if write_pointer:
        write_active_pointer(pointer)

    return {
        "namespace": namespace,
        "season": season,
        "manifest_uri": manifest_uri,
        "manifest_sha256": manifest_digest,
        "artifact_count": len(uploaded),
        "artifacts": uploaded,
        "pointer_written": write_pointer,
        "default_manifest_uri": bundle_storage_uri_for_namespace(namespace),
        "sample_artifact_uri": artifact_storage_uri_for_path(namespace, uploaded[0]["path"])
        if uploaded
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--write-pointer", action="store_true")
    args = parser.parse_args()

    get_settings()
    report = upload_namespace(
        bundle_root=args.bundle_root,
        season=args.season,
        namespace=args.namespace,
        write_pointer=args.write_pointer,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
