"""Atomically promote or roll back by replacing only the active pointer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.projection.active_release import (
    ActiveReleaseError,
    build_active_pointer,
    pointer_path,
    read_active_pointer,
    write_active_pointer,
)
from src.projection.evaluation.promotion_invariants import (
    copy_and_validate_public_browser_artifacts,
    validate_promotion_invariants,
    validate_sealed_promotion_invariants,
)
from src.projection.evaluation.release_bundle_validation import validate_release_bundle
from src.projection.release_bundle import (
    MANIFEST_FILENAME,
    ReleaseBundleError,
    bundle_root,
    load_sealed_manifest,
    public_release_dir,
    sha256_bytes,
    verify_artifact_hashes,
)
from src.projection.evaluation.accuracy_first import sha256_file


class PromoteReleaseError(RuntimeError):
    """Promotion refused because the sealed bundle is invalid or copies drifted."""


def _browser_roles(manifest: dict[str, Any]) -> dict[str, str]:
    namespace = manifest["bundle"]["namespace"]
    urls = {}
    for entry in manifest["artifacts"]:
        if entry.get("browser_consumed"):
            urls[entry["role"]] = f"data/releases/{namespace}/{entry['path']}"
    urls["manifest"] = f"data/releases/{namespace}/{MANIFEST_FILENAME}"
    return urls


def _snapshot_public_namespace(namespace: str) -> dict[str, bytes | None]:
    public = public_release_dir(namespace)
    if not public.exists():
        return {}
    snapshot: dict[str, bytes | None] = {}
    for path in public.rglob("*"):
        if path.is_file():
            rel = path.relative_to(public).as_posix()
            snapshot[rel] = path.read_bytes()
    return snapshot


def _restore_public_namespace(namespace: str, snapshot: dict[str, bytes | None]) -> None:
    import shutil

    public = public_release_dir(namespace)
    if public.exists():
        shutil.rmtree(public)
    if not snapshot:
        return
    public.mkdir(parents=True, exist_ok=True)
    for rel, data in snapshot.items():
        if data is None:
            continue
        dest = public / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)


def assert_public_copies_match(manifest: dict[str, Any], manifest_sha256: str) -> None:
    namespace = manifest["bundle"]["namespace"]
    public = public_release_dir(namespace)
    public_manifest = public / MANIFEST_FILENAME
    if not public_manifest.is_file():
        raise PromoteReleaseError(f"public sealed manifest missing: {public_manifest}")
    public_digest = sha256_bytes(public_manifest.read_bytes())
    if public_digest != manifest_sha256:
        raise PromoteReleaseError("public manifest copy hash does not match sealed bundle")
    for entry in manifest["artifacts"]:
        if not entry.get("browser_consumed"):
            continue
        path = public / entry["path"]
        if not path.is_file():
            raise PromoteReleaseError(f"public browser artifact missing: {path}")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise PromoteReleaseError(
                f"public copy hash mismatch for {entry['role']}: {actual} != {entry['sha256']}"
            )


def _record_file_state(path: Path) -> tuple[int, str] | None:
    if not path.exists():
        return None
    return (path.stat().st_mtime_ns, sha256_file(path))


def promote_release(season: int, artifact_namespace: str) -> dict[str, Any]:
    report = validate_release_bundle(
        season=season,
        namespace=artifact_namespace,
        require_active=False,
    )
    if report.get("verdict") != "pass":
        raise PromoteReleaseError(f"sealed bundle validation failed: {report}")

    root = bundle_root(season, artifact_namespace)
    manifest, digest = load_sealed_manifest(root)
    sealed_before = sha256_file(root / MANIFEST_FILENAME)
    verify_artifact_hashes(manifest, root=root)

    invariant_report = validate_sealed_promotion_invariants(
        season=season,
        namespace=artifact_namespace,
    )
    if invariant_report.get("verdict") != "pass":
        raise PromoteReleaseError(f"promotion invariants failed: {invariant_report}")

    public_before = _snapshot_public_namespace(manifest["bundle"]["namespace"])
    try:
        copy_and_validate_public_browser_artifacts(
            manifest,
            source_root=root,
            manifest_sha256=digest,
        )
    except Exception as exc:
        _restore_public_namespace(manifest["bundle"]["namespace"], public_before)
        raise PromoteReleaseError(f"public browser artifact promotion failed: {exc}") from exc

    invariant_after_copy = validate_promotion_invariants(
        season=season,
        namespace=artifact_namespace,
        include_git=False,
    )
    if invariant_after_copy.get("verdict") != "pass":
        _restore_public_namespace(manifest["bundle"]["namespace"], public_before)
        raise PromoteReleaseError(
            f"post-copy promotion invariants failed: {invariant_after_copy}"
        )

    current = None
    try:
        current = read_active_pointer(season)
    except ActiveReleaseError as exc:
        _restore_public_namespace(manifest["bundle"]["namespace"], public_before)
        raise PromoteReleaseError(f"malformed active pointer blocks promotion: {exc}") from exc

    previous = None
    if current is not None:
        previous = {"namespace": current["namespace"], "release_id": current["release_id"]}

    pointer = build_active_pointer(
        season=season,
        namespace=artifact_namespace,
        release_id=str(manifest["bundle"]["release_id"]),
        manifest_sha256=digest,
        previous=previous,
        browser_roles=_browser_roles(manifest),
    )
    try:
        write_active_pointer(pointer)
    except Exception as exc:
        _restore_public_namespace(manifest["bundle"]["namespace"], public_before)
        raise PromoteReleaseError(f"pointer write failed; public namespace restored: {exc}") from exc

    sealed_after = sha256_file(root / MANIFEST_FILENAME)
    if sealed_after != sealed_before:
        raise PromoteReleaseError("promotion rewrote the sealed bundle manifest")

    active_report = validate_release_bundle(
        season=season,
        namespace=artifact_namespace,
        require_active=True,
    )
    if active_report.get("verdict") != "pass":
        raise PromoteReleaseError(f"post-promotion require-active validation failed: {active_report}")

    return {
        "pointer": pointer,
        "pointer_path": str(pointer_path(season)),
        "manifest_sha256": digest,
        "validation": active_report,
        "promotion_invariants": invariant_after_copy,
    }


def rollback_release(season: int) -> dict[str, Any]:
    current = read_active_pointer(season)
    if current is None:
        raise PromoteReleaseError("no active pointer to roll back")
    previous = current.get("previous") or {}
    namespace = previous.get("namespace")
    if not namespace:
        raise PromoteReleaseError("active pointer has no previous namespace for rollback")
    result = promote_release(season, namespace)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--artifact-namespace", required=False)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    if args.rollback:
        result = rollback_release(args.season)
    else:
        if not args.artifact_namespace:
            parser.error("--artifact-namespace is required unless --rollback is set")
        result = promote_release(args.season, args.artifact_namespace)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
