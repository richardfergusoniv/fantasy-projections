"""Tracked promotion receipts — immutable evidence that a release was activated."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.projection.active_release import atomic_write_json
from src.projection.contracts import REPO_ROOT
from src.projection.git_provenance import ProvenanceMode
from src.projection.release_bundle import validate_namespace


RECEIPT_SCHEMA_VERSION = "release_promotion_receipt_v1"


class PromotionReceiptError(ValueError):
    """Receipt is missing, malformed, untracked, or conflicts with a prior activation."""


def receipts_dir(season: int) -> Path:
    return Path(REPO_ROOT) / "draft_assistant" / "data" / "promotion_receipts" / str(int(season))


def receipt_path(season: int, namespace: str) -> Path:
    return receipts_dir(season) / f"{validate_namespace(namespace)}.json"


def validate_promotion_receipt(
    payload: Mapping[str, Any],
    *,
    season: int | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PromotionReceiptError("promotion receipt must be a JSON object")
    if payload.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise PromotionReceiptError(
            f"unsupported receipt schema_version: {payload.get('schema_version')!r}"
        )
    required = (
        "season",
        "namespace",
        "release_id",
        "manifest_sha256",
        "source_commit",
        "provenance_mode",
        "activated_at",
        "promotion_invariants_verdict",
    )
    missing = [key for key in required if payload.get(key) in (None, "")]
    if missing:
        raise PromotionReceiptError(f"promotion receipt missing fields: {missing}")
    if season is not None and int(payload["season"]) != int(season):
        raise PromotionReceiptError(
            f"receipt season {payload['season']} does not match requested {season}"
        )
    mode = str(payload["provenance_mode"])
    if mode not in ("initial", "restore"):
        raise PromotionReceiptError(f"unsupported provenance_mode: {mode!r}")
    digest = str(payload["manifest_sha256"]).strip().lower()
    if len(digest) != 64:
        raise PromotionReceiptError("receipt manifest_sha256 is not a sha256 digest")
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "season": int(payload["season"]),
        "namespace": validate_namespace(str(payload["namespace"])),
        "release_id": str(payload["release_id"]),
        "manifest_sha256": digest,
        "source_commit": str(payload["source_commit"]).strip(),
        "provenance_mode": mode,
        "activated_at": str(payload["activated_at"]),
        "promotion_invariants_verdict": str(payload["promotion_invariants_verdict"]),
    }


def build_promotion_receipt(
    *,
    season: int,
    namespace: str,
    release_id: str,
    manifest_sha256: str,
    source_commit: str,
    provenance_mode: ProvenanceMode,
    activated_at: str | None = None,
    promotion_invariants_verdict: str = "pass",
) -> dict[str, Any]:
    return validate_promotion_receipt(
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "season": int(season),
            "namespace": namespace,
            "release_id": str(release_id),
            "manifest_sha256": str(manifest_sha256).lower(),
            "source_commit": str(source_commit).strip(),
            "provenance_mode": provenance_mode,
            "activated_at": activated_at or datetime.now(timezone.utc).isoformat(),
            "promotion_invariants_verdict": promotion_invariants_verdict,
        },
        season=season,
    )


def read_promotion_receipt(season: int, namespace: str) -> dict[str, Any] | None:
    path = receipt_path(season, namespace)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PromotionReceiptError(f"promotion receipt is not valid JSON: {exc}") from exc
    return validate_promotion_receipt(payload, season=season)


def receipt_is_git_tracked(path: Path, *, cwd: Path | None = None) -> bool:
    """True when the receipt path is present in the git index (tracked)."""
    root = cwd or Path(REPO_ROOT)
    try:
        rel = path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def release_identity_matches(
    *,
    namespace: str,
    release_id: str,
    manifest_sha256: str,
    expected_namespace: str | None,
    expected_release_id: str | None,
    expected_manifest_sha256: str | None = None,
) -> bool:
    if not expected_namespace or not expected_release_id:
        return False
    if str(expected_namespace) != str(namespace):
        return False
    if str(expected_release_id) != str(release_id):
        return False
    if expected_manifest_sha256 in (None, ""):
        return True
    return str(expected_manifest_sha256).strip().lower() == str(manifest_sha256).strip().lower()


def tracked_receipt_authorizes_restore(
    *,
    season: int,
    namespace: str,
    release_id: str,
    manifest_sha256: str,
    cwd: Path | None = None,
) -> bool:
    """Authorize restore only from a well-formed, git-tracked, identity-matching receipt."""
    path = receipt_path(season, namespace)
    if not path.exists():
        return False
    if not receipt_is_git_tracked(path, cwd=cwd):
        return False
    try:
        receipt = read_promotion_receipt(season, namespace)
    except PromotionReceiptError:
        return False
    if receipt is None:
        return False
    return release_identity_matches(
        namespace=namespace,
        release_id=release_id,
        manifest_sha256=manifest_sha256,
        expected_namespace=receipt["namespace"],
        expected_release_id=receipt["release_id"],
        expected_manifest_sha256=receipt["manifest_sha256"],
    )


def derive_provenance_mode(
    *,
    season: int,
    namespace: str,
    release_id: str,
    manifest_sha256: str,
    cwd: Path | None = None,
) -> ProvenanceMode:
    """Derive ``initial`` vs ``restore`` from pointer history or a tracked receipt.

    Validation sidecars never authorize restore. Missing or malformed pointers
    fall through to receipt evidence; with neither, mode stays ``initial``.
    """
    from src.projection.active_release import ActiveReleaseError, read_active_pointer

    try:
        pointer = read_active_pointer(season)
    except ActiveReleaseError:
        pointer = None

    if pointer is not None:
        if release_identity_matches(
            namespace=namespace,
            release_id=release_id,
            manifest_sha256=manifest_sha256,
            expected_namespace=str(pointer.get("namespace")),
            expected_release_id=str(pointer.get("release_id")),
            expected_manifest_sha256=str(pointer.get("manifest_sha256")),
        ):
            return "restore"
        previous = pointer.get("previous") or {}
        if release_identity_matches(
            namespace=namespace,
            release_id=release_id,
            manifest_sha256=manifest_sha256,
            expected_namespace=previous.get("namespace"),
            expected_release_id=previous.get("release_id"),
            expected_manifest_sha256=previous.get("manifest_sha256"),
        ):
            return "restore"

    if tracked_receipt_authorizes_restore(
        season=season,
        namespace=namespace,
        release_id=release_id,
        manifest_sha256=manifest_sha256,
        cwd=cwd,
    ):
        return "restore"
    return "initial"


def write_promotion_receipt(receipt: Mapping[str, Any], *, cwd: Path | None = None) -> Path:
    """Persist a receipt. Idempotent for identical identity; fails on conflicting reuse."""
    validated = validate_promotion_receipt(receipt)
    dest = receipt_path(int(validated["season"]), validated["namespace"])
    if dest.exists():
        try:
            existing = read_promotion_receipt(int(validated["season"]), validated["namespace"])
        except PromotionReceiptError as exc:
            raise PromotionReceiptError(
                f"existing receipt at {dest} is malformed and blocks reuse: {exc}"
            ) from exc
        assert existing is not None
        same_identity = (
            existing["release_id"] == validated["release_id"]
            and existing["manifest_sha256"] == validated["manifest_sha256"]
            and existing["source_commit"] == validated["source_commit"]
        )
        if not same_identity:
            raise PromotionReceiptError(
                f"conflicting promotion receipt for namespace {validated['namespace']}: "
                f"existing release_id={existing['release_id']} "
                f"manifest_sha256={existing['manifest_sha256']} "
                f"conflicts with "
                f"release_id={validated['release_id']} "
                f"manifest_sha256={validated['manifest_sha256']}"
            )
        # Idempotent: identical release identity may rewrite metadata (mode/timestamp/verdict).
    atomic_write_json(dest, validated)
    return dest
