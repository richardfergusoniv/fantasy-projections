"""Browser-readable active release pointer. Mutable status lives here, not on the bundle."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.projection.contracts import REPO_ROOT
from src.projection.release_bundle import (
    MANIFEST_FILENAME,
    ReleaseBundleError,
    public_release_dir,
    validate_namespace,
)


POINTER_SCHEMA_VERSION = "active_release_pointer_v1"
POINTER_STATUS_ACTIVE = "active"


class ActiveReleaseError(ValueError):
    """Pointer is missing, malformed, or inconsistent with a sealed bundle."""


def pointer_path(season: int) -> Path:
    return Path(REPO_ROOT) / "draft_assistant" / "data" / f"active_release_{int(season)}.json"


def pointer_public_base(namespace: str) -> str:
    return f"data/releases/{validate_namespace(namespace)}"


def validate_active_pointer(payload: Mapping[str, Any], *, season: int | None = None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ActiveReleaseError("active pointer must be a JSON object")
    if payload.get("schema_version") != POINTER_SCHEMA_VERSION:
        raise ActiveReleaseError(
            f"unsupported pointer schema_version: {payload.get('schema_version')!r}"
        )
    required = (
        "season",
        "status",
        "namespace",
        "release_id",
        "manifest_path",
        "manifest_sha256",
        "activated_at",
    )
    missing = [key for key in required if payload.get(key) in (None, "")]
    if missing:
        raise ActiveReleaseError(f"active pointer missing fields: {missing}")
    if payload.get("status") != POINTER_STATUS_ACTIVE:
        raise ActiveReleaseError(f"active pointer status must be 'active', got {payload.get('status')!r}")
    if season is not None and int(payload["season"]) != int(season):
        raise ActiveReleaseError(
            f"active pointer season {payload['season']} does not match requested {season}"
        )
    validate_namespace(str(payload["namespace"]))
    digest = str(payload["manifest_sha256"]).strip().lower()
    if len(digest) != 64:
        raise ActiveReleaseError("active pointer manifest_sha256 is not a sha256 digest")
    previous = payload.get("previous") or {}
    if previous:
        if not previous.get("namespace") or not previous.get("release_id"):
            raise ActiveReleaseError("previous pointer identity must include namespace and release_id")
        prev_hash = previous.get("manifest_sha256")
        if prev_hash not in (None, ""):
            prev_digest = str(prev_hash).strip().lower()
            if len(prev_digest) != 64:
                raise ActiveReleaseError("previous.manifest_sha256 is not a sha256 digest")
        else:
            prev_digest = None
    else:
        prev_digest = None
    public_base = payload.get("public_base") or pointer_public_base(str(payload["namespace"]))
    public_urls = payload.get("public_urls") or {}
    previous_block = None
    if previous.get("namespace"):
        previous_block = {
            "namespace": previous.get("namespace"),
            "release_id": previous.get("release_id"),
        }
        if prev_digest is not None:
            previous_block["manifest_sha256"] = prev_digest
    return {
        "schema_version": POINTER_SCHEMA_VERSION,
        "season": int(payload["season"]),
        "status": POINTER_STATUS_ACTIVE,
        "namespace": str(payload["namespace"]),
        "release_id": str(payload["release_id"]),
        "manifest_path": str(payload["manifest_path"]),
        "manifest_sha256": digest,
        "activated_at": str(payload["activated_at"]),
        "previous": previous_block,
        "public_base": public_base,
        "public_urls": dict(public_urls),
    }


def build_active_pointer(
    *,
    season: int,
    namespace: str,
    release_id: str,
    manifest_sha256: str,
    activated_at: str | None = None,
    previous: Mapping[str, Any] | None = None,
    browser_roles: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    ns = validate_namespace(namespace)
    public_base = pointer_public_base(ns)
    manifest_rel = f"{public_base}/{MANIFEST_FILENAME}"
    public_urls = dict(browser_roles or {})
    if "manifest" not in public_urls:
        public_urls["manifest"] = manifest_rel
    payload = {
        "schema_version": POINTER_SCHEMA_VERSION,
        "season": int(season),
        "status": POINTER_STATUS_ACTIVE,
        "namespace": ns,
        "release_id": str(release_id),
        "manifest_path": manifest_rel,
        "manifest_sha256": str(manifest_sha256).lower(),
        "activated_at": activated_at or datetime.now(timezone.utc).isoformat(),
        "previous": dict(previous) if previous else None,
        "public_base": public_base,
        "public_urls": public_urls,
    }
    return validate_active_pointer(payload, season=season)


def read_active_pointer(season: int, *, session: Any | None = None) -> dict[str, Any] | None:
    """Read the active release pointer from DB (preferred) or local filesystem."""
    from src.app.config import get_settings

    if session is not None:
        from src.app.persistence.release_pointers import ReleasePointerStore

        return ReleasePointerStore(session).read(season)

    settings = get_settings()
    if settings.app_env == "test":
        path = pointer_path(season)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ActiveReleaseError(f"active pointer is not valid JSON: {exc}") from exc
        return validate_active_pointer(payload, season=season)

    from src.app.persistence.release_pointers import try_db_read_release_pointer

    db_pointer = try_db_read_release_pointer(season)
    if db_pointer is not None:
        return db_pointer

    if settings.app_env == "production":
        return None

    path = pointer_path(season)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ActiveReleaseError(f"active pointer is not valid JSON: {exc}") from exc
    return validate_active_pointer(payload, season=season)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def write_active_pointer(payload: Mapping[str, Any], *, session: Any | None = None) -> Path:
    from src.app.config import get_settings

    pointer = validate_active_pointer(payload)
    season = int(pointer["season"])

    if session is not None:
        from src.app.persistence.release_pointers import ReleasePointerStore

        ReleasePointerStore(session).write(pointer)
    else:
        from src.app.persistence.release_pointers import try_db_write_release_pointer

        try_db_write_release_pointer(pointer)

    settings = get_settings()
    if settings.app_env == "production":
        return pointer_path(season)

    dest = pointer_path(season)
    atomic_write_json(dest, pointer)
    return dest


def derived_bundle_status(
    *,
    season: int,
    namespace: str,
    manifest_sha256: str,
    pointer: Mapping[str, Any] | None = None,
) -> str:
    """Inactive means the sealed bundle is not the one the pointer names."""
    current = pointer
    if current is None:
        try:
            current = read_active_pointer(season)
        except ActiveReleaseError:
            return "inactive"
    if current is None:
        return "inactive"
    if (
        str(current.get("namespace")) == str(namespace)
        and str(current.get("manifest_sha256")) == str(manifest_sha256)
        and current.get("status") == POINTER_STATUS_ACTIVE
    ):
        return "active"
    return "inactive"


def resolve_browser_urls(
    pointer: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    data_root: str,
) -> dict[str, str]:
    """Resolve browser-consumed artifacts from one frozen pointer + manifest."""
    if str(pointer.get("namespace")) != str(manifest["bundle"]["namespace"]):
        raise ActiveReleaseError("pointer namespace does not match sealed manifest")
    if int(pointer["season"]) != int(manifest["bundle"]["season"]):
        raise ActiveReleaseError("pointer season does not match sealed manifest")
    prefix = data_root.rstrip("/")
    public_base = str(pointer.get("public_base") or "").lstrip("/")
    # data_root is 'data' or '../data'; public_base is 'data/releases/ns'.
    namespace = pointer["namespace"]
    urls: dict[str, str] = {}
    for entry in manifest["artifacts"]:
        if not entry.get("browser_consumed"):
            continue
        urls[entry["role"]] = f"{prefix}/releases/{namespace}/{entry['path']}"
    urls["manifest"] = f"{prefix}/releases/{namespace}/{MANIFEST_FILENAME}"
    return urls


def frozen_load_session(pointer: Mapping[str, Any]) -> dict[str, Any]:
    """Capture namespace once so a mid-load pointer change cannot mix bundles."""
    validated = validate_active_pointer(pointer)
    return {
        "namespace": validated["namespace"],
        "release_id": validated["release_id"],
        "manifest_sha256": validated["manifest_sha256"],
        "season": validated["season"],
        "public_base": validated["public_base"],
        "manifest_path": validated["manifest_path"],
    }
