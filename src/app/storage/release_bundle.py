"""Resolve sealed release manifests and artifacts from local paths or S3 URIs."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from src.app.artifacts.store import LOCAL_SCHEME, S3_SCHEME, ArtifactError, get_artifact_store
from src.app.config import get_settings
from src.projection.contracts import MODEL_V3_DIR, REPO_ROOT
from src.projection.release_bundle import (
    MANIFEST_FILENAME,
    ReleaseBundleError,
    load_sealed_manifest,
    public_release_dir,
    sha256_file,
    validate_namespace,
)


class ReleaseBundleResolver:
    """Load manifest bytes and artifact paths/URIs for a sealed release namespace."""

    def __init__(self, *, season: int, namespace: str, manifest_storage_uri: str | None = None) -> None:
        self.season = season
        self.namespace = validate_namespace(namespace)
        self.manifest_storage_uri = manifest_storage_uri

    def local_roots(self) -> list[Path]:
        return [
            public_release_dir(self.namespace),
            Path(MODEL_V3_DIR) / "release_bundles" / f"season={self.season}" / f"namespace={self.namespace}",
        ]

    def _read_bytes(self, uri: str | None, *, local_path: Path | None = None) -> bytes:
        if uri and uri.startswith(S3_SCHEME):
            return get_artifact_store().get_bytes(uri)
        if uri and uri.startswith(LOCAL_SCHEME):
            path = Path(uri.removeprefix(LOCAL_SCHEME))
            return path.read_bytes()
        if local_path is not None and local_path.is_file():
            return local_path.read_bytes()
        raise ReleaseBundleError(f"artifact not readable: uri={uri!r} path={local_path}")

    def load_manifest(self) -> tuple[dict[str, Any], str, Path | None]:
        """Return (manifest dict, sha256 digest, local bundle root if known)."""
        if self.manifest_storage_uri:
            raw = self._read_bytes(self.manifest_storage_uri)
            digest = hashlib.sha256(raw).hexdigest()
            manifest = json.loads(raw.decode("utf-8"))
            return manifest, digest, None

        for root in self.local_roots():
            manifest_path = root / MANIFEST_FILENAME
            if manifest_path.is_file():
                manifest, digest = load_sealed_manifest(root)
                return manifest, digest, root
        raise ReleaseBundleError(f"no manifest for namespace {self.namespace}")

    def resolve_artifact(
        self,
        manifest: dict[str, Any],
        role: str,
        *,
        primary_root: Path | None,
    ) -> tuple[bytes | None, str | None, Path | None]:
        """Return (bytes, storage_uri, local_path) for a manifest role."""
        entry = next((row for row in manifest.get("artifacts", []) if row.get("role") == role), None)
        if entry is None:
            return None, None, None
        rel = str(entry.get("path") or "")
        if not rel or ".." in rel.replace("\\", "/"):
            raise ReleaseBundleError(f"path traversal rejected for role {role}")

        storage_uri = entry.get("storage_uri")
        if isinstance(storage_uri, str) and storage_uri.startswith((S3_SCHEME, LOCAL_SCHEME)):
            return self._read_bytes(storage_uri), storage_uri, None

        if primary_root is not None:
            candidate = primary_root / rel
            if candidate.is_file():
                return candidate.read_bytes(), None, candidate
        for root in self.local_roots():
            candidate = root / rel
            if candidate.is_file():
                return candidate.read_bytes(), None, candidate
        return None, None, None

    def verify_role_hash(self, manifest: dict[str, Any], role: str, content: bytes) -> bool:
        entry = next((row for row in manifest.get("artifacts", []) if row.get("role") == role), None)
        if entry is None:
            return False
        digest = hashlib.sha256(content).hexdigest()
        return digest == str(entry.get("sha256", "")).lower()

    def materialize_to_temp(self, content: bytes, suffix: str = "") -> Path:
        fd, name = tempfile.mkstemp(suffix=suffix)
        path = Path(name)
        try:
            path.write_bytes(content)
        finally:
            import os

            os.close(fd)
        return path

    def sha256_local(self, path: Path) -> str:
        return sha256_file(path)


def bundle_storage_uri_for_namespace(namespace: str) -> str:
    """Default S3 key for a release manifest under releases/{namespace}/."""
    settings = get_settings()
    key = f"releases/{validate_namespace(namespace)}/{MANIFEST_FILENAME}"
    return f"{S3_SCHEME}{settings.s3_bucket}/{key}"


def artifact_storage_uri_for_path(namespace: str, rel_path: str) -> str:
    settings = get_settings()
    key = f"releases/{validate_namespace(namespace)}/{rel_path.lstrip('/')}"
    return f"{S3_SCHEME}{settings.s3_bucket}/{key}"


def upload_bytes_to_storage(content: bytes, *, key: str, content_type: str) -> str:
    """Upload raw bytes to the configured artifact backend under an explicit key."""
    settings = get_settings()
    if settings.artifact_backend == "s3":
        import boto3
        from botocore.config import Config

        session = boto3.session.Session(
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            region_name=settings.s3_region,
        )
        client = session.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            config=Config(s3={"addressing_style": "path"}),
        )
        client.put_object(
            Bucket=settings.s3_bucket,
            Key=key,
            Body=content,
            ContentType=content_type,
        )
        return f"{S3_SCHEME}{settings.s3_bucket}/{key}"

    root = Path(settings.artifact_local_root)
    dest = root / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return f"{LOCAL_SCHEME}{dest.as_posix()}"


def probe_storage_round_trip() -> dict[str, Any]:
    try:
        store = get_artifact_store()
        uri = store.put_json({"probe": "readiness"})
        echoed = store.get_json(uri)
        healthy = echoed == {"probe": "readiness"}
        return {"status": "healthy" if healthy else "degraded", "writable": True, "readable": healthy}
    except (ArtifactError, OSError) as exc:
        return {"status": "degraded", "writable": False, "readable": False, "detail": type(exc).__name__}
