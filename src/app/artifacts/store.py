"""Immutable artifact storage adapters.

Local and S3 backends are deliberately contract-identical: same key
derivation, same URI parsing rules, same idempotent-put semantics, same error
types. That way a deployment can switch ``ARTIFACT_BACKEND`` without changing
any caller behaviour or any stored URI's meaning.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from src.app.config import get_settings

LOCAL_SCHEME = "local://"
S3_SCHEME = "s3://"

ARTIFACT_KEY_PREFIX = "artifacts"

#: Envelope identity for JSON artifacts written through :meth:`put_json`.
ARTIFACT_ENVELOPE_SCHEMA = "fantasy.app.artifact.envelope"
ARTIFACT_ENVELOPE_VERSION = 1

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


class ArtifactError(Exception):
    """Base class for artifact store failures."""


class ArtifactPathError(ArtifactError, ValueError):
    """Raised when a URI or key escapes, or cannot address, the artifact root."""


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def content_suffix(content_type: str) -> str:
    return ".json" if "json" in content_type else ".bin"


def derive_artifact_key(digest: str, content_type: str) -> str:
    """Content-addressed key, identical on every backend."""
    return f"{ARTIFACT_KEY_PREFIX}/{digest[:2]}/{digest[2:4]}/{digest}{content_suffix(content_type)}"


def build_manifest(
    payload_digest: str,
    *,
    schema: str,
    schema_version: int,
    inputs: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stamp schema identity, inputs, provenance and a UTC-aware timestamp."""
    return {
        "schema": schema,
        "schema_version": schema_version,
        "created_at": datetime.now(UTC).isoformat(),
        "content_sha256": payload_digest,
        "inputs": dict(inputs or {}),
        "provenance": dict(provenance or {}),
    }


def wrap_json_payload(
    payload: Any,
    *,
    inputs: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return ``(payload_digest, envelope)`` for a JSON artifact."""
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    envelope = {
        "artifact_schema": ARTIFACT_ENVELOPE_SCHEMA,
        "manifest": build_manifest(
            digest,
            schema=ARTIFACT_ENVELOPE_SCHEMA,
            schema_version=ARTIFACT_ENVELOPE_VERSION,
            inputs=inputs,
            provenance=provenance,
        ),
        "payload": payload,
    }
    return digest, envelope


def unwrap_json_payload(decoded: Any) -> Any:
    """Return the payload, transparently unwrapping a manifest envelope.

    Artifacts written before envelopes existed are plain payloads and are
    returned unchanged, so existing readers keep working.
    """
    if (
        isinstance(decoded, dict)
        and decoded.get("artifact_schema") == ARTIFACT_ENVELOPE_SCHEMA
        and "payload" in decoded
    ):
        return decoded["payload"]
    return decoded


def _validate_relative_key(key: str) -> PurePosixPath:
    """Reject anything that is not a plain relative key under the root."""
    if not key or not key.strip():
        raise ArtifactPathError("Artifact key must not be empty")
    normalized = key.replace("\\", "/").strip()
    if not normalized:
        raise ArtifactPathError("Artifact key must not be empty")
    if normalized.startswith("/") or normalized.startswith("//"):
        raise ArtifactPathError(f"Absolute artifact keys are not allowed: {key!r}")
    if _WINDOWS_DRIVE.match(normalized):
        raise ArtifactPathError(f"Drive-qualified artifact keys are not allowed: {key!r}")
    pure = PurePosixPath(normalized)
    if any(part == ".." for part in pure.parts):
        raise ArtifactPathError(f"Artifact key must not contain '..' segments: {key!r}")
    if "\x00" in normalized:
        raise ArtifactPathError("Artifact key must not contain NUL bytes")
    return pure


class ArtifactStore(ABC):
    @abstractmethod
    def put_bytes(self, content: bytes, *, content_type: str = "application/octet-stream") -> str:
        raise NotImplementedError

    @abstractmethod
    def put_json(
        self,
        payload: Any,
        *,
        inputs: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_bytes(self, uri: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def get_json(self, uri: str) -> Any:
        raise NotImplementedError

    def get_manifest(self, uri: str) -> dict[str, Any] | None:
        """Return the stamped manifest for a JSON artifact, if it has one."""
        decoded = json.loads(self.get_bytes(uri).decode("utf-8"))
        if isinstance(decoded, dict) and decoded.get("artifact_schema") == ARTIFACT_ENVELOPE_SCHEMA:
            manifest = decoded.get("manifest")
            return manifest if isinstance(manifest, dict) else None
        return None


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: str | Path | None = None) -> None:
        settings = get_settings()
        self.root = Path(root or settings.artifact_local_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._resolved_root = self.root.resolve()

    def _path_for_key(self, key: str) -> Path:
        return self.root / _validate_relative_key(key).as_posix()

    def put_bytes(self, content: bytes, *, content_type: str = "application/octet-stream") -> str:
        digest = hashlib.sha256(content).hexdigest()
        return self._write(digest, content, content_type=content_type)

    def put_json(
        self,
        payload: Any,
        *,
        inputs: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> str:
        digest, envelope = wrap_json_payload(payload, inputs=inputs, provenance=provenance)
        return self._write(digest, canonical_json_bytes(envelope), content_type="application/json")

    def _write(self, digest: str, content: bytes, *, content_type: str) -> str:
        key = derive_artifact_key(digest, content_type)
        path = self._path_for_key(key)
        # Content-addressed: an existing key already holds identical content,
        # so a repeat put is a no-op rather than a rewrite.
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(path, content)
        return f"{LOCAL_SCHEME}{path.as_posix()}"

    def get_bytes(self, uri: str) -> bytes:
        path = self._resolve(uri)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise ArtifactError(f"Artifact not found: {uri}") from exc

    def get_json(self, uri: str) -> Any:
        return unwrap_json_payload(json.loads(self.get_bytes(uri).decode("utf-8")))

    def _resolve(self, uri: str) -> Path:
        """Map a URI to a path that is provably inside the artifact root."""
        if uri.startswith(S3_SCHEME):
            raise ArtifactPathError(f"Unsupported URI for local store: {uri}")
        remainder = uri.removeprefix(LOCAL_SCHEME) if uri.startswith(LOCAL_SCHEME) else uri
        if not remainder:
            raise ArtifactPathError("Artifact URI must reference a key")
        normalized = remainder.replace("\\", "/")
        if any(part == ".." for part in PurePosixPath(normalized).parts):
            raise ArtifactPathError(f"Artifact URI must not contain '..' segments: {uri!r}")
        if "\x00" in normalized:
            raise ArtifactPathError("Artifact URI must not contain NUL bytes")

        is_absolute = normalized.startswith("/") or bool(_WINDOWS_DRIVE.match(normalized))
        candidates = [Path(normalized)] if is_absolute else [Path(normalized), self.root / normalized]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved == self._resolved_root or self._resolved_root in resolved.parents:
                return resolved
        raise ArtifactPathError(
            f"Artifact URI escapes the configured artifact root: {uri!r}"
        )


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Write via a same-directory temp file, fsync, then rename.

    A crash or exception mid-write leaves the temp file behind, never a
    truncated file at ``path``, so a reader can never observe a partial
    artifact at its content-addressed location.
    """
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


class S3ArtifactStore(ArtifactStore):
    def __init__(self) -> None:
        import boto3

        settings = get_settings()
        self.bucket = settings.s3_bucket
        session = boto3.session.Session(
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            region_name=settings.s3_region,
        )
        self.client = session.client("s3", endpoint_url=settings.s3_endpoint_url)

    def put_bytes(self, content: bytes, *, content_type: str = "application/octet-stream") -> str:
        digest = hashlib.sha256(content).hexdigest()
        return self._write(digest, content, content_type=content_type)

    def put_json(
        self,
        payload: Any,
        *,
        inputs: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> str:
        digest, envelope = wrap_json_payload(payload, inputs=inputs, provenance=provenance)
        return self._write(digest, canonical_json_bytes(envelope), content_type="application/json")

    def _write(self, digest: str, content: bytes, *, content_type: str) -> str:
        key = derive_artifact_key(digest, content_type)
        _validate_relative_key(key)
        if not self._exists(key):
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=content,
                ContentType=content_type,
            )
        return f"{S3_SCHEME}{self.bucket}/{key}"

    def _exists(self, key: str) -> bool:
        head = getattr(self.client, "head_object", None)
        if head is None:
            return False
        try:
            head(Bucket=self.bucket, Key=key)
        except Exception:  # noqa: BLE001 - any failure means "write it"
            return False
        return True

    def get_bytes(self, uri: str) -> bytes:
        bucket, key = self._parse(uri)
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError(f"Artifact not found: {uri}") from exc
        return response["Body"].read()

    def get_json(self, uri: str) -> Any:
        return unwrap_json_payload(json.loads(self.get_bytes(uri).decode("utf-8")))

    def _parse(self, uri: str) -> tuple[str, str]:
        if not uri.startswith(S3_SCHEME):
            raise ArtifactPathError(f"Unsupported S3 URI: {uri}")
        remainder = uri.removeprefix(S3_SCHEME)
        bucket, _, key = remainder.partition("/")
        if not bucket:
            raise ArtifactPathError(f"S3 URI is missing a bucket: {uri}")
        _validate_relative_key(key)
        return bucket, key


def get_artifact_store() -> ArtifactStore:
    settings = get_settings()
    if settings.artifact_backend == "s3":
        return S3ArtifactStore()
    return LocalArtifactStore()
