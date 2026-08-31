"""Artifact store adapter tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest


def _missing_object_client(encoded: bytes) -> MagicMock:
    """An S3 client mock whose HEAD reports the object does not exist yet."""
    client = MagicMock()
    client.head_object.side_effect = RuntimeError("NoSuchKey")
    client.get_object.return_value = {"Body": MagicMock(read=lambda: encoded)}
    return client


def test_local_artifact_store_roundtrip(tmp_path):
    from src.app.artifacts.store import LocalArtifactStore

    store = LocalArtifactStore(root=tmp_path)
    uri = store.put_json({"hello": "world"})
    assert uri.startswith("local://")
    assert store.get_json(uri) == {"hello": "world"}


def test_s3_artifact_store_put_and_get():
    from src.app.artifacts.store import S3ArtifactStore, wrap_json_payload

    payload = {"players": 1}
    _, envelope = wrap_json_payload(payload)
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
    mock_client = _missing_object_client(encoded)

    with patch("boto3.session.Session") as session_cls:
        session_cls.return_value.client.return_value = mock_client
        store = S3ArtifactStore()
        uri = store.put_json(payload)
        assert uri.startswith("s3://")
        assert store.get_json(uri) == payload
        mock_client.put_object.assert_called_once()


@pytest.mark.parametrize(
    "hostile_uri",
    [
        "local://../../secrets.json",
        "local://C:/Windows/System32/config/SAM",
        "local:///etc/passwd",
        "local://..\\..\\secrets.json",
        "../../secrets.json",
        "local://",
    ],
)
def test_local_store_rejects_path_traversal(tmp_path, hostile_uri):
    """A URI must never resolve outside the configured artifact root."""
    from src.app.artifacts.store import ArtifactPathError, LocalArtifactStore

    outside = tmp_path.parent / "secrets.json"
    outside.write_text("top-secret", encoding="utf-8")
    store = LocalArtifactStore(root=tmp_path / "root")

    with pytest.raises(ArtifactPathError):
        store.get_bytes(hostile_uri)


def test_local_store_write_is_atomic(tmp_path):
    """A write that fails mid-stream must leave no readable final artifact."""
    from src.app.artifacts import store as store_module
    from src.app.artifacts.store import LocalArtifactStore

    store = LocalArtifactStore(root=tmp_path)
    payload = {"partial": "x" * 64}

    original_replace = store_module.os.replace

    def explode(src, dst):  # noqa: ANN001 - mirrors os.replace signature
        raise OSError("disk full")

    with patch.object(store_module.os, "replace", explode):
        with pytest.raises(OSError):
            store.put_json(payload)

    assert store_module.os.replace is original_replace
    json_files = list(tmp_path.rglob("*.json"))
    assert json_files == [], f"partial artifact left visible: {json_files}"
    # Temp files are cleaned up too, so no orphan is mistaken for an artifact.
    assert list(tmp_path.rglob("*.tmp")) == []


def test_local_and_s3_key_derivation_are_identical(tmp_path):
    """The two backends must address the same content at the same key."""
    from src.app.artifacts.store import (
        LocalArtifactStore,
        S3ArtifactStore,
        derive_artifact_key,
        wrap_json_payload,
    )

    payload = {"parity": [1, 2, 3]}
    digest, envelope = wrap_json_payload(payload)
    expected_key = derive_artifact_key(digest, "application/json")
    assert expected_key.endswith(".json")

    local_uri = LocalArtifactStore(root=tmp_path).put_json(payload)
    assert local_uri.endswith(expected_key)

    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
    mock_client = _missing_object_client(encoded)
    with patch("boto3.session.Session") as session_cls:
        session_cls.return_value.client.return_value = mock_client
        s3_uri = S3ArtifactStore().put_json(payload)
    assert s3_uri.endswith(expected_key)
    assert mock_client.put_object.call_args.kwargs["Key"] == expected_key


def test_both_backends_reject_the_same_bad_uris(tmp_path):
    from src.app.artifacts.store import ArtifactPathError, LocalArtifactStore, S3ArtifactStore

    local = LocalArtifactStore(root=tmp_path)
    with patch("boto3.session.Session"):
        s3 = S3ArtifactStore()

    with pytest.raises(ArtifactPathError):
        local.get_bytes("s3://bucket/artifacts/ab/cd/abcd.json")
    with pytest.raises(ArtifactPathError):
        s3.get_bytes("local://artifacts/ab/cd/abcd.json")
    with pytest.raises(ArtifactPathError):
        s3.get_bytes("s3://bucket/../../etc/passwd")


def test_local_put_json_is_idempotent(tmp_path):
    from src.app.artifacts.store import LocalArtifactStore

    store = LocalArtifactStore(root=tmp_path)
    payload = {"idempotent": True}
    first = store.put_json(payload)
    manifest = store.get_manifest(first)
    second = store.put_json(payload)

    assert first == second
    assert store.get_manifest(second) == manifest, "repeat put rewrote the manifest"


def test_json_manifest_records_identity_inputs_and_utc_timestamp(tmp_path):
    from src.app.artifacts.store import (
        ARTIFACT_ENVELOPE_SCHEMA,
        LocalArtifactStore,
        canonical_json_bytes,
    )
    import hashlib

    store = LocalArtifactStore(root=tmp_path)
    payload = {"season": 2026, "week": 1}
    uri = store.put_json(
        payload,
        inputs={"source": "sleeper", "endpoint": "league/1"},
        provenance={"job": "daily-refresh"},
    )

    manifest = store.get_manifest(uri)
    assert manifest is not None
    assert manifest["schema"] == ARTIFACT_ENVELOPE_SCHEMA
    assert manifest["schema_version"] == 1
    assert manifest["inputs"] == {"source": "sleeper", "endpoint": "league/1"}
    assert manifest["provenance"] == {"job": "daily-refresh"}
    assert manifest["content_sha256"] == hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    created_at = datetime.fromisoformat(manifest["created_at"])
    assert created_at.tzinfo is not None, "created_at must be timezone-aware"
    assert created_at.utcoffset() == UTC.utcoffset(None)

    # Backward compatibility: readers still see the bare payload.
    assert store.get_json(uri) == payload


def test_legacy_unwrapped_artifacts_still_read(tmp_path):
    """Artifacts written before envelopes existed must remain readable."""
    from src.app.artifacts.store import LocalArtifactStore

    store = LocalArtifactStore(root=tmp_path)
    legacy = tmp_path / "artifacts" / "ab" / "cd" / "legacy.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps({"legacy": True}), encoding="utf-8")

    assert store.get_json(f"local://{legacy.as_posix()}") == {"legacy": True}
