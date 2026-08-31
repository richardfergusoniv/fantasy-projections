"""Pointer-driven browser surface verification behavior."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from src.projection.active_release import build_active_pointer, write_active_pointer


def _patch_pointer_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.projection.active_release.REPO_ROOT", str(tmp_path))


def _manifest_and_pointer(*, namespace: str = "live_ns") -> tuple[dict, dict, bytes]:
    asset_bytes = b"players-bytes"
    manifest = {
        "artifacts": [
            {
                "role": "players",
                "path": "players_2026.json",
                "sha256": hashlib.sha256(asset_bytes).hexdigest(),
                "browser_consumed": True,
            }
        ]
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    pointer = build_active_pointer(
        season=2026,
        namespace=namespace,
        release_id="rel-live",
        manifest_sha256=digest,
    )
    return pointer, manifest, asset_bytes


def test_verify_uses_served_pointer_when_namespace_omitted(tmp_path, monkeypatch):
    from scripts import verify_browser_surfaces as verify_mod

    _patch_pointer_root(tmp_path, monkeypatch)
    pointer, manifest, asset_bytes = _manifest_and_pointer()
    write_active_pointer(pointer)
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()

    def fake_fetch(url: str):
        if url.endswith("active_release_2026.json"):
            return 200, json.dumps(pointer).encode()
        if url.endswith("release_bundle_manifest.json"):
            return 200, manifest_bytes
        if url.endswith("index.html") or any(
            part in url for part in ("/teams/", "/totals/", "/compare/", "/sleepers/")
        ):
            return 200, b"<html></html>"
        if f"/data/releases/{pointer['namespace']}/" in url:
            if url.endswith("players_2026.json"):
                return 200, asset_bytes
            return 200, b"asset"
        return 404, b""

    with patch.object(verify_mod, "fetch", side_effect=fake_fetch):
        report = verify_mod.verify(
            "http://127.0.0.1:8766",
            2026,
            expect_namespace=None,
            expect_legacy=False,
        )

    assert report["frozen_identity"]["namespace"] == "live_ns"
    assert report["frozen_identity"]["manifest_sha256"] == pointer["manifest_sha256"]
    assert report["verdict"] == "pass"
    assert any(c["check"] == "served_pointer_matches_checked_in" and c["passed"] for c in report["checks"])
    assert all(
        c.get("frozen_namespace") == "live_ns"
        for c in report["checks"]
        if c["check"].startswith("asset:")
    )


def test_verify_explicit_namespace_mismatch_fails(tmp_path, monkeypatch):
    from scripts import verify_browser_surfaces as verify_mod

    _patch_pointer_root(tmp_path, monkeypatch)
    pointer = build_active_pointer(
        season=2026,
        namespace="live_ns",
        release_id="rel-live",
        manifest_sha256="a" * 64,
    )
    write_active_pointer(pointer)

    def fake_fetch(url: str):
        if url.endswith("active_release_2026.json"):
            return 200, json.dumps(pointer).encode()
        return 200, b"ok"

    with patch.object(verify_mod, "fetch", side_effect=fake_fetch):
        report = verify_mod.verify(
            "http://127.0.0.1:8766",
            2026,
            expect_namespace="other_ns",
            expect_legacy=False,
        )
    assert report["verdict"] == "fail"
    assert any(
        c["check"] == "active_pointer_namespace" and c["passed"] is False for c in report["checks"]
    )


def test_verify_served_pointer_drift_fails(tmp_path, monkeypatch):
    from scripts import verify_browser_surfaces as verify_mod

    _patch_pointer_root(tmp_path, monkeypatch)
    pointer = build_active_pointer(
        season=2026,
        namespace="live_ns",
        release_id="rel-live",
        manifest_sha256="a" * 64,
    )
    write_active_pointer(pointer)
    drifted = dict(pointer)
    drifted["namespace"] = "drifted_ns"
    drifted["release_id"] = "rel-drift"
    drifted["manifest_sha256"] = "c" * 64

    def fake_fetch(url: str):
        if url.endswith("active_release_2026.json"):
            return 200, json.dumps(drifted).encode()
        return 200, b"ok"

    with patch.object(verify_mod, "fetch", side_effect=fake_fetch):
        report = verify_mod.verify(
            "http://127.0.0.1:8766",
            2026,
            expect_namespace=None,
            expect_legacy=False,
        )
    assert report["verdict"] == "fail"
    assert any(
        c["check"] == "served_pointer_matches_checked_in" and c["passed"] is False
        for c in report["checks"]
    )
    assert report["frozen_identity"]["namespace"] == "drifted_ns"
