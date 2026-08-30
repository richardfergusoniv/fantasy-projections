"""Promotion, tamper detection, pointer isolation, and browser namespace resolution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.projection.active_release import (
    ActiveReleaseError,
    build_active_pointer,
    frozen_load_session,
    pointer_path,
    read_active_pointer,
    resolve_browser_urls,
    write_active_pointer,
)
from src.projection.evaluation.release_bundle_validation import validate_release_bundle
from src.projection.promote_release import PromoteReleaseError, promote_release, rollback_release
from src.projection.release_bundle import (
    MANIFEST_FILENAME,
    SCHEMA_VERSION_V2,
    player_id_set_hash,
    selected_points_vector_hash,
    treatment_block,
)
from src.projection.release_bundle_publish import seal_staged_bundle


def _patch_roots(tmp_path: Path, monkeypatch) -> None:
    model_v3 = tmp_path / "model_v3"
    model_v3.mkdir()
    monkeypatch.setattr("src.projection.release_bundle.MODEL_V3_DIR", str(model_v3))
    monkeypatch.setattr("src.projection.release_bundle.REPO_ROOT", str(tmp_path))
    monkeypatch.setattr("src.projection.active_release.REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "src.projection.release_candidate.MODEL_V3_DIR",
        str(model_v3),
    )
    monkeypatch.setattr(
        "src.projection.release_candidate.OUTPUT_DIR",
        str(tmp_path / "output"),
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _stage_bundle(tmp_path: Path, namespace: str, *, season: int = 2026) -> Path:
    from src.projection.release_bundle import bundle_root

    root = bundle_root(season, namespace)
    root.mkdir(parents=True, exist_ok=True)
    selected = "player_id,fantasy_pts_season\na,100\n"
    _write(root / "selected_board.csv", selected)
    _write(root / "players_2026.json", json.dumps({"meta": {"model_id": "accuracy_first_ensemble"}, "players": [{"player_id": "a"}]}))
    _write(root / "team_stats_2026.json", json.dumps({"players": []}))
    _write(root / "comparison_2026.json", json.dumps({"players": []}))
    _write(root / "release_report_2026.json", "{}")
    _write(root / "release_report_simulation_2026.json", "{}")
    _write(root / "release_report_board_2026.json", "{}")
    _write(root / "application_contract.json", json.dumps({"contract_hash": "a" * 64}))
    _write(
        root / "simulation_manifest_2026.json",
        json.dumps({"draw_count": 10000, "simulation_run_id": "sim-1", "canonical_projection_run_id": "proj-1"}),
    )
    return root


from tests.fixtures.release_bundle_v2 import seal_v2_bundle


def _seal(tmp_path: Path, namespace: str, *, release_id: str = "rel-1") -> tuple[dict, str]:
    return seal_v2_bundle(tmp_path, namespace, release_id=release_id)


def _patch_git(monkeypatch, tmp_path: Path, source_commit: str = "abc123def4567890abcdef1234567890abcdef12") -> None:
    monkeypatch.setattr("src.projection.git_provenance.REPO_ROOT", str(tmp_path))
    monkeypatch.setattr("src.projection.git_provenance.working_tree_dirty", lambda **_: False)
    monkeypatch.setattr("src.projection.git_provenance.current_head_commit", lambda **_: source_commit)


def test_candidate_seal_leaves_active_pointer_unchanged(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch)
    _patch_git(monkeypatch, tmp_path)
    pointer = build_active_pointer(
        season=2026,
        namespace="already_live",
        release_id="live-1",
        manifest_sha256="f" * 64,
    )
    write_active_pointer(pointer)
    before = pointer_path(2026).read_bytes()
    _seal(tmp_path, "candidate_ns")
    after = pointer_path(2026).read_bytes()
    assert before == after
    live = read_active_pointer(2026)
    assert live["namespace"] == "already_live"


def test_tampering_required_artifact_or_run_id_blocks_promotion(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch)
    _patch_git(monkeypatch, tmp_path)
    manifest, digest = _seal(tmp_path, "promo_ns")
    from src.projection.release_bundle import bundle_root

    root = bundle_root(2026, "promo_ns")
    (root / "selected_board.csv").write_text("tampered", encoding="utf-8")
    report = validate_release_bundle(season=2026, namespace="promo_ns")
    assert report["verdict"] == "fail"
    assert any(check["check"] == "artifact_hashes" and not check["passed"] for check in report["checks"])
    with pytest.raises(PromoteReleaseError):
        promote_release(2026, "promo_ns")

    _seal(tmp_path, "promo_ns2")
    root2 = bundle_root(2026, "promo_ns2")
    sim_path = root2 / "simulation_manifest_2026.json"
    sim_path.write_text(json.dumps({"draw_count": 10000, "simulation_run_id": "tampered-run"}), encoding="utf-8")
    report2 = validate_release_bundle(
        season=2026,
        namespace="promo_ns2",
        expected={"simulation_run_id": "sim-1"},
    )
    assert report2["verdict"] == "fail"


def test_promotion_and_rollback_change_only_the_pointer(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch)
    _patch_git(monkeypatch, tmp_path)
    manifest_a, digest_a = _seal(tmp_path, "ns_a", release_id="rel-a")
    manifest_b, digest_b = _seal(tmp_path, "ns_b", release_id="rel-b")
    from src.projection.release_bundle import bundle_root, sha256_bytes

    sealed_a = (bundle_root(2026, "ns_a") / MANIFEST_FILENAME).read_bytes()
    sealed_b = (bundle_root(2026, "ns_b") / MANIFEST_FILENAME).read_bytes()
    promote_release(2026, "ns_a")
    first = read_active_pointer(2026)
    assert first["namespace"] == "ns_a"
    assert first["manifest_sha256"] == digest_a
    assert first["status"] == "active"
    promote_release(2026, "ns_b")
    second = read_active_pointer(2026)
    assert second["namespace"] == "ns_b"
    assert second["previous"]["namespace"] == "ns_a"
    rollback_release(2026)
    rolled = read_active_pointer(2026)
    assert rolled["namespace"] == "ns_a"
    assert (bundle_root(2026, "ns_a") / MANIFEST_FILENAME).read_bytes() == sealed_a
    assert (bundle_root(2026, "ns_b") / MANIFEST_FILENAME).read_bytes() == sealed_b
    assert hashlib.sha256(sealed_a).hexdigest() == digest_a
    assert hashlib.sha256(sealed_b).hexdigest() == digest_b


def test_malformed_pointer_does_not_bootstrap_legacy(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch)
    path = pointer_path(2026)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ActiveReleaseError, match="not valid JSON"):
        read_active_pointer(2026)


def test_missing_pointer_is_none_for_legacy_bootstrap(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch)
    assert read_active_pointer(2026) is None


def test_browser_views_share_one_frozen_namespace(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch)
    _patch_git(monkeypatch, tmp_path)
    manifest, digest = _seal(tmp_path, "shared_ns")
    pointer = build_active_pointer(
        season=2026,
        namespace="shared_ns",
        release_id=manifest["bundle"]["release_id"],
        manifest_sha256=digest,
    )
    session = frozen_load_session(pointer)
    later_pointer = build_active_pointer(
        season=2026,
        namespace="other_ns",
        release_id="other",
        manifest_sha256="0" * 64,
    )
    assert session["namespace"] == "shared_ns"
    assert later_pointer["namespace"] != session["namespace"]
    urls = resolve_browser_urls(pointer, manifest, data_root="data")
    nested = resolve_browser_urls(pointer, manifest, data_root="../data")
    for role in ("players", "team_stats", "comparison"):
        assert urls[role].startswith("data/releases/shared_ns/")
        assert nested[role].startswith("../data/releases/shared_ns/")
        assert "shared_ns" in urls[role]
        assert "other_ns" not in urls[role]


def test_browser_pages_load_through_release_loader():
    root = Path("draft_assistant")
    pages = {
        "index.html": "js/release_loader.js",
        "teams/index.html": "../js/release_loader.js",
        "totals/index.html": "../js/release_loader.js",
        "compare/index.html": "../js/release_loader.js",
        "sleepers/index.html": "../js/release_loader.js",
    }
    for page, src in pages.items():
        text = (root / page).read_text(encoding="utf-8")
        assert src in text
    apps = [
        root / "js" / "app.js",
        root / "teams" / "js" / "app.js",
        root / "totals" / "js" / "app.js",
        root / "compare" / "js" / "app.js",
        root / "sleepers" / "js" / "app.js",
    ]
    for app in apps:
        text = app.read_text(encoding="utf-8")
        assert "FantasyRelease.loadContext" in text
        assert "fetch(`data/players_" not in text
        assert "fetch(`../data/players_" not in text
        assert "fetch(`../data/comparison_" not in text
        assert "fetch(`../data/team_stats_" not in text
    loader = (root / "js" / "release_loader.js").read_text(encoding="utf-8")
    assert 'cache: "no-cache"' in loader
    assert "pointer_missing" in loader
    assert "Never fall back to legacy" in loader
    assert "frozen" in loader


def test_require_active_passes_only_after_promotion(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch)
    _patch_git(monkeypatch, tmp_path)
    _seal(tmp_path, "need_active")
    inactive = validate_release_bundle(season=2026, namespace="need_active", require_active=True)
    assert inactive["verdict"] == "fail"
    assert inactive["derived_status"] == "inactive"
    promote_release(2026, "need_active")
    active = validate_release_bundle(season=2026, namespace="need_active", require_active=True)
    assert active["verdict"] == "pass"
    assert active["derived_status"] == "active"
    from src.projection.release_bundle import bundle_root, load_sealed_manifest

    sealed, _ = load_sealed_manifest(bundle_root(2026, "need_active"))
    assert "status" not in sealed
    assert sealed["schema_version"] == SCHEMA_VERSION_V2
