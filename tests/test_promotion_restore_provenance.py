"""Restore provenance, promotion receipts, and pointer recovery tests."""
from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src.projection.active_release import (
    build_active_pointer,
    pointer_path,
    read_active_pointer,
    validate_active_pointer,
    write_active_pointer,
)
from src.projection.evaluation.promotion_invariants import validate_sealed_promotion_invariants
from src.projection.evaluation.release_bundle_validation import validate_release_bundle
from src.projection.git_provenance import (
    GitProvenanceError,
    commit_is_ancestor,
    verify_promotion_git_state,
)
from src.projection.promote_release import PromoteReleaseError, promote_release, rollback_release
from src.projection.promotion_receipt import (
    PromotionReceiptError,
    build_promotion_receipt,
    derive_provenance_mode,
    receipt_path,
    tracked_receipt_authorizes_restore,
    write_promotion_receipt,
)
from src.projection.release_bundle import bundle_root, canonical_dumps, load_sealed_manifest
from tests.fixtures.release_bundle_v2 import seal_v2_bundle


SOURCE_COMMIT = "abc123def4567890abcdef1234567890abcdef12"
ANCESTOR_COMMIT = "1111111111111111111111111111111111111111"
HEAD_COMMIT = "2222222222222222222222222222222222222222"


def _patch_roots(tmp_path: Path, monkeypatch, *, source_commit: str = SOURCE_COMMIT) -> None:
    model_v3 = tmp_path / "model_v3"
    model_v3.mkdir()
    monkeypatch.setattr("src.projection.release_bundle.MODEL_V3_DIR", str(model_v3))
    monkeypatch.setattr("src.projection.release_bundle.REPO_ROOT", str(tmp_path))
    monkeypatch.setattr("src.projection.active_release.REPO_ROOT", str(tmp_path))
    monkeypatch.setattr("src.projection.promotion_receipt.REPO_ROOT", str(tmp_path))
    monkeypatch.setattr("src.projection.git_provenance.REPO_ROOT", str(tmp_path))
    monkeypatch.setattr("src.projection.git_provenance.working_tree_dirty", lambda **_: False)
    monkeypatch.setattr("src.projection.git_provenance.current_head_commit", lambda **_: source_commit)
    monkeypatch.setattr("src.projection.git_provenance.commit_is_ancestor", lambda commit, **_: True)


def test_verify_initial_requires_exact_head_match():
    git = {"source_commit": SOURCE_COMMIT, "source_dirty": False}
    with (
        patch("src.projection.git_provenance.working_tree_dirty", return_value=False),
        patch("src.projection.git_provenance.current_head_commit", return_value=SOURCE_COMMIT),
    ):
        assert verify_promotion_git_state(git, mode="initial") == "initial"
    with (
        patch("src.projection.git_provenance.working_tree_dirty", return_value=False),
        patch("src.projection.git_provenance.current_head_commit", return_value=HEAD_COMMIT),
    ):
        with pytest.raises(GitProvenanceError, match="does not match"):
            verify_promotion_git_state(git, mode="initial")


def test_verify_restore_accepts_ancestor_and_rejects_divergent():
    git = {"source_commit": ANCESTOR_COMMIT, "source_dirty": False}
    with (
        patch("src.projection.git_provenance.working_tree_dirty", return_value=False),
        patch("src.projection.git_provenance.current_head_commit", return_value=HEAD_COMMIT),
        patch("src.projection.git_provenance.commit_is_ancestor", return_value=True),
    ):
        assert verify_promotion_git_state(git, mode="restore") == "restore"
    with (
        patch("src.projection.git_provenance.working_tree_dirty", return_value=False),
        patch("src.projection.git_provenance.current_head_commit", return_value=HEAD_COMMIT),
        patch("src.projection.git_provenance.commit_is_ancestor", return_value=False),
    ):
        with pytest.raises(GitProvenanceError, match="not an ancestor"):
            verify_promotion_git_state(git, mode="restore")


def test_verify_rejects_dirty_tree_and_source_dirty():
    with (
        patch("src.projection.git_provenance.working_tree_dirty", return_value=True),
        patch("src.projection.git_provenance.current_head_commit", return_value=SOURCE_COMMIT),
    ):
        with pytest.raises(GitProvenanceError, match="porcelain"):
            verify_promotion_git_state(
                {"source_commit": SOURCE_COMMIT, "source_dirty": False},
                mode="restore",
            )
    with (
        patch("src.projection.git_provenance.working_tree_dirty", return_value=False),
        patch("src.projection.git_provenance.current_head_commit", return_value=SOURCE_COMMIT),
    ):
        with pytest.raises(GitProvenanceError, match="source_dirty"):
            verify_promotion_git_state(
                {"source_commit": SOURCE_COMMIT, "source_dirty": True},
                mode="initial",
            )


def test_fresh_namespace_with_only_validation_sidecar_stays_initial(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch, source_commit=SOURCE_COMMIT)
    manifest, digest = seal_v2_bundle(tmp_path, "fresh_ns", source_commit=SOURCE_COMMIT)
    report = validate_release_bundle(season=2026, namespace="fresh_ns", require_active=False)
    assert report["verdict"] == "pass"
    assert (bundle_root(2026, "fresh_ns") / "release_bundle_validation.json").exists()

    mode = derive_provenance_mode(
        season=2026,
        namespace="fresh_ns",
        release_id=str(manifest["bundle"]["release_id"]),
        manifest_sha256=digest,
    )
    assert mode == "initial"

    monkeypatch.setattr(
        "src.projection.git_provenance.current_head_commit",
        lambda **_: HEAD_COMMIT,
    )
    sealed = validate_sealed_promotion_invariants(
        season=2026,
        namespace="fresh_ns",
        provenance_mode=mode,
    )
    assert sealed["verdict"] == "fail"
    assert sealed["provenance_mode"] == "initial"
    git_check = next(c for c in sealed["checks"] if c["check"] == "git_provenance")
    assert git_check["passed"] is False


def test_restore_derivation_from_current_previous_and_receipt(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch)
    manifest, digest = seal_v2_bundle(tmp_path, "restore_ns", release_id="rel-restore")
    release_id = str(manifest["bundle"]["release_id"])

    write_active_pointer(
        build_active_pointer(
            season=2026,
            namespace="restore_ns",
            release_id=release_id,
            manifest_sha256=digest,
        )
    )
    assert (
        derive_provenance_mode(
            season=2026,
            namespace="restore_ns",
            release_id=release_id,
            manifest_sha256=digest,
        )
        == "restore"
    )

    write_active_pointer(
        build_active_pointer(
            season=2026,
            namespace="other_ns",
            release_id="other-rel",
            manifest_sha256="f" * 64,
            previous={
                "namespace": "restore_ns",
                "release_id": release_id,
                "manifest_sha256": digest,
            },
        )
    )
    assert (
        derive_provenance_mode(
            season=2026,
            namespace="restore_ns",
            release_id=release_id,
            manifest_sha256=digest,
        )
        == "restore"
    )

    pointer_path(2026).unlink()
    receipt = build_promotion_receipt(
        season=2026,
        namespace="restore_ns",
        release_id=release_id,
        manifest_sha256=digest,
        source_commit=SOURCE_COMMIT,
        provenance_mode="initial",
    )
    dest = write_promotion_receipt(receipt)
    monkeypatch.setattr(
        "src.projection.promotion_receipt.receipt_is_git_tracked",
        lambda path, **_: path == dest,
    )
    assert (
        derive_provenance_mode(
            season=2026,
            namespace="restore_ns",
            release_id=release_id,
            manifest_sha256=digest,
        )
        == "restore"
    )


def test_restore_rejects_missing_untracked_malformed_and_hash_conflict(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch)
    manifest, digest = seal_v2_bundle(tmp_path, "reject_ns", release_id="rel-reject")
    release_id = str(manifest["bundle"]["release_id"])

    assert (
        tracked_receipt_authorizes_restore(
            season=2026,
            namespace="reject_ns",
            release_id=release_id,
            manifest_sha256=digest,
        )
        is False
    )

    receipt = build_promotion_receipt(
        season=2026,
        namespace="reject_ns",
        release_id=release_id,
        manifest_sha256=digest,
        source_commit=SOURCE_COMMIT,
        provenance_mode="initial",
    )
    dest = write_promotion_receipt(receipt)
    monkeypatch.setattr(
        "src.projection.promotion_receipt.receipt_is_git_tracked",
        lambda path, **_: False,
    )
    assert (
        derive_provenance_mode(
            season=2026,
            namespace="reject_ns",
            release_id=release_id,
            manifest_sha256=digest,
        )
        == "initial"
    )

    dest.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(
        "src.projection.promotion_receipt.receipt_is_git_tracked",
        lambda path, **_: True,
    )
    assert (
        derive_provenance_mode(
            season=2026,
            namespace="reject_ns",
            release_id=release_id,
            manifest_sha256=digest,
        )
        == "initial"
    )

    dest.unlink()
    write_promotion_receipt(receipt)
    assert (
        derive_provenance_mode(
            season=2026,
            namespace="reject_ns",
            release_id=release_id,
            manifest_sha256="0" * 64,
        )
        == "initial"
    )


def test_malformed_pointer_falls_back_to_receipt_then_initial(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch)
    manifest, digest = seal_v2_bundle(tmp_path, "fallback_ns", release_id="rel-fallback")
    release_id = str(manifest["bundle"]["release_id"])

    pointer_path(2026).parent.mkdir(parents=True, exist_ok=True)
    pointer_path(2026).write_text("{bad", encoding="utf-8")
    assert (
        derive_provenance_mode(
            season=2026,
            namespace="fallback_ns",
            release_id=release_id,
            manifest_sha256=digest,
        )
        == "initial"
    )

    receipt = build_promotion_receipt(
        season=2026,
        namespace="fallback_ns",
        release_id=release_id,
        manifest_sha256=digest,
        source_commit=SOURCE_COMMIT,
        provenance_mode="initial",
    )
    dest = write_promotion_receipt(receipt)
    monkeypatch.setattr(
        "src.projection.promotion_receipt.receipt_is_git_tracked",
        lambda path, **_: path == dest,
    )
    assert (
        derive_provenance_mode(
            season=2026,
            namespace="fallback_ns",
            release_id=release_id,
            manifest_sha256=digest,
        )
        == "restore"
    )


def test_pointer_previous_manifest_sha256_optional_compatibility(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch)
    without_hash = validate_active_pointer(
        {
            "schema_version": "active_release_pointer_v1",
            "season": 2026,
            "status": "active",
            "namespace": "cur_ns",
            "release_id": "cur-rel",
            "manifest_path": "data/releases/cur_ns/release_bundle_manifest.json",
            "manifest_sha256": "a" * 64,
            "activated_at": "2026-08-30T00:00:00+00:00",
            "previous": {"namespace": "prev_ns", "release_id": "prev-rel"},
            "public_base": "data/releases/cur_ns",
            "public_urls": {},
        },
        season=2026,
    )
    assert without_hash["previous"] == {"namespace": "prev_ns", "release_id": "prev-rel"}
    assert "manifest_sha256" not in without_hash["previous"]

    with_hash = validate_active_pointer(
        {
            "schema_version": "active_release_pointer_v1",
            "season": 2026,
            "status": "active",
            "namespace": "cur_ns",
            "release_id": "cur-rel",
            "manifest_path": "data/releases/cur_ns/release_bundle_manifest.json",
            "manifest_sha256": "a" * 64,
            "activated_at": "2026-08-30T00:00:00+00:00",
            "previous": {
                "namespace": "prev_ns",
                "release_id": "prev-rel",
                "manifest_sha256": "b" * 64,
            },
            "public_base": "data/releases/cur_ns",
            "public_urls": {},
        },
        season=2026,
    )
    assert with_hash["previous"]["manifest_sha256"] == "b" * 64

    write_active_pointer(without_hash)
    assert (
        derive_provenance_mode(
            season=2026,
            namespace="prev_ns",
            release_id="prev-rel",
            manifest_sha256="b" * 64,
        )
        == "restore"
    )


def test_receipt_idempotent_and_conflicts_on_reuse(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch)
    first = build_promotion_receipt(
        season=2026,
        namespace="idem_ns",
        release_id="rel-1",
        manifest_sha256="a" * 64,
        source_commit=SOURCE_COMMIT,
        provenance_mode="initial",
    )
    write_promotion_receipt(first)
    write_promotion_receipt(first)  # idempotent
    conflict = build_promotion_receipt(
        season=2026,
        namespace="idem_ns",
        release_id="rel-2",
        manifest_sha256="c" * 64,
        source_commit=SOURCE_COMMIT,
        provenance_mode="restore",
    )
    with pytest.raises(PromotionReceiptError, match="conflicting"):
        write_promotion_receipt(conflict)


def test_promote_writes_receipt_and_records_mode(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch, source_commit=SOURCE_COMMIT)
    seal_v2_bundle(tmp_path, "promo_receipt_ns", release_id="rel-promo")
    monkeypatch.setattr(
        "src.projection.promotion_receipt.receipt_is_git_tracked",
        lambda path, **_: False,
    )
    result = promote_release(2026, "promo_receipt_ns")
    assert result["provenance_mode"] == "initial"
    assert result["promotion_invariants"]["provenance_mode"] == "initial"
    receipt_file = receipt_path(2026, "promo_receipt_ns")
    assert receipt_file.exists()
    payload = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "release_promotion_receipt_v1"
    assert payload["provenance_mode"] == "initial"
    assert payload["promotion_invariants_verdict"] == "pass"


def test_rollback_and_rollforward_preserve_sealed_bytes(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch, source_commit=SOURCE_COMMIT)
    base_manifest, base_digest = seal_v2_bundle(
        tmp_path, "base_ns", release_id="rel-base", source_commit=SOURCE_COMMIT
    )
    cand_manifest, cand_digest = seal_v2_bundle(
        tmp_path, "cand_ns", release_id="rel-cand", source_commit=SOURCE_COMMIT
    )
    monkeypatch.setattr(
        "src.projection.promotion_receipt.receipt_is_git_tracked",
        lambda path, **_: True,
    )
    promote_release(2026, "base_ns")
    promote_release(2026, "cand_ns")
    base_before = (bundle_root(2026, "base_ns") / "release_bundle_manifest.json").read_bytes()
    cand_before = (bundle_root(2026, "cand_ns") / "release_bundle_manifest.json").read_bytes()

    rollback_release(2026)
    assert read_active_pointer(2026)["namespace"] == "base_ns"
    assert (bundle_root(2026, "base_ns") / "release_bundle_manifest.json").read_bytes() == base_before
    assert (bundle_root(2026, "cand_ns") / "release_bundle_manifest.json").read_bytes() == cand_before

    promote_release(2026, "cand_ns")
    assert read_active_pointer(2026)["namespace"] == "cand_ns"
    assert read_active_pointer(2026)["previous"]["manifest_sha256"] == base_digest
    assert (bundle_root(2026, "base_ns") / "release_bundle_manifest.json").read_bytes() == base_before
    assert (bundle_root(2026, "cand_ns") / "release_bundle_manifest.json").read_bytes() == cand_before


def test_failed_promotion_preserves_pointer_and_public_namespace(tmp_path, monkeypatch):
    _patch_roots(tmp_path, monkeypatch, source_commit=SOURCE_COMMIT)
    seal_v2_bundle(tmp_path, "live_ns", release_id="rel-live")
    promote_release(2026, "live_ns")
    before_pointer = pointer_path(2026).read_bytes()
    public = Path(tmp_path) / "draft_assistant" / "data" / "releases" / "live_ns"
    marker = public / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    seal_v2_bundle(tmp_path, "bad_ns", release_id="rel-bad")
    root = bundle_root(2026, "bad_ns")
    players = json.loads((root / "players_2026.json").read_text(encoding="utf-8"))
    players["meta"]["selected_board_sha256"] = "0" * 64
    (root / "players_2026.json").write_text(json.dumps(players), encoding="utf-8")

    with pytest.raises(PromoteReleaseError):
        promote_release(2026, "bad_ns")
    assert pointer_path(2026).read_bytes() == before_pointer
    assert marker.read_text(encoding="utf-8") == "keep"
    assert read_active_pointer(2026)["namespace"] == "live_ns"


def test_public_promotion_apis_have_no_bypass_or_mode_params():
    from src.projection import promote_release as promote_module

    for name in ("promote_release", "rollback_release"):
        params = inspect.signature(getattr(promote_module, name)).parameters
        for forbidden in ("skip_git", "mode", "allow", "provenance", "provenance_mode", "force"):
            assert forbidden not in params


def test_commit_is_ancestor_uses_merge_base(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "a.txt").write_text("a", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "a"], cwd=repo, check=True, capture_output=True)
    first = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    (repo / "b.txt").write_text("b", encoding="utf-8")
    subprocess.run(["git", "add", "b.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "b"], cwd=repo, check=True, capture_output=True)
    assert commit_is_ancestor(first, cwd=repo) is True
    subprocess.run(["git", "checkout", "--orphan", "other"], cwd=repo, check=True, capture_output=True)
    (repo / "c.txt").write_text("c", encoding="utf-8")
    subprocess.run(["git", "add", "c.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "c"], cwd=repo, check=True, capture_output=True)
    # HEAD is now on an orphan branch; the first commit is not an ancestor.
    assert commit_is_ancestor(first, cwd=repo) is False
    subprocess.run(["git", "checkout", "master"], cwd=repo, check=False, capture_output=True)
    subprocess.run(["git", "checkout", "main"], cwd=repo, check=False, capture_output=True)


def test_check_promotion_provenance_derives_restore_and_warns_on_naive_initial(
    tmp_path, monkeypatch
):
    from scripts.check_promotion_provenance import check_promotion_provenance

    _patch_roots(tmp_path, monkeypatch, source_commit=SOURCE_COMMIT)
    manifest, digest = seal_v2_bundle(tmp_path, "check_ns", release_id="rel-check")
    release_id = str(manifest["bundle"]["release_id"])
    write_active_pointer(
        build_active_pointer(
            season=2026,
            namespace="check_ns",
            release_id=release_id,
            manifest_sha256=digest,
        )
    )
    monkeypatch.setattr(
        "src.projection.git_provenance.current_head_commit",
        lambda **_: HEAD_COMMIT,
    )

    report = check_promotion_provenance(season=2026, namespace="check_ns")
    assert report["provenance_mode"] == "restore"
    assert report["promotable"] is True
    assert report["verdict"] == "pass"
    assert report["naive_initial_warning"] is not None
    assert "naive initial-mode" in report["naive_initial_warning"]
