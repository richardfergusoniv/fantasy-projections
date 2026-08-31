"""Git provenance for sealed release bundles and promotion-time verification."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Literal

from src.projection.contracts import REPO_ROOT

ProvenanceMode = Literal["initial", "restore"]


class GitProvenanceError(RuntimeError):
    """Git working tree or commit state blocks promotion."""


def _run_git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise GitProvenanceError(
            f"git {' '.join(args)} failed: {(result.stderr or result.stdout).strip()}"
        )
    return (result.stdout or "").strip()


def current_head_commit(*, cwd: Path | None = None) -> str:
    return _run_git("rev-parse", "HEAD", cwd=cwd)


def working_tree_dirty(*, cwd: Path | None = None) -> bool:
    status = _run_git("status", "--porcelain", cwd=cwd)
    return bool(status)


def commit_is_ancestor(commit: str, *, cwd: Path | None = None) -> bool:
    """True when ``commit`` is an ancestor of HEAD (inclusive via merge-base)."""
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise GitProvenanceError(
        f"git merge-base --is-ancestor failed: {(result.stderr or result.stdout).strip()}"
    )


def capture_git_provenance(*, cwd: Path | None = None) -> dict[str, Any]:
    """Record the exact source commit at bundle construction time."""
    commit = current_head_commit(cwd=cwd)
    dirty = working_tree_dirty(cwd=cwd)
    if dirty:
        raise GitProvenanceError(
            "bundle construction requires a clean git checkout (git status --porcelain is non-empty)"
        )
    return {"source_commit": commit, "source_dirty": False}


def verify_promotion_git_state(
    manifest_git: dict[str, Any],
    *,
    cwd: Path | None = None,
    mode: ProvenanceMode = "initial",
) -> ProvenanceMode:
    """Verify clean-tree git provenance for promotion.

    ``initial`` requires ``HEAD == source_commit``.
    ``restore`` requires ``source_commit`` to be an ancestor of ``HEAD``.
    Both modes require ``source_dirty == false`` and a clean worktree.
    """
    if mode not in ("initial", "restore"):
        raise GitProvenanceError(f"unsupported provenance mode: {mode!r}")
    if manifest_git.get("source_dirty") is not False:
        raise GitProvenanceError("bundle records source_dirty != false")
    expected_commit = str(manifest_git.get("source_commit") or "").strip()
    if not expected_commit:
        raise GitProvenanceError("bundle missing git.source_commit")
    if working_tree_dirty(cwd=cwd):
        raise GitProvenanceError("promotion requires git status --porcelain to be empty")
    actual_commit = current_head_commit(cwd=cwd)
    if mode == "initial":
        if actual_commit != expected_commit:
            raise GitProvenanceError(
                f"HEAD {actual_commit} does not match bundle source_commit {expected_commit}"
            )
        return "initial"
    if not commit_is_ancestor(expected_commit, cwd=cwd):
        raise GitProvenanceError(
            f"bundle source_commit {expected_commit} is not an ancestor of HEAD {actual_commit}"
        )
    return "restore"
