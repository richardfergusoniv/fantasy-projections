"""Fail-closed consensus snapshot pinning for shadow RB/WR attribution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.projection.contracts import REPO_ROOT
from src.projection.evaluation.accuracy_first import (
    TOP_ADP,
    load_consensus_snapshot,
    sha256_file,
)
from src.projection.shadow.forbidden import assert_input_path_allowed

CONSENSUS_DIR = Path(REPO_ROOT) / "data" / "consensus"
DEFAULT_FREEZE_MANIFEST = (
    Path(REPO_ROOT) / "output" / "accuracy_first_2026" / "freeze_manifest.json"
)
CONSENSUS_SEASONS = (2023, 2024, 2025)


class ConsensusPinError(RuntimeError):
    """Missing file, missing frozen hash, or byte-level SHA-256 mismatch."""


def expected_consensus_hashes(
    freeze_manifest_path: str | Path = DEFAULT_FREEZE_MANIFEST,
) -> dict[int, str]:
    """Read expected 2023–2025 consensus hashes from the frozen evidence manifest."""
    path = Path(freeze_manifest_path)
    assert_input_path_allowed(path)
    if not path.is_file():
        raise ConsensusPinError(f"Frozen evidence manifest missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files") or {}
    expected: dict[int, str] = {}
    for season in CONSENSUS_SEASONS:
        key = f"source:data/consensus/consensus_{season}.json"
        digest = files.get(key)
        if not digest:
            raise ConsensusPinError(
                f"Missing expected consensus hash in freeze manifest: {key}"
            )
        expected[int(season)] = str(digest)
    return expected


def membership_set_hash(player_ids: list[str] | pd.Series) -> str:
    """Stable SHA-256 over the sorted unique player-ID membership set."""
    ids = sorted({str(pid) for pid in player_ids})
    payload = "\n".join(ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_pinned_consensus(
    season: int,
    *,
    expected_hash: str,
    consensus_dir: str | Path = CONSENSUS_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load consensus only when the snapshot bytes match the frozen hash."""
    path = Path(consensus_dir) / f"consensus_{int(season)}.json"
    assert_input_path_allowed(path)
    if not path.is_file():
        raise ConsensusPinError(f"Consensus snapshot missing: {path}")
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ConsensusPinError(
            f"Consensus SHA-256 mismatch for {season}: "
            f"expected={expected_hash} actual={actual}"
        )
    rows, meta = load_consensus_snapshot(path, expected_season=int(season))
    top120 = rows[rows["adp"].notna() & rows["adp"].le(TOP_ADP)].copy()
    member_ids = top120["player_id"].astype(str).tolist()
    record = {
        "season": int(season),
        "path": str(path).replace("\\", "/"),
        "expected_hash": expected_hash,
        "actual_hash": actual,
        "as_of": meta.get("as_of"),
        "row_count": int(len(rows)),
        "top120_count": int(len(top120)),
        "top120_membership_hash": membership_set_hash(member_ids),
        "hash_match": True,
    }
    return rows, record


def persist_top120_membership(
    season: int,
    consensus: pd.DataFrame,
    *,
    out_path: str | Path,
    pin_record: dict[str, Any],
) -> dict[str, Any]:
    """Write the exact top-120 membership set used by a fold."""
    top120 = consensus[
        consensus["adp"].notna() & pd.to_numeric(consensus["adp"], errors="coerce").le(TOP_ADP)
    ].copy()
    player_ids = sorted(top120["player_id"].astype(str).unique().tolist())
    payload = {
        "schema_version": "shadow_v1_rb_wr_top120_membership_v1",
        "season": int(season),
        "top120_membership_hash": membership_set_hash(player_ids),
        "player_ids": player_ids,
        "consensus_pin": pin_record,
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def require_all_pinned_consensus(
    seasons: tuple[int, ...] = CONSENSUS_SEASONS,
    *,
    freeze_manifest_path: str | Path = DEFAULT_FREEZE_MANIFEST,
    consensus_dir: str | Path = CONSENSUS_DIR,
) -> dict[int, dict[str, Any]]:
    """Fail closed before any fold metrics: every season must pin-match."""
    expected = expected_consensus_hashes(freeze_manifest_path)
    pins: dict[int, dict[str, Any]] = {}
    for season in seasons:
        if season not in expected:
            raise ConsensusPinError(f"No frozen hash for consensus season {season}")
        _, record = load_pinned_consensus(
            season,
            expected_hash=expected[season],
            consensus_dir=consensus_dir,
        )
        pins[int(season)] = record
    return pins
