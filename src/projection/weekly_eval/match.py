"""Join diagnostics for Role 2 board ↔ snapshot matching.

Does not fuzzy-match names. Live weeks must use the same ``player_id`` as the
M3 / sealed board. Fixture dry-run names (``p-qb``) will not join to gsis ids.
"""
from __future__ import annotations

import pandas as pd

from src.projection.weekly_eval.leakage import JOIN_KEYS


def _key_tuples(frame: pd.DataFrame) -> set[tuple[str, int, int, str]]:
    if frame is None or frame.empty:
        return set()
    missing = [k for k in JOIN_KEYS if k not in frame.columns]
    if missing:
        return set()
    return set(
        zip(
            frame["player_id"].astype(str),
            pd.to_numeric(frame["season"], errors="coerce").fillna(0).astype(int),
            pd.to_numeric(frame["week"], errors="coerce").fillna(0).astype(int),
            frame["market"].astype(str),
        )
    )


def describe_match(
    board: pd.DataFrame,
    snapshots: pd.DataFrame | None,
    joined: pd.DataFrame,
) -> dict[str, object]:
    """Report unmatched keys so a silent n_matched=0 is diagnosable."""
    board_keys = _key_tuples(board)
    snap_keys = _key_tuples(snapshots if snapshots is not None else pd.DataFrame())
    joined_keys = _key_tuples(joined)
    unmatched_board = board_keys - joined_keys
    unmatched_snap = snap_keys - joined_keys
    return {
        "join_keys": list(JOIN_KEYS),
        "n_board_unmatched": len(unmatched_board),
        "n_snapshot_unmatched": len(unmatched_snap),
        "unmatched_board_ids_sample": sorted({row[0] for row in unmatched_board})[:8],
        "unmatched_snapshot_ids_sample": sorted({row[0] for row in unmatched_snap})[:8],
    }


def empty_match(*, board_n: int = 0, snapshot_n: int = 0) -> dict[str, object]:
    return {
        "join_keys": list(JOIN_KEYS),
        "n_board_unmatched": int(board_n),
        "n_snapshot_unmatched": int(snapshot_n),
        "unmatched_board_ids_sample": [],
        "unmatched_snapshot_ids_sample": [],
    }
