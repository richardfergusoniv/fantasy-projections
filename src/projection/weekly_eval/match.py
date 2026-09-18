"""Join diagnostics for Role 2 board ↔ snapshot matching.

Does not fuzzy-match names. Live weeks must use the same ``player_id`` as the
M3 / sealed board. Fixture dry-run names (``p-qb``) will not join to gsis ids.
"""
from __future__ import annotations

import pandas as pd

from src.projection.weekly_eval.leakage import JOIN_KEYS


def _missing_join_keys(frame: pd.DataFrame) -> list[str]:
    return [k for k in JOIN_KEYS if k not in frame.columns]


def _require_join_keys(frame: pd.DataFrame, *, label: str) -> None:
    if frame is None or frame.empty:
        return
    missing = _missing_join_keys(frame)
    if missing:
        raise ValueError(f"{label} missing join keys: {missing}")


def _key_tuples(
    frame: pd.DataFrame | None,
    *,
    label: str,
) -> set[tuple[str, int, int, str]]:
    if frame is None or frame.empty:
        return set()
    _require_join_keys(frame, label=label)
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
    board_keys = _key_tuples(board, label="board")
    snap_keys = _key_tuples(
        snapshots if snapshots is not None else pd.DataFrame(),
        label="snapshots",
    )
    joined_keys = _key_tuples(joined, label="joined")
    unmatched_board = board_keys - joined_keys
    unmatched_snap = snap_keys - joined_keys
    return {
        "join_keys": list(JOIN_KEYS),
        "n_board_unmatched": len(unmatched_board),
        "n_snapshot_unmatched": len(unmatched_snap),
        "unmatched_board_ids_sample": sorted({row[0] for row in unmatched_board})[:8],
        "unmatched_snapshot_ids_sample": sorted({row[0] for row in unmatched_snap})[:8],
    }


def empty_match(
    board: pd.DataFrame | None = None,
    snapshots: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Unmatched counts are distinct join-key tuples, same as ``describe_match``."""
    board_keys = _key_tuples(board if board is not None else pd.DataFrame(), label="board")
    snap_keys = _key_tuples(
        snapshots if snapshots is not None else pd.DataFrame(),
        label="snapshots",
    )
    return {
        "join_keys": list(JOIN_KEYS),
        "n_board_unmatched": len(board_keys),
        "n_snapshot_unmatched": len(snap_keys),
        "unmatched_board_ids_sample": sorted({row[0] for row in board_keys})[:8],
        "unmatched_snapshot_ids_sample": sorted({row[0] for row in snap_keys})[:8],
    }
