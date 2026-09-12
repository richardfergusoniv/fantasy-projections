"""Persist Matchup / lineup boards in ``decision_snapshot``.

The expensive work is joint draws + optimize inside ``LineupService``. This
module stores a ready-made board (owner/opponent × actual/optimized) keyed by
league, week, projection run, board source, and roster fingerprint so
``GET …/lineup/{week}`` can return cached JSON until rosters or props change.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.app.persistence.models import DecisionSnapshot, ProjectionRun

#: One row per league/week/board_source (+ run + roster fingerprint in payload).
MATCHUP_BOARD_KIND = "matchup_board"

BOARD_SOURCES = ("vegas_props", "league_value")


def board_source_for_projection_source(projection_source: str | None) -> str:
    """Map APP_PROJECTION_SOURCE / query override → UI board_source."""
    if projection_source in {"weekly_props", "vegas_props"}:
        return "vegas_props"
    return "league_value"


def projection_source_for_board_source(board_source: str) -> str:
    if board_source == "vegas_props":
        return "weekly_props"
    return "sealed_release"


def roster_input_fingerprint(
    *,
    owner_roster_id: int,
    owner_snapshot_id: str,
    opponent_roster_id: int | None,
    opponent_snapshot_id: str | None,
    owner_starters: list | None,
    opponent_starters: list | None,
    owner_players: list | None = None,
    opponent_players: list | None = None,
) -> str:
    """Stable input key: roster snapshot rows + starters + full player pools.

    Players are included so a bench add/drop without a starter change still
    invalidates optimized lineups (even if snapshot ids were somehow reused).
    """
    own = ",".join(str(p) for p in (owner_starters or []) if p)
    opp = ",".join(str(p) for p in (opponent_starters or []) if p)
    own_pool = ",".join(sorted(str(p) for p in (owner_players or []) if p))
    opp_pool = ",".join(sorted(str(p) for p in (opponent_players or []) if p))
    return (
        f"owner={owner_roster_id}:{owner_snapshot_id}:[{own}]:pool[{own_pool}]|"
        f"opp={opponent_roster_id or 0}:{opponent_snapshot_id or 'none'}:[{opp}]:pool[{opp_pool}]"
    )


def projection_run_exists(session: Session, run_id: str | None) -> bool:
    if not run_id:
        return False
    return (
        session.query(ProjectionRun.id)
        .filter(ProjectionRun.id == run_id)
        .first()
        is not None
    )


def load_matchup_board(
    session: Session,
    *,
    league_id: str,
    week: int,
    board_source: str,
    projection_run_id: str,
    input_fingerprint: str,
) -> dict[str, Any] | None:
    """Return newest matching board payload, or None on miss/stale.

    ``board_source`` is the **requested** UI source (vegas_props | league_value),
    not the effective source after fallback. Rows without ``cache_key_source``
    (pre-fix schema) are treated as misses so sticky wrong boards recompute.
    """
    rows = (
        session.query(DecisionSnapshot)
        .filter(
            DecisionSnapshot.kind == MATCHUP_BOARD_KIND,
            DecisionSnapshot.league_id == league_id,
            DecisionSnapshot.week == week,
            DecisionSnapshot.projection_run_id == projection_run_id,
        )
        .order_by(DecisionSnapshot.created_at.desc())
        .limit(8)
        .all()
    )
    for row in rows:
        payload = row.result_json or {}
        # Require explicit requested-source key; do not match on effective
        # board_source alone (that conflated vegas fallback with league_value).
        if payload.get("cache_key_source") != board_source:
            continue
        if payload.get("input_fingerprint") != input_fingerprint:
            continue
        if payload.get("schema_version", 0) < 2:
            continue
        return {
            **payload,
            "decision_snapshot_id": row.id,
            "cached_at": row.created_at.isoformat()
            if isinstance(row.created_at, datetime)
            else str(row.created_at),
        }
    return None


def store_matchup_board(
    session: Session,
    *,
    league_id: str,
    week: int,
    board_source: str,
    projection_run_id: str,
    roster_snapshot_id: str | None,
    input_fingerprint: str,
    board: dict[str, Any],
    effective_board_source: str | None = None,
) -> DecisionSnapshot | None:
    """Upsert-style write: drop stale twins, insert fresh board.

    ``board_source`` is the requested UI cache key. ``effective_board_source`` is
    what decisions actually scored (may differ on weekly_props fallback).
    """
    if not projection_run_exists(session, projection_run_id):
        return None

    effective = effective_board_source or board.get("board_source") or board_source

    stale = (
        session.query(DecisionSnapshot)
        .filter(
            DecisionSnapshot.kind == MATCHUP_BOARD_KIND,
            DecisionSnapshot.league_id == league_id,
            DecisionSnapshot.week == week,
            DecisionSnapshot.projection_run_id == projection_run_id,
        )
        .all()
    )
    for row in stale:
        payload = row.result_json or {}
        # Drop twins for this requested source, plus legacy rows that only
        # keyed on effective board_source (schema_version < 2).
        key = payload.get("cache_key_source")
        if key == board_source or (
            key is None and payload.get("board_source") in {board_source, effective}
        ):
            session.delete(row)

    result_json = {
        **board,
        "board_source": effective,
        "cache_key_source": board_source,
        "requested_board_source": board_source,
        "input_fingerprint": input_fingerprint,
        "schema_version": 2,
    }
    row = DecisionSnapshot(
        kind=MATCHUP_BOARD_KIND,
        league_id=league_id,
        week=week,
        projection_run_id=projection_run_id,
        roster_snapshot_id=roster_snapshot_id,
        result_json=result_json,
        created_at=datetime.now(UTC),
    )
    session.add(row)
    session.flush()
    return row


def invalidate_matchup_boards(
    session: Session,
    *,
    league_id: str | None = None,
    week: int | None = None,
) -> int:
    """Delete cached boards (call after roster sync or projection promote)."""
    query = session.query(DecisionSnapshot).filter(
        DecisionSnapshot.kind == MATCHUP_BOARD_KIND
    )
    if league_id is not None:
        query = query.filter(DecisionSnapshot.league_id == league_id)
    if week is not None:
        query = query.filter(DecisionSnapshot.week == week)
    rows = query.all()
    count = len(rows)
    for row in rows:
        session.delete(row)
    if count:
        session.flush()
    return count


def mode_payload_from_board(board: dict[str, Any], opponent_mode: str) -> dict[str, Any]:
    """Slice a stored board into the existing lineup recommend response shape."""
    by_mode = board.get("by_opponent_mode") or {}
    payload = by_mode.get(opponent_mode)
    if not isinstance(payload, dict):
        raise KeyError(f"missing_opponent_mode:{opponent_mode}")
    out = dict(payload)
    cache_hit = bool(board.get("_cache_hit"))
    cache_label = "hit" if cache_hit else "miss"
    meta = dict(out.get("meta") or {})
    meta["decision_cache"] = cache_label
    if board.get("decision_snapshot_id"):
        meta["decision_snapshot_id"] = board.get("decision_snapshot_id")
    # Surface actual/optimized sides for clients that want the full board.
    if board.get("owner_actual") is not None:
        out["owner_actual"] = board["owner_actual"]
    if board.get("owner_optimized") is not None:
        out["owner_optimized"] = board["owner_optimized"]
    if board.get("requested_board_source") is not None:
        out["requested_board_source"] = board["requested_board_source"]
        meta["requested_board_source"] = board["requested_board_source"]
    out["meta"] = meta
    out["decision_cache"] = cache_label
    return out
