"""Matchup board decision_snapshot cache — hit, miss, invalidate."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from src.app.config import get_settings
from src.app.decisions.matchup_cache import (
    MATCHUP_BOARD_KIND,
    invalidate_matchup_boards,
)
from src.app.decisions.services import LineupService
from src.app.persistence.models import DecisionSnapshot, RosterSnapshot
from src.app.releases.bridge import ReleaseBridge
from src.app.seed import seed_development_data


@pytest.fixture()
def seeded_lineup(db_session: Session, monkeypatch):
    monkeypatch.setenv("APP_PROJECTION_SOURCE", "sealed_release")
    get_settings.cache_clear()
    seed_development_data(db_session, email="owner@example.com")
    bridge = ReleaseBridge(db_session)
    if bridge.sync_preseason_pointer(2026) is None:
        pytest.skip("no active release bundle")
    yield db_session
    get_settings.cache_clear()


def test_recommend_caches_board_then_serves_hit(seeded_lineup: Session):
    service = LineupService(seeded_lineup)
    first = service.recommend("fixture-standard", 1, opponent_mode="current")
    assert first["decision_cache"] == "miss"
    assert first["starters"]
    assert first["board_source"] == "league_value"

    rows = (
        seeded_lineup.query(DecisionSnapshot)
        .filter(DecisionSnapshot.kind == MATCHUP_BOARD_KIND)
        .all()
    )
    assert len(rows) == 1
    board = rows[0].result_json
    assert board["owner_actual"]["starters"]
    assert board["owner_optimized"]["starters"]
    assert board["opponent_actual"] is not None
    assert board["opponent_optimized"] is not None
    assert set(board["by_opponent_mode"]) == {"current", "optimized"}

    second = service.recommend("fixture-standard", 1, opponent_mode="current")
    assert second["decision_cache"] == "hit"
    assert second["recommended_starters"] == first["recommended_starters"]
    assert second["expected_points"] == first["expected_points"]

    # Switching opponent_mode must not recompute when the board is warm.
    optimized = service.recommend("fixture-standard", 1, opponent_mode="optimized")
    assert optimized["decision_cache"] == "hit"
    assert optimized["opponent_mode"] == "optimized"


def test_roster_change_invalidates_fingerprint(seeded_lineup: Session):
    from src.app.decisions.services import _resolve_owner_roster_id

    service = LineupService(seeded_lineup)
    first = service.recommend("fixture-standard", 1, opponent_mode="current")
    assert first["decision_cache"] == "miss"

    owner_roster_id = _resolve_owner_roster_id(
        seeded_lineup, "fixture-standard", explicit_roster_id=None
    )
    latest = (
        seeded_lineup.query(RosterSnapshot)
        .filter(
            RosterSnapshot.league_id == "fixture-standard",
            RosterSnapshot.week == 1,
            RosterSnapshot.roster_id == owner_roster_id,
        )
        .order_by(RosterSnapshot.fetched_at.desc(), RosterSnapshot.id.desc())
        .first()
    )
    assert latest is not None
    starters = list(latest.starters or [])
    assert len(starters) >= 2
    # Sync appends a new snapshot when starters change.
    seeded_lineup.add(
        RosterSnapshot(
            league_id=latest.league_id,
            week=latest.week,
            roster_id=latest.roster_id,
            fetched_at=datetime.now(UTC),
            players=list(latest.players or []),
            starters=list(reversed(starters)),
            reserve=list(latest.reserve or []),
        )
    )
    seeded_lineup.flush()

    refreshed = service.recommend("fixture-standard", 1, opponent_mode="current")
    assert refreshed["decision_cache"] == "miss"


def test_invalidate_matchup_boards_drops_rows(seeded_lineup: Session):
    service = LineupService(seeded_lineup)
    service.recommend("fixture-standard", 1, opponent_mode="current")
    assert (
        seeded_lineup.query(DecisionSnapshot)
        .filter(DecisionSnapshot.kind == MATCHUP_BOARD_KIND)
        .count()
        == 1
    )
    dropped = invalidate_matchup_boards(
        seeded_lineup, league_id="fixture-standard", week=1
    )
    assert dropped == 1
    assert (
        seeded_lineup.query(DecisionSnapshot)
        .filter(DecisionSnapshot.kind == MATCHUP_BOARD_KIND)
        .count()
        == 0
    )
    again = service.recommend("fixture-standard", 1, opponent_mode="current")
    assert again["decision_cache"] == "miss"


def test_vegas_and_league_value_are_separate_cache_keys(seeded_lineup: Session):
    from src.app.decisions.matchup_cache import (
        load_matchup_board,
        store_matchup_board,
    )
    from src.app.persistence.models import ProjectionRun

    run = (
        seeded_lineup.query(ProjectionRun)
        .order_by(ProjectionRun.as_of.desc())
        .first()
    )
    assert run is not None
    fingerprint = "owner=1:snap:[a]|opp=2:snap2:[b]"
    for source in ("league_value", "vegas_props"):
        stored = store_matchup_board(
            seeded_lineup,
            league_id="fixture-standard",
            week=1,
            board_source=source,
            projection_run_id=run.id,
            roster_snapshot_id="snap",
            input_fingerprint=fingerprint,
            board={
                "by_opponent_mode": {
                    "current": {"opponent_mode": "current", "board_source": source},
                    "optimized": {"opponent_mode": "optimized", "board_source": source},
                },
                "owner_actual": {"starters": []},
                "owner_optimized": {"starters": []},
                "opponent_actual": {"starters": []},
                "opponent_optimized": {"starters": []},
            },
        )
        assert stored is not None

    league = load_matchup_board(
        seeded_lineup,
        league_id="fixture-standard",
        week=1,
        board_source="league_value",
        projection_run_id=run.id,
        input_fingerprint=fingerprint,
    )
    vegas = load_matchup_board(
        seeded_lineup,
        league_id="fixture-standard",
        week=1,
        board_source="vegas_props",
        projection_run_id=run.id,
        input_fingerprint=fingerprint,
    )
    assert league is not None and vegas is not None
    assert league["board_source"] == "league_value"
    assert vegas["board_source"] == "vegas_props"
    assert league["decision_snapshot_id"] != vegas["decision_snapshot_id"]
