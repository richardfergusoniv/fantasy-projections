"""Matchup board decision_snapshot cache — hit, miss, invalidate, accuracy."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from src.app.config import get_settings
from src.app.decisions.matchup_cache import (
    MATCHUP_BOARD_KIND,
    invalidate_matchup_boards,
)
from src.app.decisions.services import (
    LeagueContextError,
    LineupService,
    _public_decision_error,
)
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
    assert board["schema_version"] == 2
    assert board["cache_key_source"] == "league_value"
    assert board["owner_actual"]["starters"]
    assert board["owner_optimized"]["starters"]
    assert board["opponent_actual"] is not None
    assert board["opponent_optimized"] is not None
    assert set(board["by_opponent_mode"]) == {"current", "optimized"}

    second = service.recommend("fixture-standard", 1, opponent_mode="current")
    assert second["decision_cache"] == "hit"
    assert second["recommended_starters"] == first["recommended_starters"]
    assert second["expected_points"] == first["expected_points"]
    assert [p["player_id"] for p in second["starters"]] == [
        p["player_id"] for p in first["starters"]
    ]

    # Switching opponent_mode must not recompute when the board is warm.
    optimized = service.recommend("fixture-standard", 1, opponent_mode="optimized")
    assert optimized["decision_cache"] == "hit"
    assert optimized["opponent_mode"] == "optimized"


def test_cache_hit_matches_fresh_bypass(seeded_lineup: Session):
    """decision_cache hit must not drift from a fresh recompute."""
    service = LineupService(seeded_lineup)
    miss = service.recommend("fixture-standard", 1, opponent_mode="current")
    assert miss["decision_cache"] == "miss"
    hit = service.recommend("fixture-standard", 1, opponent_mode="current")
    assert hit["decision_cache"] == "hit"
    fresh = service.recommend(
        "fixture-standard", 1, opponent_mode="current", bypass_cache=True
    )
    assert fresh["decision_cache"] == "miss"

    def _ids(payload, key="starters"):
        return [row["player_id"] for row in payload[key]]

    assert _ids(hit) == _ids(fresh) == _ids(miss)
    assert hit["expected_points"] == fresh["expected_points"] == miss["expected_points"]
    assert hit["opponent_expected_points"] == fresh["opponent_expected_points"]
    assert _ids(hit, "opponent_starter_details") == _ids(
        fresh, "opponent_starter_details"
    )
    assert hit["current_starters"] == fresh["current_starters"]
    assert hit["recommended_starters"] == fresh["recommended_starters"]


def test_starters_are_submitted_sleeper_actual(seeded_lineup: Session):
    """Matchup 'You' column must be current Sleeper starters, not optimized."""
    from src.app.decisions.services import _resolve_owner_roster_id

    service = LineupService(seeded_lineup)
    payload = service.recommend("fixture-standard", 1, opponent_mode="current")
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
    shown = [row["player_id"] for row in payload["starters"]]
    assert shown == payload["current_starters"]
    assert payload["expected_points"] == payload["current_expected_points"]
    # Optimized / recommended may differ from actual; both are published.
    assert payload["recommended_starters"]
    assert payload["owner_actual"]["expected_points"] == payload["expected_points"]


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
    fingerprint = "owner=1:snap:[a]:pool[a]|opp=2:snap2:[b]:pool[b]"
    for source in ("league_value", "vegas_props"):
        stored = store_matchup_board(
            seeded_lineup,
            league_id="fixture-standard",
            week=1,
            board_source=source,
            projection_run_id=run.id,
            roster_snapshot_id="snap",
            input_fingerprint=fingerprint,
            effective_board_source=source,
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
    assert league["cache_key_source"] == "league_value"
    assert vegas["cache_key_source"] == "vegas_props"
    assert league["decision_snapshot_id"] != vegas["decision_snapshot_id"]


def test_weekly_props_missing_pointer_fails_fast(seeded_lineup: Session, monkeypatch):
    """Vegas request without a promoted weekly_props run must not hydrate sealed."""
    monkeypatch.setenv("APP_PROJECTION_SOURCE", "weekly_props")
    get_settings.cache_clear()
    service = LineupService(seeded_lineup)
    with pytest.raises(LeagueContextError, match="missing_weekly_props_pointer"):
        service.recommend(
            "fixture-standard",
            1,
            opponent_mode="current",
            projection_source="weekly_props",
        )
    assert (
        seeded_lineup.query(DecisionSnapshot)
        .filter(DecisionSnapshot.kind == MATCHUP_BOARD_KIND)
        .count()
        == 0
    )
    code, message = _public_decision_error(
        LeagueContextError("missing_weekly_props_pointer:no promoted weekly_props run")
    )
    assert code == "weekly_props_unavailable"
    assert "League Value" in message


def test_legacy_schema_v1_cache_is_treated_as_miss(seeded_lineup: Session):
    from src.app.persistence.models import ProjectionRun

    service = LineupService(seeded_lineup)
    # Warm a real board so we know recommend works, then replace with legacy row.
    service.recommend("fixture-standard", 1, opponent_mode="current")
    run = (
        seeded_lineup.query(ProjectionRun)
        .order_by(ProjectionRun.as_of.desc())
        .first()
    )
    assert run is not None
    seeded_lineup.query(DecisionSnapshot).filter(
        DecisionSnapshot.kind == MATCHUP_BOARD_KIND
    ).delete()
    seeded_lineup.flush()
    # Manually insert a schema_version=1 style payload (no cache_key_source).
    row = DecisionSnapshot(
        kind=MATCHUP_BOARD_KIND,
        league_id="fixture-standard",
        week=1,
        projection_run_id=run.id,
        result_json={
            "board_source": "league_value",
            "input_fingerprint": "stale",
            "schema_version": 1,
            "by_opponent_mode": {
                "current": {"opponent_mode": "current", "starters": []},
                "optimized": {"opponent_mode": "optimized", "starters": []},
            },
        },
        created_at=datetime.now(UTC),
    )
    seeded_lineup.add(row)
    seeded_lineup.flush()

    again = service.recommend("fixture-standard", 1, opponent_mode="current")
    assert again["decision_cache"] == "miss"
    assert again["starters"]
