"""Regression coverage for matchup snapshots consumed by lineup decisions."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from src.app.api.v1.leagues import get_matchups
from src.app.decisions.services import LineupService, _resolve_owner_roster_id
from src.app.persistence.models import AppUser, LeagueMember, MatchupSnapshot
from src.app.persistence.repositories import LeagueRepository


def _snapshot(
    *,
    league_id: str,
    roster_id: int,
    matchup_id: int,
    fetched_at: datetime,
    points: float,
) -> MatchupSnapshot:
    return MatchupSnapshot(
        league_id=league_id,
        week=7,
        roster_id=roster_id,
        matchup_id=matchup_id,
        fetched_at=fetched_at,
        points=points,
    )


def test_lineup_resolves_opponent_from_matchup_pairing(db_session: Session):
    now = datetime.now(UTC)
    db_session.add_all(
        [
            _snapshot(
                league_id="league-1",
                roster_id=1,
                matchup_id=22,
                fetched_at=now,
                points=10,
            ),
            _snapshot(
                league_id="league-1",
                roster_id=2,
                matchup_id=21,
                fetched_at=now,
                points=20,
            ),
            _snapshot(
                league_id="league-1",
                roster_id=3,
                matchup_id=22,
                fetched_at=now,
                points=30,
            ),
        ]
    )
    db_session.flush()

    assert LineupService(db_session)._matchup_opponent_roster_id("league-1", 7, 1) == 3


def test_decisions_resolve_configured_owner_roster(monkeypatch, db_session: Session):
    from src.app.config import get_settings

    monkeypatch.setenv("SLEEPER_USER_ID", "owner-3")
    get_settings.cache_clear()
    db_session.add(
        LeagueMember(
            league_id="league-1",
            user_id="owner-3",
            roster_id=3,
            display_name="Owner",
        )
    )
    db_session.flush()

    assert _resolve_owner_roster_id(
        db_session, "league-1", explicit_roster_id=None
    ) == 3

    get_settings.cache_clear()


def test_matchup_endpoint_returns_latest_score_per_roster(db_session: Session):
    now = datetime.now(UTC)
    db_session.add_all(
        [
            _snapshot(
                league_id="league-1",
                roster_id=1,
                matchup_id=22,
                fetched_at=now - timedelta(minutes=5),
                points=7,
            ),
            _snapshot(
                league_id="league-1",
                roster_id=1,
                matchup_id=22,
                fetched_at=now,
                points=17,
            ),
        ]
    )
    db_session.flush()

    payload = get_matchups(
        "league-1", 7, user=AppUser(email="owner@example.com"), db=db_session
    )

    assert payload["matchups"] == [{"roster_id": 1, "matchup_id": 22, "points": 17}]


def test_owner_roster_resolves_when_manager_has_two_teams(
    monkeypatch, db_session: Session
):
    """``league_member`` is unique on ``(league_id, roster_id)``, not user id.

    A manager running two teams in one league therefore has two rows, which
    must resolve to a stable roster rather than raising ``MultipleResultsFound``
    out of the request path.
    """
    from src.app.config import get_settings

    monkeypatch.setenv("SLEEPER_USER_ID", "owner-3")
    get_settings.cache_clear()
    db_session.add_all(
        [
            LeagueMember(
                league_id="league-1",
                user_id="owner-3",
                roster_id=6,
                display_name="Owner B",
            ),
            LeagueMember(
                league_id="league-1",
                user_id="owner-3",
                roster_id=3,
                display_name="Owner A",
            ),
        ]
    )
    db_session.flush()

    assert (
        _resolve_owner_roster_id(db_session, "league-1", explicit_roster_id=None) == 3
    )


def test_owner_roster_id_is_none_when_user_is_not_a_member(db_session: Session):
    assert (
        LeagueRepository(db_session).owner_roster_id(
            league_id="league-1", user_id="stranger"
        )
        is None
    )


def test_lineup_falls_back_when_paired_roster_has_no_snapshot(
    db_session: Session, monkeypatch
):
    """A live pairing can name a roster whose own snapshot has not landed yet.

    Without a fallback the opponent lineup is empty, which scores the opponent
    at zero and reports a near-certain win instead of failing loudly.
    """
    from src.app.config import get_settings
    from src.app.releases.bridge import ReleaseBridge
    from src.app.seed import seed_development_data

    monkeypatch.setenv("APP_PROJECTION_SOURCE", "sealed_release")
    monkeypatch.setenv("WEEKLY_RND_ENABLED", "false")
    get_settings.cache_clear()
    seed_development_data(db_session, email="owner@example.com")
    if ReleaseBridge(db_session).sync_preseason_pointer(2026) is None:
        pytest.skip("no active release bundle")

    # fixture-standard seeds roster snapshots for rosters 1 and 2 only, so the
    # pairing below points at a roster the lineup path cannot load.
    now = datetime.now(UTC)
    db_session.add_all(
        [
            MatchupSnapshot(
                league_id="fixture-standard",
                week=1,
                roster_id=1,
                matchup_id=5,
                fetched_at=now,
                points=0.0,
            ),
            MatchupSnapshot(
                league_id="fixture-standard",
                week=1,
                roster_id=9,
                matchup_id=5,
                fetched_at=now,
                points=0.0,
            ),
        ]
    )
    db_session.flush()

    result = LineupService(db_session).recommend("fixture-standard", 1)

    assert result["opponent_starters"]
    assert result["opponent_expected_points"] > 0
