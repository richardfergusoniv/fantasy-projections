"""Regression coverage for matchup snapshots consumed by lineup decisions."""

from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from src.app.api.v1.leagues import get_matchups
from src.app.decisions.services import LineupService, _resolve_owner_roster_id
from src.app.persistence.models import AppUser, LeagueMember, MatchupSnapshot


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
