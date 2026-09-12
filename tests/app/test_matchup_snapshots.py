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


def test_lineup_prefers_coherent_generation_when_owner_row_stale(db_session: Session):
    """Owner matchup row skipped on later syncs must not pair across generations.

    Prefer an older owner row that still shares fetched_at with its peer over a
    newer owner-only row that would fall back to a rematched peer.
    """
    old = datetime.now(UTC) - timedelta(days=10)
    mid = datetime.now(UTC) - timedelta(days=1)
    new = datetime.now(UTC)
    db_session.add_all(
        [
            # Coherent historical pairing: owner 1 vs 3
            _snapshot(
                league_id="league-1",
                roster_id=1,
                matchup_id=4,
                fetched_at=old,
                points=0,
            ),
            _snapshot(
                league_id="league-1",
                roster_id=3,
                matchup_id=4,
                fetched_at=old,
                points=0,
            ),
            # Later rematch written for peer only (owner upsert skipped historically)
            _snapshot(
                league_id="league-1",
                roster_id=3,
                matchup_id=9,
                fetched_at=mid,
                points=5,
            ),
            _snapshot(
                league_id="league-1",
                roster_id=5,
                matchup_id=9,
                fetched_at=mid,
                points=5,
            ),
            # Newest owner-only refresh still claims matchup 4 (stale claim)
            _snapshot(
                league_id="league-1",
                roster_id=1,
                matchup_id=4,
                fetched_at=new,
                points=0,
            ),
        ]
    )
    db_session.flush()

    # Walks back to the coherent old generation (1 vs 3), not cross-gen 1 vs 3@mid
    # via matchup_id fallback against the rematched world.
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


def test_submitted_seat_labels_follow_sleeper_order():
    from src.app.decisions.lineup import assign_submitted_seat_labels, expand_seats
    from src.app.scoring.compiler import compile_sleeper_scoring

    contract = compile_sleeper_scoring(
        {"pass_td": 4, "rush_td": 6, "rec": 1},
        ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"],
    )
    submitted = ["qb", "rb1", "rb2", "wr1", "wr2", "te", "flex", "k", "dst"]
    labels = assign_submitted_seat_labels(contract, submitted)
    assert [labels[pid] for pid in submitted] == [seat.slot for seat in expand_seats(contract)]


def test_matchup_upsert_writes_shared_generation(db_session: Session):
    from src.app.league.sleeper.sync import SleeperSyncService

    old = datetime.now(UTC) - timedelta(days=5)
    # Split generation (historical bug): owner stale, peer fresh, same matchup_id.
    db_session.add_all(
        [
            _snapshot(
                league_id="league-sync",
                roster_id=1,
                matchup_id=2,
                fetched_at=old,
                points=0,
            ),
            _snapshot(
                league_id="league-sync",
                roster_id=2,
                matchup_id=2,
                fetched_at=old + timedelta(days=4),
                points=0,
            ),
        ]
    )
    db_session.flush()

    sync = SleeperSyncService(db_session, use_fixtures=True)
    inserted = sync._upsert_matchup_snapshots(
        "league-sync",
        [
            {"roster_id": 1, "matchup_id": 2, "points": 0},
            {"roster_id": 2, "matchup_id": 2, "points": 0},
        ],
        week=7,
    )
    assert inserted == 2
    rows = (
        db_session.query(MatchupSnapshot)
        .filter(MatchupSnapshot.league_id == "league-sync", MatchupSnapshot.week == 7)
        .order_by(MatchupSnapshot.fetched_at.desc())
        .all()
    )
    newest_times = {row.fetched_at for row in rows[:2]}
    assert len(newest_times) == 1
    assert LineupService(db_session)._matchup_opponent_roster_id("league-sync", 7, 1) == 2

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
