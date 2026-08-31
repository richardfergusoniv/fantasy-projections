"""End-to-end contract for a fixture Sleeper refresh.

These tests exist because the two fixture universes used to disagree. The seed
described six leagues keyed by canonical (GSIS) player ids; the Sleeper fixture
set described one league whose roster held ids that appeared nowhere else. A
`POST /api/v1/sync` therefore overwrote every roster with unprojectable ids and
silently broke lineup recommendations for all six leagues — while the status-code
smoke checks stayed green, because they only ran before the sync.

So the assertions here are deliberately about *content after a refresh*, not
about status codes: which leagues exist, which slots each league fills, whether
rostered ids resolve to real players, and whether the market signal survives the
trip from Sleeper into the waiver engine.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("APP_ENABLE_DEV_AUTH", "true")
os.environ.setdefault("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:?cache=shared")

#: The product's six leagues: two redraft, four dynasty.
EXPECTED_LEAGUES = {
    "fixture-standard",
    "fixture-ppfd",
    "fixture-superflex",
    "fixture-dynasty",
    "fixture-yardage-bonus",
    "fixture-k-dst",
}
SUPERFLEX_LEAGUES = {"fixture-superflex", "fixture-dynasty"}
KICKER_LEAGUES = {
    "fixture-standard",
    "fixture-superflex",
    "fixture-dynasty",
    "fixture-k-dst",
}
DEFENSE_LEAGUES = {
    "fixture-standard",
    "fixture-superflex",
    "fixture-dynasty",
    "fixture-yardage-bonus",
    "fixture-k-dst",
}


@pytest.fixture()
def synced(monkeypatch):
    """Seed, log in, and run one full fixture refresh through the public API."""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_ENABLE_DEV_AUTH", "true")
    from src.app.config import get_settings
    from src.app.factory import create_app
    from src.app.middleware.rate_limit import limiter
    from src.app.persistence.database import get_session, init_db
    from src.app.seed import seed_development_data

    get_settings.cache_clear()
    limiter.reset()
    init_db()
    with get_session() as session:
        seed = seed_development_data(session, email="owner@example.com")

    client = TestClient(create_app())
    link = client.post(
        "/api/v1/auth/magic-link", json={"email": "owner@example.com"}
    ).json()["development_link"]
    csrf = client.post(
        "/api/v1/auth/verify", json={"token": link.split("token=")[-1]}
    ).json()["csrf_token"]
    headers = {"X-CSRF-Token": csrf}

    before = {
        league_id: client.get(
            f"/api/v1/leagues/{league_id}/lineup/1?opponent_mode=current"
        )
        for league_id in seed["leagues"]
    }
    # The app test harness shares one in-memory database, so a fixed key would
    # be deduplicated against an earlier test's job and skip the refresh.
    run_key = uuid.uuid4().hex
    sync = client.post(
        "/api/v1/sync", headers={**headers, "Idempotency-Key": f"sync-{run_key}"}
    )
    assert sync.status_code == 200, sync.text
    assert sync.json()["status"] == "succeeded", sync.json()

    try:
        yield client, seed, headers, before
    finally:
        limiter.reset()


def test_fixture_refresh_covers_all_six_leagues(synced):
    client, _seed, _headers, _before = synced

    listed = {row["league_id"] for row in client.get("/api/v1/leagues").json()["leagues"]}
    assert EXPECTED_LEAGUES <= listed

    types = {}
    hashes = {}
    for league_id in sorted(EXPECTED_LEAGUES):
        rules = client.get(f"/api/v1/leagues/{league_id}/rules")
        assert rules.status_code == 200, rules.text
        body = rules.json()
        hashes[league_id] = body["contract_hash"]
        types[league_id] = body.get("league_type") or body.get("type")
        # A league whose rules cannot be reproduced exactly must not be
        # publishable, so an unsupported key here is a hard failure.
        assert not body["rules"]["unsupported_keys"], league_id
        assert not body["rules"]["unsupported_slots"], league_id

    # Six leagues, six genuinely different scoring contracts.
    assert len(set(hashes.values())) == 6, hashes


def test_sync_leaves_every_league_recommendable(synced):
    """The regression that motivated this file: sync used to break lineups."""
    client, seed, _headers, before = synced

    for league_id in seed["leagues"]:
        after = client.get(f"/api/v1/leagues/{league_id}/lineup/1?opponent_mode=current")
        assert before[league_id].status_code == 200, league_id
        assert after.status_code == 200, f"{league_id}: {after.text}"
        body = after.json()
        # No rostered player may become unprojectable because of a refresh.
        assert body["players_without_projection"] == [], league_id
        assert body["recommended_starters"], league_id
        assert body["projection_available"] is True, league_id


def test_each_league_fills_only_the_slots_it_declares(synced):
    """Superflex, kicker, and defense coverage must follow the league's rules."""
    client, _seed, _headers, _before = synced

    slots_by_league: dict[str, dict[str, str]] = {}
    for league_id in sorted(EXPECTED_LEAGUES):
        body = client.get(
            f"/api/v1/leagues/{league_id}/lineup/1?opponent_mode=current"
        ).json()
        slots_by_league[league_id] = body["slot_assignments"]

    used = {lid: set(slots.values()) for lid, slots in slots_by_league.items()}
    assert {lid for lid, s in used.items() if "SUPER_FLEX" in s} == SUPERFLEX_LEAGUES
    assert {lid for lid, s in used.items() if "K" in s} == KICKER_LEAGUES
    assert {lid for lid, s in used.items() if "DEF" in s} == DEFENSE_LEAGUES

    # A Superflex slot may only hold a Superflex-eligible player.
    for league_id in SUPERFLEX_LEAGUES:
        body = client.get(
            f"/api/v1/leagues/{league_id}/lineup/1?opponent_mode=current"
        ).json()
        positions = {row["player_id"]: row["position"] for row in body["starters"]}
        for player_id, slot in slots_by_league[league_id].items():
            if slot == "SUPER_FLEX":
                assert positions[player_id] in {"QB", "RB", "WR", "TE"}


def test_synced_rosters_hold_canonical_player_ids(synced):
    """Sleeper ids are resolved at ingest, not stored raw onto rosters."""
    client, _seed, _headers, _before = synced
    from src.app.persistence.database import get_session
    from src.app.persistence.models import PlayerIdentity, RosterSnapshot

    with get_session() as session:
        known = {row.player_id for row in session.query(PlayerIdentity).all()}
        sleeper_ids = {
            row.sleeper_id
            for row in session.query(PlayerIdentity).all()
            if row.sleeper_id and row.sleeper_id != row.player_id
        }
        rostered: set[str] = set()
        for snapshot in (
            session.query(RosterSnapshot)
            .filter(RosterSnapshot.league_id.in_(sorted(EXPECTED_LEAGUES)))
            .all()
        ):
            rostered.update(snapshot.players or [])

    assert rostered, "fixture refresh produced no roster snapshots"
    # Every rostered id is a canonical identity, and none is a raw Sleeper id.
    assert rostered <= known
    assert rostered.isdisjoint(sleeper_ids)


def test_trending_adds_reach_the_waiver_engine_as_canonical_ids(synced):
    """The market signal must survive Sleeper ids -> canonical ids.

    It used to be read from a column that does not exist, under the wrong payload
    keys, in the wrong id space — so it was silently always empty.
    """
    client, _seed, _headers, _before = synced

    body = client.get("/api/v1/leagues/fixture-standard/waivers/1").json()
    considered = body["trending_adds_considered"]
    assert considered, "trending adds were recorded by sync but never reached waivers"
    assert all(not pid.startswith("sl-") for pid in considered), considered

    # And it stays a market signal: urgency is an input, never a projection.
    for row in body["recommendations"]:
        assert "trending_adds" in row["inputs"]


def test_repeated_refresh_is_idempotent(synced):
    """A second refresh must not duplicate rows or change the recommendation."""
    client, _seed, headers, _before = synced
    from src.app.persistence.database import get_session
    from src.app.persistence.models import LeagueRuleSnapshot, RosterSnapshot

    def counts() -> tuple[int, int]:
        with get_session() as session:
            return (
                session.query(RosterSnapshot).count(),
                session.query(LeagueRuleSnapshot).count(),
            )

    first_counts = counts()
    first_lineup = client.get(
        "/api/v1/leagues/fixture-standard/lineup/1?opponent_mode=current"
    ).json()["recommended_starters"]

    again = client.post(
        "/api/v1/sync", headers={**headers, "Idempotency-Key": uuid.uuid4().hex}
    )
    assert again.status_code == 200, again.text

    assert counts() == first_counts
    assert (
        client.get(
            "/api/v1/leagues/fixture-standard/lineup/1?opponent_mode=current"
        ).json()["recommended_starters"]
        == first_lineup
    )
