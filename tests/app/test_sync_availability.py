"""Sleeper sync, availability, and dynasty tests."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:?cache=shared")


def _source_snapshot(session: Session, **overrides) -> "SourceSnapshot":  # noqa: F821
    from src.app.persistence.models import SourceSnapshot

    fields = {
        "endpoint": "players/nfl",
        "request_params_json": {},
        "fetched_at": datetime.now(UTC),
        "body_hash": "abc",
        "artifact_uri": "local://test",
        "health_verdict": "healthy",
        "is_complete": True,
    }
    fields.update(overrides)
    snapshot = SourceSnapshot(**fields)
    session.add(snapshot)
    session.flush()
    return snapshot


def test_sleeper_fixture_sync(db_session: Session):
    from src.app.league.sleeper.sync import SleeperSyncService

    sync = SleeperSyncService(db_session, use_fixtures=True)
    user = sync.connect_user("fixture_owner")
    assert user["user_id"] == "fixture-user-1"
    availability = sync.sync_player_availability()
    # Repeat syncs report an already-active event as unchanged instead of
    # activating a duplicate, so either counter proves the payload was applied.
    assert availability["activated"] + availability["unchanged"] >= 1
    assert availability["payload_stale"] is False
    leagues = sync.sync_leagues("fixture-user-1", 2026)
    assert "fixture-standard" in leagues
    from src.app.persistence.repositories import LeagueRepository

    repo = LeagueRepository(db_session)
    rules = repo.latest_rules("fixture-standard")
    assert rules is not None
    assert rules.contract_hash


def test_availability_lifecycle(db_session: Session):
    from src.app.availability.service import AvailabilityService, EvidenceClaim

    service = AvailabilityService(db_session)
    snapshot = _source_snapshot(db_session)
    claim = EvidenceClaim(
        player_id="p1",
        status="questionable",
        reported_injury="ankle",
        expected_return_min="2026-09-20",
        expected_return_max="2026-10-01",
        claim_confidence=0.7,
        sources=[
            {
                "url": "https://www.espn.com/nfl/story/_/id/p1-ankle",
                "title": "Report",
                "published_at": "2026-08-30",
            }
        ],
    )
    evidence = service.add_evidence(claim)
    event = service.activate_event(
        player_id="p1",
        event_type="injury",
        source_snapshot_id=snapshot.id,
        evidence_ids=[evidence.id],
        policy={"play_probability": 0.6},
    )
    assert event.cleared_at is None
    assert service.active_policy_for_player("p1")["play_probability"] == 0.6
    assert service.try_clear_for_player("p1", snapshot, player_count=20_000) == 1
    assert service.active_policy_for_player("p1")["play_probability"] == 1.0
    assert service.try_clear_for_player("p1", snapshot, player_count=5) == 0


def test_sleeper_sync_is_idempotent(db_session: Session):
    """Re-running the full fixture sync must not duplicate snapshot rows."""

    from src.app.league.sleeper.sync import SleeperSyncService
    from src.app.persistence.models import (
        LeagueRuleSnapshot,
        MatchupSnapshot,
        RosterSnapshot,
        SourceSnapshot,
        TradedPick,
    )

    tracked = (LeagueRuleSnapshot, RosterSnapshot, MatchupSnapshot, TradedPick, SourceSnapshot)

    def counts() -> dict[str, int]:
        return {model.__name__: db_session.query(model).count() for model in tracked}

    sync = SleeperSyncService(db_session, use_fixtures=True)
    sync.connect_user("fixture_owner")
    sync.sync_player_availability()
    sync.sync_leagues("fixture-user-1", 2026)
    after_first = counts()

    sync.connect_user("fixture_owner")
    sync.sync_player_availability()
    sync.sync_leagues("fixture-user-1", 2026)
    after_second = counts()

    assert after_second == after_first
    assert after_first["RosterSnapshot"] >= 2
    assert after_first["LeagueRuleSnapshot"] >= 1
    assert after_first["MatchupSnapshot"] >= 2


def test_season_chain_traversal_terminates_on_cycle(db_session: Session):
    """chain-a -> chain-b -> chain-a must be walked once and then stop."""

    from src.app.league.sleeper.sync import SleeperSyncService
    from src.app.persistence.models import League

    sync = SleeperSyncService(db_session, use_fixtures=True)
    chain = sync.sync_season_history(
        {"league_id": "chain-head", "season": 2026, "previous_league_id": "chain-a"}
    )

    assert chain == ["chain-a", "chain-b"]
    rows = {row.league_id: row for row in db_session.query(League).filter(League.league_id.in_(chain)).all()}
    assert rows["chain-a"].previous_league_id == "chain-b"
    assert rows["chain-b"].previous_league_id == "chain-a"
    assert sync.sync_season_history({"league_id": "chain-a", "previous_league_id": "chain-a"}) == []


def test_season_chain_traversal_is_depth_bounded(db_session: Session, monkeypatch):
    from src.app.league.sleeper import sync as sync_module

    monkeypatch.setattr(sync_module, "MAX_SEASON_CHAIN_DEPTH", 1)
    sync = sync_module.SleeperSyncService(db_session, use_fixtures=True)
    chain = sync.sync_season_history(
        {"league_id": "chain-head", "season": 2026, "previous_league_id": "chain-a"}
    )
    assert chain == ["chain-a"]


def test_trending_adds_are_stored_as_market_signal_only(db_session: Session):
    from src.app.league.sleeper.sync import MARKET_SIGNAL_ENDPOINT, SleeperSyncService
    from src.app.persistence.models import PlayerProjection, SourceSnapshot

    projections_before = db_session.query(PlayerProjection).count()
    sync = SleeperSyncService(db_session, use_fixtures=True)
    signal = sync.sync_market_signals("fixture-standard", week=1)
    other = sync.sync_market_signals("fixture-ppfd", week=1)

    assert signal is not None
    assert signal["signal_type"] == "market_urgency"
    assert signal["projection_input"] is False
    assert len(signal["players"]) == 3
    # Scoped lookup: one league's market urgency must not be served to another.
    assert sync.latest_market_signal("fixture-standard") == signal
    assert sync.latest_market_signal("fixture-ppfd") == other
    assert (
        db_session.query(SourceSnapshot).filter(SourceSnapshot.endpoint == MARKET_SIGNAL_ENDPOINT).count() >= 1
    )
    # The invariant that matters is that market urgency never becomes a
    # forecast: syncing the signal writes no projection rows. (Requiring a
    # disjoint *id space* would be the wrong test — it is exactly what used to
    # make the signal unusable in the waiver engine.)
    assert db_session.query(PlayerProjection).count() == projections_before


def test_league_drafts_are_imported(db_session: Session):
    from src.app.league.sleeper.sync import SleeperSyncService
    from src.app.persistence.models import LeagueDraftRule

    sync = SleeperSyncService(db_session, use_fixtures=True)
    drafts = sync.sync_drafts("fixture-superflex")
    sync.sync_drafts("fixture-superflex")

    assert [draft["draft_id"] for draft in drafts] == ["fixture-superflex-draft"]
    assert drafts[0]["rounds"] == 3
    # Re-running the import must not append a second rule row.
    rules = (
        db_session.query(LeagueDraftRule)
        .filter(
            LeagueDraftRule.league_id == "fixture-superflex",
            LeagueDraftRule.rule == "max_pf",
        )
        .all()
    )
    assert len(rules) == 1


def test_redraft_league_never_gets_a_rookie_pick_order_rule(db_session: Session):
    """Rookie-pick order is a dynasty concept; a redraft draft states none."""

    from src.app.league.sleeper.sync import SleeperSyncService
    from src.app.persistence.models import LeagueDraftRule

    sync = SleeperSyncService(db_session, use_fixtures=True)
    sync.sync_drafts("fixture-standard")

    assert (
        db_session.query(LeagueDraftRule)
        .filter(LeagueDraftRule.league_id == "fixture-standard")
        .count()
        == 0
    )


def test_the_four_dynasty_leagues_carry_the_two_configured_pick_rules(db_session: Session):
    """Two leagues use max potential points, two use reverse standings."""

    from src.app.league.sleeper.sync import SleeperSyncService
    from src.app.persistence.models import LeagueDraftRule

    sync = SleeperSyncService(db_session, use_fixtures=True)
    for league_id in (
        "fixture-superflex",
        "fixture-yardage-bonus",
        "fixture-dynasty",
        "fixture-k-dst",
    ):
        sync.sync_drafts(league_id)

    rules = {
        row.league_id: row.rule for row in db_session.query(LeagueDraftRule).all()
    }
    assert rules["fixture-superflex"] == "max_pf"
    assert rules["fixture-yardage-bonus"] == "max_pf"
    assert rules["fixture-dynasty"] == "reverse_standings"
    assert rules["fixture-k-dst"] == "reverse_standings"


def test_members_link_from_roster_owner_without_user_metadata(db_session: Session):
    """The fixtures carry no `user.metadata.roster_id`; linking must still work."""

    from src.app.league.sleeper.sync import SleeperSyncService
    from src.app.persistence.models import LeagueMember

    sync = SleeperSyncService(db_session, use_fixtures=True)
    users = sync.client.get_users("fixture-standard")
    rosters = sync.client.get_rosters("fixture-standard")
    assert all("roster_id" not in (user.get("metadata") or {}) for user in users)

    sync._link_members("fixture-standard", users, rosters)
    members = {
        row.roster_id: row
        for row in db_session.query(LeagueMember).filter(LeagueMember.league_id == "fixture-standard").all()
    }
    assert members[1].user_id == "fixture-user-1"
    assert members[1].display_name == "Owner"
    assert members[2].user_id == "fixture-user-2"


def test_identity_resolution_never_merges_namesakes(db_session: Session):
    from src.app.availability.sync import AvailabilitySyncService
    from src.app.persistence.models import PlayerIdentity

    db_session.add_all(
        [
            PlayerIdentity(
                player_id="ns-buf",
                sleeper_id="ns-sleeper-1",
                gsis_id="00-ns-1",
                name="Namesake Player",
                position="QB",
                team="BUF",
            ),
            PlayerIdentity(
                player_id="ns-jax",
                sleeper_id="ns-sleeper-2",
                gsis_id="00-ns-2",
                name="Namesake Player",
                position="TE",
                team="JAX",
            ),
        ]
    )
    db_session.flush()

    service = AvailabilitySyncService(db_session)
    assert service.resolve_player_id("ns-sleeper-1", {"gsis_id": "00-ns-1"}) == "ns-buf"
    assert service.resolve_player_id("ns-sleeper-2", {"gsis_id": "00-ns-2"}) == "ns-jax"


def test_dynasty_manager_state(db_session: Session):
    from src.app.decisions.dynasty import DynastyService

    service = DynastyService(db_session)
    result = service.infer_manager_state(
        league_id="fixture-dynasty",
        roster_id=1,
        lineup_strength=0.8,
        ros_win_prob=0.7,
        multi_year_value=0.6,
        pick_capital=0.3,
    )
    assert result.label in {"contender", "fringe", "retooling", "rebuilding"}
    assert abs(sum(result.probabilities.values()) - 1.0) < 0.01
    pick = service.project_rookie_pick_slot(
        league_id="fixture-dynasty",
        roster_id=1,
        optimal_points=1200,
        potential_points=1300,
        projected_record=6.5,
    )
    assert pick["rule"] in {"max_pf", "reverse_standings"}


def test_weekly_close_guard_blocks_unfinished_week():
    from src.app.jobs.schedule_guards import weekly_close_allowed

    allowed, reason = weekly_close_allowed({"week_has_completed": False, "season_type": "regular"})
    assert allowed is False
    assert reason == "scheduled_games_not_final"
    allowed, _ = weekly_close_allowed({"week_has_completed": True, "display_week": 1, "week": 1})
    assert allowed is True


def test_full_release_postponed_when_week_open(db_session: Session):
    from unittest.mock import patch

    from src.app.jobs.handlers import run_full_release

    with patch("src.app.jobs.handlers.weekly_close_allowed", return_value=(False, "scheduled_games_not_final")):
        result = run_full_release(db_session)
    assert result["status"] == "postponed"
