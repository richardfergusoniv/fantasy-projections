"""Regression coverage for decision load failures after live identity drift."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import ClassVar

import pytest
from sqlalchemy.orm import Session

from src.app.api.v1.leagues import get_matchups, get_rosters
from src.app.availability.gsis_link import link_identities_to_release_players
from src.app.config import get_settings
from src.app.decisions.services import (
    _USERNAME_TO_USER_ID_CACHE,
    LeagueContextError,
    LineupService,
    TradeService,
    _public_decision_error,
    _resolve_owner_roster_id,
)
from src.app.persistence.models import (
    AppUser,
    League,
    LeagueMember,
    LeagueRuleSnapshot,
    MatchupSnapshot,
    PlayerIdentity,
    RosterSnapshot,
)
from src.app.persistence.repositories import LeagueRepository
from src.app.projections.loader import PlayerSummary


def _league(session: Session, league_id: str = "league-1", *, season: int = 2026) -> League:
    row = League(
        league_id=league_id,
        season=season,
        name="Test League",
        league_type="redraft",
        raw_json={"roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "BN"]},
    )
    session.add(row)
    session.add(
        LeagueRuleSnapshot(
            league_id=league_id,
            fetched_at=datetime.now(UTC),
            raw_json={"pass_yd": 0.04, "pass_td": 4, "rush_yd": 0.1, "rush_td": 6, "rec": 1, "rec_yd": 0.1, "rec_td": 6},
            normalized_json={},
            contract_hash="test-contract",
        )
    )
    session.flush()
    return row


def test_latest_roster_snapshot_per_roster_dedupes_history(db_session: Session):
    now = datetime.now(UTC)
    db_session.add_all(
        [
            RosterSnapshot(
                league_id="league-1",
                week=1,
                roster_id=7,
                fetched_at=now - timedelta(hours=2),
                players=["old"],
                starters=["old"],
                reserve=[],
            ),
            RosterSnapshot(
                league_id="league-1",
                week=1,
                roster_id=7,
                fetched_at=now,
                players=["new"],
                starters=["new"],
                reserve=[],
            ),
            RosterSnapshot(
                league_id="league-1",
                week=1,
                roster_id=8,
                fetched_at=now,
                players=["other"],
                starters=["other"],
                reserve=[],
            ),
        ]
    )
    db_session.flush()
    rows = LeagueRepository(db_session).latest_rosters("league-1", 1)
    assert [(r.roster_id, r.players) for r in rows] == [(7, ["new"]), (8, ["other"])]


def test_equal_timestamp_roster_tie_breaks_on_id(db_session: Session):
    now = datetime.now(UTC)
    older = RosterSnapshot(
        id="a-older-id",
        league_id="league-1",
        week=1,
        roster_id=1,
        fetched_at=now,
        players=["a"],
        starters=[],
        reserve=[],
    )
    newer = RosterSnapshot(
        id="z-newer-id",
        league_id="league-1",
        week=1,
        roster_id=1,
        fetched_at=now,
        players=["z"],
        starters=[],
        reserve=[],
    )
    db_session.add_all([older, newer])
    db_session.flush()
    rows = LeagueRepository(db_session).latest_rosters("league-1", 1)
    assert rows[0].players == ["z"]


def test_matchup_endpoint_equal_timestamp_tie_break(db_session: Session):
    now = datetime.now(UTC)
    db_session.add_all(
        [
            MatchupSnapshot(
                id="a1",
                league_id="league-1",
                week=7,
                roster_id=1,
                matchup_id=2,
                fetched_at=now,
                points=1,
            ),
            MatchupSnapshot(
                id="z9",
                league_id="league-1",
                week=7,
                roster_id=1,
                matchup_id=2,
                fetched_at=now,
                points=9,
            ),
        ]
    )
    db_session.flush()
    payload = get_matchups("league-1", 7, user=AppUser(email="owner@example.com"), db=db_session)
    assert payload["matchups"] == [{"roster_id": 1, "matchup_id": 2, "points": 9}]


def test_matchup_pairing_bye_and_invalid_matchup_id(db_session: Session):
    now = datetime.now(UTC)
    db_session.add(
        MatchupSnapshot(
            league_id="league-1",
            week=7,
            roster_id=1,
            matchup_id=0,
            fetched_at=now,
            points=0,
        )
    )
    db_session.flush()
    assert LineupService(db_session)._matchup_opponent_roster_id("league-1", 7, 1) is None


def test_owner_matchup_without_paired_roster_snapshot_degrades(monkeypatch, db_session: Session):
    _league(db_session)
    now = datetime.now(UTC)
    db_session.add_all(
        [
            LeagueMember(league_id="league-1", user_id="owner-7", roster_id=7, display_name="Owner"),
            RosterSnapshot(
                league_id="league-1",
                week=1,
                roster_id=7,
                fetched_at=now,
                players=["00-qb"],
                starters=["00-qb"],
                reserve=[],
            ),
            MatchupSnapshot(
                league_id="league-1",
                week=1,
                roster_id=7,
                matchup_id=3,
                fetched_at=now,
                points=0,
            ),
            MatchupSnapshot(
                league_id="league-1",
                week=1,
                roster_id=9,
                matchup_id=3,
                fetched_at=now,
                points=0,
            ),
            PlayerIdentity(player_id="00-qb", sleeper_id="s-qb", gsis_id="00-qb", name="QB", position="QB", team="BUF"),
        ]
    )
    db_session.flush()
    monkeypatch.setenv("SLEEPER_USER_ID", "owner-7")
    get_settings.cache_clear()

    class _Bundle:
        namespace = "test-bundle"
        meta: ClassVar[dict[str, str]] = {"scoring": "ppr"}

        def load_bundle(self):
            return self

        def get(self, pid):
            if pid == "00-qb":
                return PlayerSummary(
                    player_id="00-qb",
                    name="QB",
                    position="QB",
                    team="BUF",
                    mean_points=18.0,
                    quantiles={"p10": 10, "p50": 18, "p90": 26},
                    availability_probability=1.0,
                )
            return None

        def load(self):
            summary = self.get("00-qb")
            return {"00-qb": summary} if summary else {}

        def available_pool(self, rostered):
            summary = self.get("00-qb")
            return [summary] if summary else []

        def as_of(self):
            return now.isoformat()

    monkeypatch.setattr(
        "src.app.decisions.services.get_bundle_loader",
        lambda season: _Bundle(),
    )
    monkeypatch.setattr(
        "src.app.projections.service.ProjectionService.matchup_win_probability_allowed",
        lambda self, **kwargs: True,
    )
    # Opponent roster 9 has a matchup row but no roster snapshot → degraded WP.
    result = LineupService(db_session).recommend("league-1", 1, opponent_mode="current")
    assert result["matchup_win_probability_available"] is False
    assert result["win_probability"] is None
    assert result["matchup_degraded"] is True
    get_settings.cache_clear()


def test_configured_owner_not_roster_one(monkeypatch, db_session: Session):
    monkeypatch.setenv("SLEEPER_USER_ID", "owner-11")
    get_settings.cache_clear()
    db_session.add(
        LeagueMember(league_id="league-1", user_id="owner-11", roster_id=11, display_name="Owner")
    )
    db_session.flush()
    assert _resolve_owner_roster_id(db_session, "league-1", explicit_roster_id=None) == 11
    get_settings.cache_clear()


def test_configured_owner_missing_membership_fails_closed(monkeypatch, db_session: Session):
    monkeypatch.setenv("SLEEPER_USER_ID", "missing-owner")
    get_settings.cache_clear()
    db_session.add(
        LeagueMember(league_id="league-1", user_id="someone-else", roster_id=3, display_name="Other")
    )
    db_session.flush()
    with pytest.raises(LeagueContextError, match="owner_roster_not_found"):
        _resolve_owner_roster_id(db_session, "league-1", explicit_roster_id=None)
    get_settings.cache_clear()


def test_empty_league_memberships_are_sync_gap(monkeypatch, db_session: Session):
    monkeypatch.setenv("SLEEPER_USER_ID", "owner-7")
    get_settings.cache_clear()
    with pytest.raises(LeagueContextError, match="league_membership_not_synced"):
        _resolve_owner_roster_id(db_session, "league-empty", explicit_roster_id=None)
    code, message = _public_decision_error(
        LeagueContextError("league_membership_not_synced:league=league-empty")
    )
    assert code == "league_membership_unavailable"
    assert "SLEEPER_USER_ID" not in message
    get_settings.cache_clear()


def test_quoted_sleeper_user_id_still_resolves_owner(monkeypatch, db_session: Session):
    monkeypatch.setenv("SLEEPER_USER_ID", ' "739931264659927040" \n')
    get_settings.cache_clear()
    db_session.add(
        LeagueMember(
            league_id="league-1",
            user_id="739931264659927040",
            roster_id=6,
            display_name="Owner",
        )
    )
    db_session.flush()
    assert get_settings().sleeper_user_id == "739931264659927040"
    assert _resolve_owner_roster_id(db_session, "league-1", explicit_roster_id=None) == 6
    get_settings.cache_clear()


def test_numeric_sleeper_user_id_coerces_to_exact_string(monkeypatch, db_session: Session):
    # Constructor path covers JSON/env parsers that promote snowflakes to int.
    get_settings.cache_clear()
    from src.app.config import Settings

    settings = Settings(sleeper_user_id=739931264659927040)  # type: ignore[arg-type]
    assert settings.sleeper_user_id == "739931264659927040"
    db_session.add(
        LeagueMember(
            league_id="league-1",
            user_id="739931264659927040",
            roster_id=9,
            display_name="Owner",
        )
    )
    db_session.flush()
    monkeypatch.setenv("SLEEPER_USER_ID", "739931264659927040")
    get_settings.cache_clear()
    assert _resolve_owner_roster_id(db_session, "league-1", explicit_roster_id=None) == 9
    get_settings.cache_clear()




def test_username_fallback_resolves_owner_when_user_id_mismatches(monkeypatch, db_session: Session):
    """SLEEPER_USERNAME recovers when SLEEPER_USER_ID does not match members."""
    _USERNAME_TO_USER_ID_CACHE.clear()
    monkeypatch.setenv("SLEEPER_USER_ID", "stale-or-quoted-id")
    monkeypatch.setenv("SLEEPER_USERNAME", "rdfergus15")
    get_settings.cache_clear()
    _league(db_session)
    db_session.add(
        LeagueMember(
            league_id="league-1",
            user_id="739931264659927040",
            roster_id=4,
            display_name="Richard",
        )
    )
    db_session.flush()

    def _fake_get_user(self, username: str):
        assert username == "rdfergus15"
        return {"user_id": "739931264659927040", "username": "rdfergus15"}

    monkeypatch.setattr(
        "src.app.league.sleeper.client.SleeperClient.get_user",
        _fake_get_user,
        raising=True,
    )
    assert _resolve_owner_roster_id(db_session, "league-1", explicit_roster_id=None) == 4
    _USERNAME_TO_USER_ID_CACHE.clear()
    get_settings.cache_clear()


def test_list_leagues_hides_empty_historical_by_default(monkeypatch, db_session: Session):
    from src.app.api.v1.leagues import list_leagues

    monkeypatch.setenv("SLEEPER_USER_ID", "owner-6")
    get_settings.cache_clear()
    _league(db_session, "league-2026", season=2026)
    _league(db_session, "league-2025-empty", season=2025)
    db_session.add(
        LeagueMember(
            league_id="league-2026",
            user_id="owner-6",
            roster_id=6,
            display_name="Owner",
        )
    )
    db_session.flush()
    payload = list_leagues(user=AppUser(email="owner@example.com"), db=db_session)
    ids = {row["league_id"] for row in payload["leagues"]}
    assert "league-2026" in ids
    assert "league-2025-empty" not in ids
    assert payload["active_season"] == 2026
    assert payload["default_league_id"] == "league-2026"
    ready = next(row for row in payload["leagues"] if row["league_id"] == "league-2026")
    assert ready["decision_ready"] is True
    get_settings.cache_clear()


def test_list_leagues_exposes_owner_roster_id(monkeypatch, db_session: Session):
    from src.app.api.v1.leagues import list_leagues

    monkeypatch.setenv("SLEEPER_USER_ID", "owner-6")
    get_settings.cache_clear()
    _league(db_session, "league-owned")
    _league(db_session, "league-orphan")
    db_session.add(
        LeagueMember(
            league_id="league-owned",
            user_id="owner-6",
            roster_id=6,
            display_name="Owner",
        )
    )
    db_session.flush()
    payload = list_leagues(user=AppUser(email="owner@example.com"), db=db_session)
    by_id = {row["league_id"]: row["owner_roster_id"] for row in payload["leagues"]}
    assert by_id["league-owned"] == 6
    assert by_id["league-orphan"] is None
    get_settings.cache_clear()


def test_owner_with_multiple_memberships_fails_closed(monkeypatch, db_session: Session):
    monkeypatch.setenv("SLEEPER_USER_ID", "multi-owner")
    get_settings.cache_clear()
    db_session.add_all(
        [
            LeagueMember(league_id="league-1", user_id="multi-owner", roster_id=2, display_name="A"),
            LeagueMember(league_id="league-1", user_id="multi-owner", roster_id=5, display_name="B"),
        ]
    )
    db_session.flush()
    with pytest.raises(LeagueContextError, match="owner_roster_ambiguous"):
        _resolve_owner_roster_id(db_session, "league-1", explicit_roster_id=None)
    get_settings.cache_clear()


def test_sleeper_user_id_absent_fails_in_production(monkeypatch, db_session: Session):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("SLEEPER_USER_ID", raising=False)
    get_settings.cache_clear()
    with pytest.raises(LeagueContextError, match="owner_identity_unconfigured"):
        _resolve_owner_roster_id(db_session, "league-1", explicit_roster_id=None)
    get_settings.cache_clear()


def test_public_decision_error_codes_are_safe():
    code, message = _public_decision_error(
        LeagueContextError("no_projected_players_on_roster:league=x,week=1")
    )
    assert code == "identity_resolution_incomplete"
    assert "exception" not in message.lower()
    assert "league=x" not in message
    owner_code, owner_message = _public_decision_error(
        LeagueContextError("owner_roster_not_found:league=x")
    )
    assert owner_code == "owner_roster_unavailable"
    assert "league=x" not in owner_message


def test_gsis_link_from_release_players(db_session: Session):
    db_session.add(
        PlayerIdentity(
            player_id="6786",
            sleeper_id="6786",
            gsis_id=None,
            name="CeeDee Lamb",
            position="WR",
            team="DAL",
        )
    )
    db_session.flush()
    players = {
        "00-0036358": PlayerSummary(
            player_id="00-0036358",
            name="CeeDee Lamb",
            position="WR",
            team="DAL",
            mean_points=16.0,
            quantiles={},
            availability_probability=1.0,
        )
    }
    stats = link_identities_to_release_players(db_session, players)
    assert stats.linked == 1
    row = db_session.get(PlayerIdentity, "6786")
    assert row is not None
    assert row.gsis_id == "00-0036358"


def test_rosters_endpoint_dedupes_historical_rows(db_session: Session):
    now = datetime.now(UTC)
    db_session.add_all(
        [
            RosterSnapshot(
                league_id="league-1",
                week=1,
                roster_id=1,
                fetched_at=now - timedelta(days=1),
                players=["a"],
                starters=[],
                reserve=[],
            ),
            RosterSnapshot(
                league_id="league-1",
                week=1,
                roster_id=1,
                fetched_at=now,
                players=["b"],
                starters=[],
                reserve=[],
            ),
        ]
    )
    db_session.flush()
    payload = get_rosters("league-1", user=AppUser(email="owner@example.com"), db=db_session)
    assert len(payload["rosters"]) == 1
    assert payload["rosters"][0]["players"] == ["b"]


def test_trade_does_not_depend_on_matchup_snapshots(db_session: Session, monkeypatch):
    """Trade evaluation keys off request roster sides, not MatchupSnapshot rows."""
    from src.app.decisions.trades import TradeSide

    _league(db_session)
    now = datetime.now(UTC)
    db_session.add_all(
        [
            RosterSnapshot(
                league_id="league-1",
                week=1,
                roster_id=2,
                fetched_at=now,
                players=["00-a"],
                starters=["00-a"],
                reserve=[],
            ),
            RosterSnapshot(
                league_id="league-1",
                week=1,
                roster_id=3,
                fetched_at=now,
                players=["00-b"],
                starters=["00-b"],
                reserve=[],
            ),
            PlayerIdentity(player_id="00-a", sleeper_id="a", gsis_id="00-a", name="A", position="RB", team="DAL"),
            PlayerIdentity(player_id="00-b", sleeper_id="b", gsis_id="00-b", name="B", position="WR", team="DET"),
        ]
    )
    db_session.flush()

    class _Bundle:
        namespace = "test-bundle"
        meta: ClassVar[dict[str, str]] = {"scoring": "ppr"}

        def load_bundle(self):
            return self

        def get(self, pid):
            return PlayerSummary(
                player_id=pid,
                name=pid,
                position="RB" if pid.endswith("a") else "WR",
                team="DAL",
                mean_points=10.0,
                quantiles={"p10": 5, "p50": 10, "p90": 15},
                availability_probability=1.0,
            )

        def load(self):
            return {pid: self.get(pid) for pid in ("00-a", "00-b")}

        def as_of(self):
            return now.isoformat()

    monkeypatch.setattr("src.app.decisions.services.get_bundle_loader", lambda season: _Bundle())
    result = TradeService(db_session).evaluate(
        "league-1",
        TradeSide(roster_id=2, player_ids=["00-a"]),
        TradeSide(roster_id=3, player_ids=["00-b"]),
        horizon="ros",
        week=1,
    )
    assert result is not None
    # No MatchupSnapshot rows exist for this league/week; trade still evaluates.
