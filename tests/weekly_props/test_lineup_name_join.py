"""MVP: lineup consumes weekly_props via Sleeper name join (no sealed required)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

import pytest
from sqlalchemy.orm import Session

from src.app.config import get_settings
from src.app.decisions.services import LineupService
from src.app.persistence.models import (
    ActiveProjectionPointer,
    League,
    LeagueMember,
    LeagueRuleSnapshot,
    MatchupSnapshot,
    PlayerIdentity,
    PlayerProjection,
    ProjectionRun,
    RosterSnapshot,
)


def _seed_props_lineup_league(session: Session) -> None:
    now = datetime.now(UTC)
    session.add(
        League(
            league_id="props-league",
            season=2026,
            name="Props League",
            league_type="redraft",
            raw_json={"roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "BN"]},
        )
    )
    session.add(
        LeagueRuleSnapshot(
            league_id="props-league",
            fetched_at=now,
            raw_json={
                "pass_yd": 0.04,
                "pass_td": 4,
                "rush_yd": 0.1,
                "rush_td": 6,
                "rec": 1,
                "rec_yd": 0.1,
                "rec_td": 6,
            },
            normalized_json={},
            contract_hash="props-contract",
        )
    )
    session.add(
        LeagueMember(
            league_id="props-league",
            user_id="owner-1",
            roster_id=1,
            display_name="Owner",
        )
    )
    # Sleeper roster ids only — no GSIS on identity (product: names from Sleeper).
    session.add_all(
        [
            PlayerIdentity(
                player_id="sleeper-qb",
                sleeper_id="sleeper-qb",
                gsis_id=None,
                name="Josh Allen",
                position="QB",
                team="BUF",
            ),
            PlayerIdentity(
                player_id="sleeper-rb",
                sleeper_id="sleeper-rb",
                gsis_id=None,
                name="Saquon Barkley",
                position="RB",
                team="PHI",
            ),
            PlayerIdentity(
                player_id="sleeper-wr",
                sleeper_id="sleeper-wr",
                gsis_id=None,
                name="A.J. Brown",
                position="WR",
                team="PHI",
            ),
            PlayerIdentity(
                player_id="sleeper-te",
                sleeper_id="sleeper-te",
                gsis_id=None,
                name="Travis Kelce",
                position="TE",
                team="KC",
            ),
            PlayerIdentity(
                player_id="sleeper-wr2",
                sleeper_id="sleeper-wr2",
                gsis_id=None,
                name="CeeDee Lamb",
                position="WR",
                team="DAL",
            ),
        ]
    )
    session.add(
        RosterSnapshot(
            league_id="props-league",
            week=2,
            roster_id=1,
            fetched_at=now,
            players=[
                "sleeper-qb",
                "sleeper-rb",
                "sleeper-wr",
                "sleeper-te",
                "sleeper-wr2",
            ],
            starters=[
                "sleeper-qb",
                "sleeper-rb",
                "sleeper-wr",
                "sleeper-te",
                "sleeper-wr2",
            ],
            reserve=[],
        )
    )
    session.add(
        MatchupSnapshot(
            league_id="props-league",
            week=2,
            roster_id=1,
            matchup_id=1,
            fetched_at=now,
            points=0,
        )
    )

    run = ProjectionRun(
        id="weekly-props-2026-w02-testmvp",
        mode="weekly",
        season=2026,
        week=2,
        as_of=now,
        model_version="weekly_props_v1",
        input_hash="mvp-hash",
        status="active",
        manifest_uri="fixture://weekly-props-mvp",
        artifact_mode="market",
    )
    session.add(run)
    session.flush()
    session.add(
        ActiveProjectionPointer(
            mode="weekly", season=2026, week=2, run_id=run.id
        )
    )
    # Props run keyed by market/GSIS-like ids; join is by name to Sleeper.
    for pid, name, pos, team, points in (
        ("00-allen", "Josh Allen", "QB", "BUF", 22.4),
        ("00-barkley", "Saquon Barkley", "RB", "PHI", 18.1),
        ("00-brown", "A.J. Brown", "WR", "PHI", 14.6),
        ("00-kelce", "Travis Kelce", "TE", "KC", 11.2),
        ("00-lamb", "CeeDee Lamb", "WR", "DAL", 15.3),
    ):
        session.add(
            PlayerProjection(
                run_id=run.id,
                player_id=pid,
                team=team,
                opponent=None,
                availability_probability=1.0,
                mean_json={
                    "name": name,
                    "position": pos,
                    "team": team,
                    "points": points,
                    "component_sources": {"points": "weekly_props_consensus"},
                },
                quantiles_json={"0.1": points * 0.7, "0.5": points, "0.9": points * 1.3},
            )
        )
    session.flush()


def test_lineup_uses_weekly_props_by_sleeper_name(db_session: Session, monkeypatch):
    monkeypatch.setenv("APP_PROJECTION_SOURCE", "weekly_props")
    monkeypatch.setenv("SLEEPER_USER_ID", "owner-1")
    get_settings.cache_clear()
    _seed_props_lineup_league(db_session)

    class _ColdBundle:
        """Sealed release deliberately unavailable — Vegas path must not need it."""

        namespace = "cold"
        meta: ClassVar[dict] = {}

        def load_bundle(self):
            return None

        def get(self, pid):
            return None

        def load(self):
            return {}

        def as_of(self):
            return datetime.now(UTC).isoformat()

    monkeypatch.setattr(
        "src.app.decisions.services.get_bundle_loader",
        lambda season: _ColdBundle(),
    )
    monkeypatch.setattr(
        "src.app.projections.service.ProjectionService.matchup_win_probability_allowed",
        lambda self, **kwargs: False,
    )

    result = LineupService(db_session).recommend("props-league", 2, opponent_mode="current")
    assert result["board_source"] == "vegas_props"
    assert result["effective_source"] == "weekly_props"
    assert result["projection_run_id"] == "weekly-props-2026-w02-testmvp"
    assert result["expected_points"] > 0
    names = {row["name"] for row in result["starters"]}
    assert "Josh Allen" in names
    assert "Saquon Barkley" in names
    points_by_name = {row["name"]: row["expected_points"] for row in result["starters"]}
    assert points_by_name["Josh Allen"] == pytest.approx(22.4, abs=0.5)
    get_settings.cache_clear()


def test_board_source_league_value_when_sealed(db_session: Session, monkeypatch):
    monkeypatch.setenv("APP_PROJECTION_SOURCE", "sealed_release")
    monkeypatch.setenv("SLEEPER_USER_ID", "owner-1")
    get_settings.cache_clear()
    from src.app.decisions.services import _LeagueContext

    _seed_props_lineup_league(db_session)

    class _ColdBundle:
        namespace = "cold"
        meta: ClassVar[dict] = {"scoring": "ppr"}

        def load_bundle(self):
            return self

        def get(self, pid):
            return None

        def load(self):
            return {}

        def as_of(self):
            return datetime.now(UTC).isoformat()

    monkeypatch.setattr(
        "src.app.decisions.services.get_bundle_loader",
        lambda season: _ColdBundle(),
    )
    # Sealed path without projections will fail on empty roster draws; only
    # assert the board_source seam on a constructed context.
    ctx = _LeagueContext(db_session, "props-league", 2)
    assert ctx.board_source() == "league_value"
    get_settings.cache_clear()
