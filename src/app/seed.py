"""Fixture-backed development seed for six representative leagues."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from src.app.persistence.models import (
    AppUser,
    League,
    LeagueDraftRule,
    LeagueMember,
    LeagueRuleSnapshot,
    PlayerIdentity,
    RosterSnapshot,
)
from src.app.scoring.compiler import compile_sleeper_scoring

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "seed"


def _load(name: str) -> dict:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _upsert_roster_snapshot(
    session: Session, *, league_id: str, week: int, roster: dict
) -> None:
    """Idempotent roster snapshot so re-seeding does not duplicate rows."""
    existing = (
        session.query(RosterSnapshot)
        .filter(
            RosterSnapshot.league_id == league_id,
            RosterSnapshot.week == week,
            RosterSnapshot.roster_id == roster["roster_id"],
        )
        .order_by(RosterSnapshot.fetched_at.desc())
        .first()
    )
    if existing is None:
        session.add(
            RosterSnapshot(
                league_id=league_id,
                week=week,
                roster_id=roster["roster_id"],
                fetched_at=datetime.now(UTC),
                players=roster.get("players", []),
                starters=roster.get("starters", []),
                reserve=roster.get("reserve", []),
            )
        )
        return
    existing.players = roster.get("players", [])
    existing.starters = roster.get("starters", [])
    existing.reserve = roster.get("reserve", [])
    existing.fetched_at = datetime.now(UTC)


def seed_development_data(session: Session, *, email: str) -> dict:
    user = session.query(AppUser).filter(AppUser.email == email).one_or_none()
    if user is None:
        user = AppUser(email=email)
        session.add(user)
        session.flush()

    manifest = _load("leagues_manifest.json")
    seeded = []
    for league_fixture in manifest["leagues"]:
        league_data = _load(league_fixture["league_file"])
        existing = (
            session.query(League)
            .filter(League.league_id == league_data["league_id"], League.season == league_data["season"])
            .one_or_none()
        )
        if existing is None:
            existing = League(
                league_id=league_data["league_id"],
                season=league_data["season"],
                name=league_data["name"],
                league_type=league_data["type"],
                status="active",
                raw_json=league_data,
            )
            session.add(existing)
        else:
            existing.name = league_data["name"]
            existing.league_type = league_data["type"]
            existing.raw_json = league_data
        contract = compile_sleeper_scoring(
            league_data["scoring_settings"],
            league_data.get("roster_positions", []),
        )
        snapshot = LeagueRuleSnapshot(
            league_id=league_data["league_id"],
            fetched_at=datetime.now(UTC),
            raw_json=league_data["scoring_settings"],
            normalized_json=contract.to_dict(),
            contract_hash=contract.contract_hash,
        )
        session.add(snapshot)
        # Rookie-pick order is a dynasty concept. Redraft leagues intentionally
        # get no draft-order rule, and re-seeding updates rather than appends.
        draft_rule = league_fixture.get("draft_order_rule")
        if draft_rule:
            existing_rule = (
                session.query(LeagueDraftRule)
                .filter(LeagueDraftRule.league_id == league_data["league_id"])
                .order_by(LeagueDraftRule.confirmed_at.desc())
                .first()
            )
            if existing_rule is None:
                session.add(
                    LeagueDraftRule(
                        league_id=league_data["league_id"],
                        rule=draft_rule,
                    )
                )
            else:
                existing_rule.rule = draft_rule
        for member in league_fixture.get("members", []):
            existing_member = (
                session.query(LeagueMember)
                .filter(
                    LeagueMember.league_id == league_data["league_id"],
                    LeagueMember.roster_id == member["roster_id"],
                )
                .one_or_none()
            )
            if existing_member is None:
                session.add(
                    LeagueMember(
                        league_id=league_data["league_id"],
                        user_id=member["user_id"],
                        roster_id=member["roster_id"],
                        display_name=member["display_name"],
                    )
                )
            else:
                existing_member.user_id = member["user_id"]
                existing_member.display_name = member["display_name"]
        roster_files = [league_fixture["roster_file"]]
        if "opponent_roster_file" in league_fixture:
            roster_files.append(league_fixture["opponent_roster_file"])
        for roster_file in roster_files:
            roster = _load(roster_file)
            _upsert_roster_snapshot(
                session,
                league_id=league_data["league_id"],
                week=manifest.get("week", 1),
                roster=roster,
            )
        seeded.append(league_data["league_id"])

    players = _load("players.json")
    for player in players:
        session.merge(
            PlayerIdentity(
                player_id=player["player_id"],
                sleeper_id=player.get("sleeper_id"),
                gsis_id=player.get("gsis_id"),
                name=player["name"],
                position=player["position"],
                team=player.get("team"),
            )
        )

    from src.app.releases.bridge import ReleaseBridge
    from src.app.projections.weekly_run import WeeklyProjectionService
    from src.app.decisions.tendencies import ManagerTendencyService

    bridge = ReleaseBridge(session)
    preseason_run_id = bridge.sync_preseason_pointer(manifest.get("season", 2026))
    weekly_service = WeeklyProjectionService(session)
    weekly_run_id = weekly_service.promote_week(manifest.get("season", 2026), manifest.get("week", 1))
    ros_run_id = weekly_service.promote_ros(manifest.get("season", 2026), from_week=manifest.get("week", 1))
    dynasty_run_id = weekly_service.promote_dynasty(manifest.get("season", 2026))
    ManagerTendencyService(session).rebuild(seeded[0] if seeded else "fixture-standard")
    session.flush()
    return {
        "user_id": user.id,
        "leagues": seeded,
        "preseason_run_id": preseason_run_id,
        "weekly_run_id": weekly_run_id,
        "ros_run_id": ros_run_id,
        "dynasty_run_id": dynasty_run_id,
        "projection_run_id": weekly_run_id or preseason_run_id,
    }
