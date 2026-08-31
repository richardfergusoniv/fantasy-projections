"""Repository layer — persistence without HTTP coupling."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from src.app.persistence.models import (
    ActiveProjectionPointer,
    AvailabilityEvent,
    InjuryEvidence,
    League,
    LeagueMember,
    LeagueRuleSnapshot,
    LeagueTransaction,
    MatchupSnapshot,
    PlayerIdentity,
    PlayerProjection,
    ProjectionRun,
    RosterSnapshot,
    SourceSnapshot,
    TradedPick,
)


class LeagueRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_leagues(self) -> list[League]:
        return self.session.query(League).order_by(League.name).all()

    def get_league(self, league_id: str) -> League | None:
        return self.session.query(League).filter(League.league_id == league_id).one_or_none()

    def upsert_league(self, *, league_id: str, season: int, name: str, league_type: str, raw_json: dict) -> League:
        row = (
            self.session.query(League)
            .filter(League.league_id == league_id, League.season == season)
            .one_or_none()
        )
        if row is None:
            row = League(
                league_id=league_id,
                season=season,
                name=name,
                league_type=league_type,
                raw_json=raw_json,
            )
            self.session.add(row)
        else:
            row.name = name
            row.league_type = league_type
            row.raw_json = raw_json
        self.session.flush()
        return row

    def latest_rules(self, league_id: str) -> LeagueRuleSnapshot | None:
        return (
            self.session.query(LeagueRuleSnapshot)
            .filter(LeagueRuleSnapshot.league_id == league_id)
            .order_by(LeagueRuleSnapshot.fetched_at.desc())
            .first()
        )

    def add_rule_snapshot(self, snapshot: LeagueRuleSnapshot) -> LeagueRuleSnapshot:
        self.session.add(snapshot)
        self.session.flush()
        return snapshot

    def latest_rosters(self, league_id: str, week: int) -> list[RosterSnapshot]:
        return (
            self.session.query(RosterSnapshot)
            .filter(RosterSnapshot.league_id == league_id, RosterSnapshot.week == week)
            .order_by(RosterSnapshot.fetched_at.desc())
            .all()
        )

    def add_roster_snapshot(self, snapshot: RosterSnapshot) -> RosterSnapshot:
        self.session.add(snapshot)
        self.session.flush()
        return snapshot

    def upsert_member(self, *, league_id: str, user_id: str, roster_id: int, display_name: str) -> LeagueMember:
        row = (
            self.session.query(LeagueMember)
            .filter(LeagueMember.league_id == league_id, LeagueMember.roster_id == roster_id)
            .one_or_none()
        )
        if row is None:
            row = LeagueMember(
                league_id=league_id,
                user_id=user_id,
                roster_id=roster_id,
                display_name=display_name,
            )
            self.session.add(row)
        else:
            row.user_id = user_id
            row.display_name = display_name
        self.session.flush()
        return row


class ProjectionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def active_run(self, *, mode: str, season: int, week: int | None = None) -> ProjectionRun | None:
        pointer = (
            self.session.query(ActiveProjectionPointer)
            .filter(
                ActiveProjectionPointer.mode == mode,
                ActiveProjectionPointer.season == season,
                ActiveProjectionPointer.week == week,
            )
            .one_or_none()
        )
        if pointer is None:
            return None
        return self.session.query(ProjectionRun).filter(ProjectionRun.id == pointer.run_id).one_or_none()

    def player_projections(self, run_id: str, player_ids: list[str] | None = None) -> list[PlayerProjection]:
        query = self.session.query(PlayerProjection).filter(PlayerProjection.run_id == run_id)
        if player_ids:
            query = query.filter(PlayerProjection.player_id.in_(player_ids))
        return query.all()


class AvailabilityRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def active_events(self, player_id: str | None = None) -> list[AvailabilityEvent]:
        query = self.session.query(AvailabilityEvent).filter(AvailabilityEvent.cleared_at.is_(None))
        if player_id:
            query = query.filter(AvailabilityEvent.player_id == player_id)
        return query.all()

    def add_event(self, event: AvailabilityEvent) -> AvailabilityEvent:
        self.session.add(event)
        self.session.flush()
        return event

    def clear_event(self, event_id: str, *, cleared_at: datetime | None = None) -> AvailabilityEvent:
        event = self.session.query(AvailabilityEvent).filter(AvailabilityEvent.id == event_id).one()
        event.cleared_at = cleared_at or datetime.now(UTC)
        self.session.flush()
        return event

    def evidence_for_player(self, player_id: str) -> list[InjuryEvidence]:
        return (
            self.session.query(InjuryEvidence)
            .filter(InjuryEvidence.player_id == player_id)
            .order_by(InjuryEvidence.fetched_at.desc())
            .all()
        )


class SourceRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_snapshot(self, snapshot: SourceSnapshot) -> SourceSnapshot:
        self.session.add(snapshot)
        self.session.flush()
        return snapshot

    def latest_for_endpoint(self, endpoint: str) -> SourceSnapshot | None:
        return (
            self.session.query(SourceSnapshot)
            .filter(SourceSnapshot.endpoint == endpoint)
            .order_by(SourceSnapshot.fetched_at.desc())
            .first()
        )
