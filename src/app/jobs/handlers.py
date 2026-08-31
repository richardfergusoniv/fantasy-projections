"""Worker job implementations."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from src.app.availability.research_job import research_changed_players
from src.app.config import get_settings
from src.app.decisions.tendencies import ManagerTendencyService
from src.app.jobs.schedule_guards import weekly_close_allowed
from src.app.league.sleeper.sync import SleeperSyncService
from src.app.persistence.models import (
    ActiveProjectionPointer,
    AvailabilityEvent,
    League,
    PlayerProjection,
    ProjectionRun,
    PromotionEvent,
    RosterSnapshot,
)
from src.app.projections.loader import ReleaseBundleLoader
from src.app.projections.weekly_run import (
    WeeklyProjectionService,
    current_weekly_model_version,
)
from src.app.releases.bridge import ReleaseBridge
from src.app.releases.gates import scoring_contract_fingerprint, validate_scoring_contracts
from src.app.releases.incremental import IncrementalSimulationService, build_impact_set

#: Used only when neither the NFL state feed nor the database can say what the
#: current season/week is. Always reported back in the job metadata as
#: ``season_week_source="fallback_default"`` so it is never silently assumed.
FALLBACK_SEASON = 2026
FALLBACK_WEEK = 1


@dataclass(frozen=True)
class SeasonWeek:
    season: int
    week: int
    source: str

    def to_dict(self) -> dict:
        return {"season": self.season, "week": self.week, "source": self.source}


def resolve_season_week(session: Session, *, nfl_state: dict | None = None) -> SeasonWeek:
    """Derive the operating season/week from live state, then the database."""
    if nfl_state:
        season = nfl_state.get("season")
        week = nfl_state.get("display_week") or nfl_state.get("week")
        if season and week:
            return SeasonWeek(int(season), int(week), "nfl_state")

    pointer = (
        session.query(ActiveProjectionPointer)
        .filter(
            ActiveProjectionPointer.mode == "weekly",
            ActiveProjectionPointer.week.isnot(None),
        )
        .order_by(
            ActiveProjectionPointer.season.desc(),
            ActiveProjectionPointer.week.desc(),
        )
        .first()
    )
    if pointer is not None:
        return SeasonWeek(int(pointer.season), int(pointer.week), "active_pointer")

    league = session.query(League).order_by(League.season.desc()).first()
    if league is not None:
        roster = (
            session.query(RosterSnapshot)
            .filter(RosterSnapshot.league_id == league.league_id)
            .order_by(RosterSnapshot.week.desc())
            .first()
        )
        week = int(roster.week) if roster is not None else FALLBACK_WEEK
        return SeasonWeek(int(league.season), week, "league_state")

    return SeasonWeek(FALLBACK_SEASON, FALLBACK_WEEK, "fallback_default")


def _nfl_state(sync: SleeperSyncService) -> dict | None:
    try:
        return sync.client.get_nfl_state()
    except Exception:  # noqa: BLE001 — state feed is advisory, never fatal
        return None


def _league_dependencies(session: Session, week: int) -> dict[str, set[str]]:
    """League rosters that consume projections, so decisions widen with players."""
    dependencies: dict[str, set[str]] = {}
    for snapshot in session.query(RosterSnapshot).filter(RosterSnapshot.week == week).all():
        bucket = dependencies.setdefault(snapshot.league_id, set())
        bucket.update(str(player) for player in (snapshot.players or []))
    return dependencies


def _opponent_map(session: Session, run_id: str) -> dict[str, str] | None:
    rows = (
        session.query(PlayerProjection.team, PlayerProjection.opponent)
        .filter(
            PlayerProjection.run_id == run_id,
            PlayerProjection.opponent.isnot(None),
        )
        .distinct()
        .all()
    )
    mapping = {team: opponent for team, opponent in rows if team and opponent}
    return mapping or None


def _promotion_metadata(session: Session, run_id: str) -> dict:
    event = (
        session.query(PromotionEvent)
        .filter(PromotionEvent.candidate_run_id == run_id, PromotionEvent.promoted.is_(True))
        .order_by(PromotionEvent.created_at.desc())
        .first()
    )
    return dict(event.validation_json or {}) if event is not None else {}


def _maybe_incremental_weekly(
    session: Session,
    season: int,
    week: int,
    weekly_run_id: str | None,
    *,
    automatic: bool,
    weekly_context_changed: bool = False,
) -> tuple[str | None, dict]:
    """Refresh the weekly run for changed availability, widening when unsure."""
    if weekly_run_id is None:
        return None, {"mode": "no_base_run"}
    changed = {
        event.player_id
        for event in session.query(AvailabilityEvent)
        .filter(AvailabilityEvent.cleared_at.is_(None))
        .all()
    }
    if not changed:
        return weekly_run_id, {"mode": "no_changes"}
    players = ReleaseBundleLoader(season=season).load()
    if not players:
        return weekly_run_id, {"mode": "no_release_bundle"}

    base_run = session.query(ProjectionRun).filter(ProjectionRun.id == weekly_run_id).one_or_none()
    baseline_meta = _promotion_metadata(session, weekly_run_id)
    league_ids = [row.league_id for row in session.query(League).all()]
    impact = build_impact_set(
        changed,
        players,
        opponent_map=_opponent_map(session, weekly_run_id),
        league_dependencies=_league_dependencies(session, week),
        weekly_context_changed=weekly_context_changed,
        scoring_contract_hash=scoring_contract_fingerprint(session, league_ids),
        baseline_scoring_contract_hash=baseline_meta.get("scoring_contract_fingerprint"),
        model_version=current_weekly_model_version(season, week),
        baseline_model_version=base_run.model_version if base_run is not None else None,
    )
    if impact.requires_full_refresh:
        refreshed = WeeklyProjectionService(session).promote_week(
            season, week, automatic=automatic, league_ids=league_ids
        )
        return refreshed or weekly_run_id, {
            "mode": "full_refresh",
            **impact.to_dict(),
        }
    incremental_run_id = IncrementalSimulationService(session).promote_affected_week(
        season,
        week,
        impact,
        base_run_id=weekly_run_id,
        automatic=automatic,
    )
    return incremental_run_id or weekly_run_id, {
        "mode": "incremental" if incremental_run_id else "unchanged",
        **impact.to_dict(),
    }


def run_daily_refresh(session: Session, *, automatic: bool = True) -> dict:
    settings = get_settings()
    sync = SleeperSyncService(session, use_fixtures=settings.use_sleeper_fixtures)
    user_id = settings.sleeper_user_id or "fixture-user-1"
    season_week = resolve_season_week(session, nfl_state=_nfl_state(sync))
    availability = sync.sync_player_availability()
    research = research_changed_players(session)
    leagues = sync.sync_leagues(user_id, season=season_week.season)
    bridge = ReleaseBridge(session)
    preseason_run_id = bridge.sync_preseason_pointer(season_week.season, automatic=automatic)

    # The scoring-contract gate runs *before* promotion (it used to run after,
    # where it could observe a failure but not prevent it) and is also handed to
    # the publication pipeline as a blocking gate.
    scoring_gate = validate_scoring_contracts(session, leagues)
    weekly = WeeklyProjectionService(session)
    weekly_run_id = weekly.promote_week(
        season_week.season,
        season_week.week,
        automatic=automatic,
        league_ids=leagues,
    )
    weekly_run_id, incremental = _maybe_incremental_weekly(
        session,
        season_week.season,
        season_week.week,
        weekly_run_id,
        automatic=automatic,
    )
    for league_id in leagues:
        ManagerTendencyService(session).rebuild(league_id)
    return {
        "leagues_synced": len(leagues),
        "sleeper_source": settings.sleeper_mode,
        # Surfaced so an operator can see a Sleeper roster id that no longer maps
        # to a known player instead of silently losing that player's projection.
        "unresolved_player_ids": sorted(sync.unresolved_player_ids),
        "availability": availability,
        "injury_research": research,
        "preseason_run_id": preseason_run_id,
        "weekly_run_id": weekly_run_id,
        "scoring_gate": scoring_gate.to_dict(),
        "incremental": incremental,
        "season": season_week.season,
        "week": season_week.week,
        "season_week_source": season_week.source,
        "automatic": automatic,
    }


def run_full_release(session: Session, *, automatic: bool = True) -> dict:
    settings = get_settings()
    sync = SleeperSyncService(session, use_fixtures=settings.use_sleeper_fixtures)
    nfl_state = _nfl_state(sync) or {}
    season_week = resolve_season_week(session, nfl_state=nfl_state)
    allowed, reason = weekly_close_allowed(nfl_state)
    if not allowed:
        return {
            "status": "postponed",
            "reason": reason,
            "season": season_week.season,
            "week": season_week.week,
            "season_week_source": season_week.source,
        }
    availability = sync.sync_player_availability()
    research = research_changed_players(session)
    bridge = ReleaseBridge(session)
    preseason_run_id = bridge.sync_preseason_pointer(season_week.season, automatic=automatic)
    league_ids = [row.league_id for row in session.query(League).all()]
    scoring_gate = validate_scoring_contracts(session, league_ids)
    weekly = WeeklyProjectionService(session)
    weekly_run_id = weekly.promote_week(
        season_week.season,
        season_week.week,
        automatic=automatic,
        league_ids=league_ids,
    )
    weekly_run_id, incremental = _maybe_incremental_weekly(
        session,
        season_week.season,
        season_week.week,
        weekly_run_id,
        automatic=automatic,
    )
    ros_run_id = weekly.promote_ros(
        season_week.season, from_week=season_week.week, automatic=automatic, league_ids=league_ids
    )
    dynasty_run_id = weekly.promote_dynasty(
        season_week.season, automatic=automatic, league_ids=league_ids
    )
    return {
        "status": "promoted" if weekly_run_id else "unchanged",
        "availability": availability,
        "injury_research": research,
        "preseason_run_id": preseason_run_id,
        "weekly_run_id": weekly_run_id,
        "ros_run_id": ros_run_id,
        "dynasty_run_id": dynasty_run_id,
        "scoring_gate": scoring_gate.to_dict(),
        "incremental": incremental,
        "season": season_week.season,
        "week": season_week.week,
        "season_week_source": season_week.source,
        "automatic": automatic,
    }


JOB_HANDLERS = {
    "daily-refresh": run_daily_refresh,
    "sunday-early": run_daily_refresh,
    "sunday-afternoon": run_daily_refresh,
    "sunday-night": run_daily_refresh,
    "monday-night": run_daily_refresh,
    "weekly-close-preliminary": run_full_release,
    "weekly-correction": run_full_release,
    "full-release": run_full_release,
}
