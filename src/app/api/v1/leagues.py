"""League API endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.app.api.deps import (
    get_current_user,
    get_db,
    require_csrf,
    require_idempotency_key,
)
from src.app.config import get_settings
from src.app.decisions.draft_board import DraftBoardService
from src.app.decisions.services import LineupService, WaiverService
from src.app.jobs.handlers import run_daily_refresh
from src.app.jobs.runner import JobRunner
from src.app.league.sleeper.owner_config import load_owner_config
from src.app.league.sleeper.sync import SleeperSyncService
from src.app.logging import get_logger
from src.app.persistence.models import (
    AppUser,
    League,
    LeagueDraftRule,
    LeagueMember,
    LeagueRuleSnapshot,
    MatchupSnapshot,
    RosterSnapshot,
)
from src.app.persistence.repositories import ProjectionRepository
from src.app.projections.loader import get_bundle_loader
from src.app.projections.source import (
    ProjectionSource,
    configured_projection_source,
    weekly_rnd_enabled,
)

logger = get_logger(__name__)

router = APIRouter()


def _configured_league_ids() -> frozenset[str]:
    """Owner-configured league ids when owner config is set."""

    settings = get_settings()
    if not settings.sleeper_owner_config and not settings.sleeper_owner_json:
        return frozenset()
    try:
        return load_owner_config(settings.sleeper_owner_config).allowed_league_ids
    except (FileNotFoundError, OSError, ValueError):
        return frozenset()


def _unprocessable(code: str, message: str, *, league_id: str, exc: Exception) -> HTTPException:
    """Log the underlying reason server-side and return a safe client detail.

    Service ValueErrors carry snapshot ids, roster internals and file paths;
    those belong in logs, not in an HTTP body.
    """
    logger.warning(
        "decision_request_rejected",
        code=code,
        league_id=league_id,
        exception_type=type(exc).__name__,
        reason=str(exc),
    )
    return HTTPException(status_code=400, detail={"code": code, "message": message})


def _meta(session: Session, league_id: str, *, week: int = 1) -> dict:
    snapshot = (
        session.query(LeagueRuleSnapshot)
        .filter(LeagueRuleSnapshot.league_id == league_id)
        .order_by(LeagueRuleSnapshot.fetched_at.desc())
        .first()
    )
    league = session.query(League).filter(League.league_id == league_id).one_or_none()
    season = league.season if league else 2026
    source = configured_projection_source()
    projections = ProjectionRepository(session)
    run = None
    if source == ProjectionSource.WEEKLY_V2_RND and weekly_rnd_enabled():
        run = projections.active_run(mode="weekly", season=season, week=week)
    if run is None:
        run = projections.active_run(mode="preseason", season=season, week=None)
    projection_run_id = "fixture"
    if source in {ProjectionSource.SEALED_RELEASE, ProjectionSource.STATUS_ADJUSTED_RELEASE}:
        bundle = get_bundle_loader(season).load_bundle()
        if bundle is not None:
            projection_run_id = f"preseason-{bundle.namespace}"
        elif run is not None:
            projection_run_id = run.id
    elif run is not None:
        projection_run_id = run.id
    return {
        "data_as_of": (snapshot.fetched_at if snapshot else datetime.now(UTC)).isoformat(),
        "rule_snapshot_id": snapshot.id if snapshot else None,
        "projection_run_id": projection_run_id,
        "projection_source": source.value,
    }


@router.get("/leagues")
def list_leagues(user: AppUser = Depends(get_current_user), db: Session = Depends(get_db)):
    leagues = db.query(League).all()
    configured_ids = _configured_league_ids()
    payload = []
    for league in leagues:
        snapshot = (
            db.query(LeagueRuleSnapshot)
            .filter(LeagueRuleSnapshot.league_id == league.league_id)
            .order_by(LeagueRuleSnapshot.fetched_at.desc())
            .first()
        )
        scoring_type = "custom"
        if snapshot and snapshot.normalized_json:
            scoring_type = snapshot.normalized_json.get("scoring_type", scoring_type)
        payload.append(
            {
                "id": league.league_id,
                "league_id": league.league_id,
                "name": league.name,
                "type": league.league_type,
                "season": league.season,
                "scoring_type": scoring_type,
                "is_dynasty": league.league_type == "dynasty",
                "is_configured": (
                    league.league_id in configured_ids if configured_ids else True
                ),
                "roster_positions": league.raw_json.get("roster_positions", []) if league.raw_json else [],
            }
        )
    meta = _meta(db, leagues[0].league_id if leagues else "")
    if configured_ids:
        meta["configured_league_ids"] = sorted(configured_ids)
    return {"leagues": payload, **meta}


@router.get("/leagues/{league_id}")
def get_league(league_id: str, user: AppUser = Depends(get_current_user), db: Session = Depends(get_db)):
    league = db.query(League).filter(League.league_id == league_id).one_or_none()
    if league is None:
        raise HTTPException(status_code=404, detail="League not found")
    return {"league": {"league_id": league.league_id, "name": league.name, "type": league.league_type, "season": league.season, "raw": league.raw_json}, **_meta(db, league_id)}


@router.get("/leagues/{league_id}/rules")
def get_rules(league_id: str, user: AppUser = Depends(get_current_user), db: Session = Depends(get_db)):
    snapshot = (
        db.query(LeagueRuleSnapshot)
        .filter(LeagueRuleSnapshot.league_id == league_id)
        .order_by(LeagueRuleSnapshot.fetched_at.desc())
        .first()
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Rules not found")
    return {"rules": snapshot.normalized_json, "contract_hash": snapshot.contract_hash, **_meta(db, league_id)}


class DraftOrderRuleUpdate(BaseModel):
    rule: str


@router.put("/leagues/{league_id}/draft-order-rule")
def update_draft_order_rule(
    league_id: str,
    payload: DraftOrderRuleUpdate,
    user: AppUser = Depends(require_csrf),
    db: Session = Depends(get_db),
    idempotency_key: str = Depends(require_idempotency_key),
):
    record = LeagueDraftRule(league_id=league_id, rule=payload.rule)
    db.add(record)
    db.commit()
    return {"status": "ok", "rule": payload.rule, **_meta(db, league_id)}


@router.get("/leagues/{league_id}/rosters")
def get_rosters(league_id: str, user: AppUser = Depends(get_current_user), db: Session = Depends(get_db)):
    from src.app.persistence.models import PlayerIdentity

    rosters = db.query(RosterSnapshot).filter(RosterSnapshot.league_id == league_id).all()
    members = {
        row.roster_id: row.display_name
        for row in db.query(LeagueMember).filter(LeagueMember.league_id == league_id).all()
    }
    identities: dict[str, PlayerIdentity] = {}
    for row in db.query(PlayerIdentity).all():
        identities[row.player_id] = row
        if row.sleeper_id:
            identities.setdefault(row.sleeper_id, row)

    def player_label(player_id: str) -> dict:
        row = identities.get(str(player_id))
        if row is None:
            return {"player_id": str(player_id), "name": str(player_id)}
        return {
            "player_id": row.player_id,
            "name": row.name,
            "position": row.position,
            "team": row.team,
        }

    return {
        "rosters": [
            {
                "roster_id": r.roster_id,
                "week": r.week,
                "players": r.players,
                "starters": r.starters,
                "reserve": r.reserve,
                "manager_name": members.get(r.roster_id),
                "player_details": [player_label(pid) for pid in (r.players or []) if pid],
            }
            for r in rosters
        ],
        "managers": {str(roster_id): name for roster_id, name in members.items()},
        **_meta(db, league_id),
    }


@router.get("/leagues/{league_id}/matchups/{week}")
def get_matchups(league_id: str, week: int, user: AppUser = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (
        db.query(MatchupSnapshot)
        .filter(MatchupSnapshot.league_id == league_id, MatchupSnapshot.week == week)
        .order_by(MatchupSnapshot.matchup_id, MatchupSnapshot.roster_id)
        .all()
    )
    return {
        "week": week,
        "matchups": [
            {
                "roster_id": row.roster_id,
                "matchup_id": row.matchup_id,
                "points": row.points,
            }
            for row in rows
        ],
        **_meta(db, league_id, week=week),
    }


@router.get("/leagues/{league_id}/lineup/{week}")
def recommend_lineup(
    league_id: str,
    week: int,
    opponent_mode: str = "current",
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        result = LineupService(db).recommend(league_id, week, opponent_mode=opponent_mode)
    except ValueError as exc:
        raise _unprocessable(
            "lineup_unavailable",
            "Lineup recommendation is unavailable for this league and week.",
            league_id=league_id,
            exc=exc,
        ) from exc
    snapshot = (
        db.query(LeagueRuleSnapshot)
        .filter(LeagueRuleSnapshot.league_id == league_id)
        .order_by(LeagueRuleSnapshot.fetched_at.desc())
        .first()
    )
    return {
        **result,
        "rule_snapshot_id": snapshot.id if snapshot else None,
    }


@router.get("/leagues/{league_id}/waivers/{week}")
def waiver_recommendations(
    league_id: str,
    week: int,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        result = WaiverService(db).recommend(league_id, week)
    except ValueError as exc:
        raise _unprocessable(
            "waivers_unavailable",
            "Waiver recommendations are unavailable for this league and week.",
            league_id=league_id,
            exc=exc,
        ) from exc
    return {**result, **_meta(db, league_id)}


@router.get("/leagues/{league_id}/draft/board")
def draft_board(league_id: str, user: AppUser = Depends(get_current_user), db: Session = Depends(get_db)):
    from src.app.config import get_settings
    from src.app.league.sleeper.client import SleeperClient

    league = db.query(League).filter(League.league_id == league_id).one_or_none()
    season = league.season if league else 2026
    board = DraftBoardService(db).load_board(season, league_id=league_id)
    settings = get_settings()
    client = SleeperClient(use_fixtures=settings.use_sleeper_fixtures)
    nfl_state = client.get_nfl_state()
    raw = (league.raw_json or {}) if league else {}
    draft_id = raw.get("draft_id")
    context = {
        "draft_status": "live" if draft_id else "preseason",
        "draft_id": draft_id,
        "season": nfl_state.get("season"),
        "nfl_week": nfl_state.get("week"),
        "current_pick": (raw.get("metadata") or {}).get("current_pick"),
        "on_clock_roster_id": (raw.get("metadata") or {}).get("on_clock_roster_id"),
    }
    return {
        "board": board,
        "entries": board["entries"],
        "context": context,
        **board,
        **_meta(db, league_id),
    }


@router.get("/leagues/{league_id}/draft/checklist")
def draft_checklist(
    league_id: str,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Market-ordered checklist with checks/Xs — does not use VORP."""
    from src.app.decisions.draft_checklist import DraftChecklistService

    league = db.query(League).filter(League.league_id == league_id).one_or_none()
    season = league.season if league else 2026
    payload = DraftChecklistService(db).load(season, league_id=league_id)
    # Checklist freshness comes from the sealed artifact's own generated_at, so the
    # payload wins over _meta's rule-snapshot/projection-run values for the two keys
    # they share (data_as_of, projection_run_id).
    return {**_meta(db, league_id), **payload}


@router.post("/sleeper/connect")
def sleeper_connect(
    user: AppUser = Depends(require_csrf),
    db: Session = Depends(get_db),
    idempotency_key: str = Depends(require_idempotency_key),
):
    settings = get_settings()
    username = settings.sleeper_username or "fixture_owner"
    sync = SleeperSyncService(db, use_fixtures=settings.use_sleeper_fixtures)
    user_data = sync.connect_user(username)
    return {"status": "connected", "user_id": user_data.get("user_id"), "username": user_data.get("username")}


@router.post("/sync")
def sync_leagues(
    user: AppUser = Depends(require_csrf),
    db: Session = Depends(get_db),
    idempotency_key: str = Depends(require_idempotency_key),
):
    runner = JobRunner(db)
    job = runner.run("sync-leagues", lambda: run_daily_refresh(db), idempotency_key=idempotency_key)
    return {"status": job.status, "job_id": job.id, "metadata": job.metadata_json}
