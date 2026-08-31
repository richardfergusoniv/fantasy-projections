"""League API endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.app.api.deps import get_current_user, get_db, require_csrf, require_idempotency_key
from src.app.jobs.runner import JobRunner
from src.app.jobs.handlers import run_daily_refresh
from src.app.decisions.draft_board import DraftBoardService
from src.app.decisions.services import LineupService, WaiverService
from src.app.league.sleeper.sync import SleeperSyncService
from src.app.persistence.models import (
    AppUser,
    League,
    LeagueDraftRule,
    LeagueRuleSnapshot,
    LeagueMember,
    MatchupSnapshot,
    RosterSnapshot,
)
from src.app.persistence.repositories import ProjectionRepository
from src.app.scoring.compiler import compile_sleeper_scoring
from src.app.config import get_settings
from src.app.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


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
    weekly_run = ProjectionRepository(session).active_run(mode="weekly", season=season, week=week)
    if weekly_run is None:
        weekly_run = ProjectionRepository(session).active_run(mode="preseason", season=season, week=None)
    return {
        "data_as_of": (snapshot.fetched_at if snapshot else datetime.now(UTC)).isoformat(),
        "rule_snapshot_id": snapshot.id if snapshot else None,
        "projection_run_id": weekly_run.id if weekly_run else "fixture",
    }


@router.get("/leagues")
def list_leagues(user: AppUser = Depends(get_current_user), db: Session = Depends(get_db)):
    leagues = db.query(League).all()
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
                "roster_positions": league.raw_json.get("roster_positions", []) if league.raw_json else [],
            }
        )
    meta = _meta(db, leagues[0].league_id if leagues else "")
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
    rosters = db.query(RosterSnapshot).filter(RosterSnapshot.league_id == league_id).all()
    return {
        "rosters": [
            {
                "roster_id": r.roster_id,
                "week": r.week,
                "players": r.players,
                "starters": r.starters,
                "reserve": r.reserve,
            }
            for r in rosters
        ],
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
    board = DraftBoardService().load_board(season)
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
