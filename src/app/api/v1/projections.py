"""Projection endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.app.api.deps import get_current_user, get_db
from src.app.persistence.models import AppUser, League, PlayerProjection, ProjectionRun
from src.app.persistence.repositories import ProjectionRepository

router = APIRouter()

#: Discriminator on every projection payload. ``projected`` means the numbers
#: came from a stored, promoted run; ``unavailable`` means there is nothing to
#: show. There is deliberately no third mode that invents numbers.
MODE_PROJECTED = "projected"
MODE_UNAVAILABLE = "unavailable"


@router.get("/projections/players/{player_id}")
def get_player_projection(
    player_id: str,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    projection = (
        db.query(PlayerProjection)
        .filter(PlayerProjection.player_id == player_id)
        .order_by(PlayerProjection.id.desc())
        .first()
    )
    if projection is None:
        return {
            "player_id": player_id,
            "mode": MODE_UNAVAILABLE,
            "mean": None,
            "quantiles": None,
            "availability_probability": None,
            "reason": "no_promoted_projection_for_player",
            "data_as_of": datetime.now(UTC).isoformat(),
            "projection_run_id": None,
        }
    run = db.query(ProjectionRun).filter(ProjectionRun.id == projection.run_id).one()
    return {
        "player_id": player_id,
        "mode": MODE_PROJECTED,
        "mean": projection.mean_json,
        "quantiles": projection.quantiles_json,
        "availability_probability": projection.availability_probability,
        "data_as_of": run.as_of.isoformat(),
        "projection_run_id": run.id,
    }


@router.get("/leagues/{league_id}/rankings")
def rankings(
    league_id: str,
    mode: str = "weekly",
    week: int = 1,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    league = db.query(League).filter(League.league_id == league_id).one_or_none()
    if league is None:
        raise HTTPException(status_code=404, detail="League not found")
    repo = ProjectionRepository(db)
    run = repo.active_run(mode=mode, season=league.season, week=week if mode == "weekly" else None)
    if run is None:
        run = repo.active_run(mode="preseason", season=league.season, week=None)
    if run is None:
        return {
            "league_id": league_id,
            "mode": mode,
            "availability": MODE_UNAVAILABLE,
            "reason": "no_promoted_projection_run",
            "rankings": [],
            "data_as_of": datetime.now(UTC).isoformat(),
            "projection_run_id": None,
        }
    projections = repo.player_projections(run.id)
    ranked = sorted(
        projections,
        key=lambda row: float((row.mean_json or {}).get("points", 0.0)),
        reverse=True,
    )
    return {
        "league_id": league_id,
        "mode": mode,
        "availability": MODE_PROJECTED,
        "rankings": [
            {
                "player_id": row.player_id,
                "name": (row.mean_json or {}).get("name"),
                "position": (row.mean_json or {}).get("position"),
                "team": row.team,
                "points": (row.mean_json or {}).get("points"),
            }
            for row in ranked[:100]
        ],
        "data_as_of": run.as_of.isoformat(),
        "projection_run_id": run.id,
    }
