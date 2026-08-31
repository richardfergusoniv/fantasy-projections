"""Player evidence endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.app.api.deps import get_current_user, get_db
from src.app.persistence.models import AppUser, InjuryEvidence, PlayerProjection, ProjectionRun
from src.app.persistence.repositories import ProjectionRepository

router = APIRouter()


@router.get("/players/{player_id}/injury-evidence")
def injury_evidence(player_id: str, user: AppUser = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(InjuryEvidence).filter(InjuryEvidence.player_id == player_id).all()
    return {
        "player_id": player_id,
        "evidence": [
            {
                "id": row.id,
                "source_url": row.source_url,
                "source_title": row.source_title,
                "claim": row.claim_json,
                "confidence": row.confidence,
            }
            for row in rows
        ],
        "data_as_of": datetime.now(UTC).isoformat(),
    }


@router.get("/players/{player_id}/projection-changes")
def projection_changes(
    player_id: str,
    season: int = 2026,
    week: int = 1,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    repo = ProjectionRepository(db)
    current = repo.active_run(mode="weekly", season=season, week=week)
    previous_id = None
    if current:
        pointer = repo.active_run(mode="weekly", season=season, week=week)
        if pointer:
            from src.app.persistence.models import ActiveProjectionPointer

            row = (
                db.query(ActiveProjectionPointer)
                .filter(
                    ActiveProjectionPointer.mode == "weekly",
                    ActiveProjectionPointer.season == season,
                    ActiveProjectionPointer.week == week,
                )
                .one_or_none()
            )
            previous_id = row.previous_run_id if row else None
    changes = []
    if current and previous_id:
        current_row = (
            db.query(PlayerProjection)
            .filter(PlayerProjection.run_id == current.id, PlayerProjection.player_id == player_id)
            .one_or_none()
        )
        previous_row = (
            db.query(PlayerProjection)
            .filter(PlayerProjection.run_id == previous_id, PlayerProjection.player_id == player_id)
            .one_or_none()
        )
        if current_row and previous_row:
            current_points = float((current_row.mean_json or {}).get("points", 0.0))
            previous_points = float((previous_row.mean_json or {}).get("points", 0.0))
            delta = current_points - previous_points
            if abs(delta) > 0.01:
                changes.append(
                    {
                        "player_id": player_id,
                        "from_run_id": previous_id,
                        "to_run_id": current.id,
                        "delta_points": delta,
                        "drivers": ["weekly_promotion"],
                    }
                )
    run = current or db.query(ProjectionRun).order_by(ProjectionRun.as_of.desc()).first()
    return {
        "player_id": player_id,
        "changes": changes,
        "data_as_of": run.as_of.isoformat() if run else datetime.now(UTC).isoformat(),
        "projection_run_id": run.id if run else "fixture",
    }
