"""Dynasty endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.app.api.deps import get_current_user, get_db, require_csrf, require_idempotency_key
from src.app.decisions.dynasty import DynastyService
from src.app.persistence.models import AppUser, ManagerState

router = APIRouter()


class ManagerStateOverride(BaseModel):
    label: str


@router.get("/leagues/{league_id}/dynasty/{roster_id}")
def dynasty_state(
    league_id: str,
    roster_id: int,
    week: int = 1,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Contender/rebuilder state and rookie-pick projection for one roster.

    Both come from the league's own draws, rosters, and traded picks. This route
    used to pass fixed numbers into the inference, so every roster in every
    league returned the same label, the same probabilities, and the same
    projected pick. Evaluating also persists `manager_state`, which is what the
    trade engine reads to attach contender context to an evaluation.
    """
    service = DynastyService(db)
    result = service.evaluate_roster(league_id, roster_id, week=week)
    pick = service.project_rookie_pick_for_roster(league_id, roster_id, week=week)
    return {
        "league_id": league_id,
        "roster_id": roster_id,
        "week": week,
        "manager_state": {
            "label": result.overridden_label or result.label,
            "inferred_label": result.label,
            "overridden_label": result.overridden_label,
            "probabilities": result.probabilities,
            "features": result.features,
            # Named so a thin inference is visible rather than implied.
            "unavailable_features": result.unavailable_features,
            "feature_coverage": result.feature_coverage,
        },
        "rookie_pick_projection": pick,
        "data_as_of": datetime.now(UTC).isoformat(),
    }


@router.put("/leagues/{league_id}/managers/{roster_id}/state")
def override_manager_state(
    league_id: str,
    roster_id: int,
    payload: ManagerStateOverride,
    user: AppUser = Depends(require_csrf),
    db: Session = Depends(get_db),
    idempotency_key: str = Depends(require_idempotency_key),
):
    row = ManagerState(
        league_id=league_id,
        roster_id=roster_id,
        as_of=datetime.now(UTC),
        label=payload.label,
        probabilities_json={},
        features_json={},
        overridden_label=payload.label,
    )
    db.add(row)
    db.commit()
    return {"status": "ok", "overridden_label": payload.label}
