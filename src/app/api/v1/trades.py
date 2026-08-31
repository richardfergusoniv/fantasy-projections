"""Trade API wired to services and tendency learning."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy.orm import Session

from src.app.api.deps import get_current_user, get_db, require_csrf, require_idempotency_key
from src.app.decisions.services import TradeService
from src.app.decisions.tendencies import ManagerTendencyService
from src.app.decisions.trades import TradeSide
from src.app.logging import get_logger
from src.app.persistence.models import AppUser, TradeProposal
from src.app.persistence.repositories import ProjectionRepository

logger = get_logger(__name__)

router = APIRouter()

#: Domain rejections mapped to stable client-facing codes. Keyed by exception
#: class name so this stays decoupled from the decision layer's internals.
_TRADE_REJECTION_CODES = {
    "RedraftPickNotTradeable": "picks_not_tradeable_in_redraft",
}

MAX_PLAYERS_PER_SIDE = 12
MAX_PICKS_PER_SIDE = 12
MAX_ROSTER_ID = 64

#: GSIS / Sleeper style identifiers only. Keeps free-form text (and anything
#: that could be smuggled into a prompt or a query) out of trade requests.
PlayerId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9][A-Za-z0-9._\-]*$"),
]


class PickAsset(BaseModel):
    """A draft pick reference. Pick *value* is computed server-side.

    ``extra="forbid"`` is what rejects a client-supplied ``value``: a caller
    cannot assert what its own picks are worth.
    """

    model_config = ConfigDict(extra="forbid")

    season: int = Field(ge=2000, le=2100)
    round: int = Field(ge=1, le=10)
    original_roster_id: int | None = Field(default=None, ge=1, le=MAX_ROSTER_ID)


class TradeSideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roster_id: int = Field(ge=1, le=MAX_ROSTER_ID)
    player_ids: list[PlayerId] = Field(default_factory=list, max_length=MAX_PLAYERS_PER_SIDE)
    pick_assets: list[PickAsset] = Field(default_factory=list, max_length=MAX_PICKS_PER_SIDE)


class TradeEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    side_a: TradeSideRequest
    side_b: TradeSideRequest
    horizon: Literal["weekly", "ros", "dynasty"] = "ros"


def _to_trade_side(side: TradeSideRequest) -> TradeSide:
    return TradeSide(
        roster_id=side.roster_id,
        player_ids=list(side.player_ids),
        pick_assets=[pick.model_dump(exclude_none=True) for pick in side.pick_assets],
    )


@router.post("/leagues/{league_id}/trades/evaluate")
def evaluate_trade_endpoint(
    league_id: str,
    payload: TradeEvaluateRequest,
    user: AppUser = Depends(require_csrf),
    db: Session = Depends(get_db),
    idempotency_key: str = Depends(require_idempotency_key),
):
    service = TradeService(db)
    try:
        result = service.evaluate(
            league_id,
            _to_trade_side(payload.side_a),
            _to_trade_side(payload.side_b),
            horizon=payload.horizon,
        )
    except ValueError as exc:
        reason = _TRADE_REJECTION_CODES.get(type(exc).__name__, "trade_not_evaluable")
        logger.warning(
            "trade_evaluation_rejected",
            league_id=league_id,
            reason=reason,
            exception_type=type(exc).__name__,
            detail=str(exc),
        )
        raise HTTPException(
            status_code=422,
            detail={
                "code": "trade_not_evaluable",
                "reason": reason,
                "message": "This trade cannot be evaluated for this league.",
            },
        ) from exc
    run = ProjectionRepository(db).active_run(mode="ros", season=2026, week=None)
    return {
        "league_id": league_id,
        "horizon": payload.horizon,
        "objective": result.objective,
        "fairness": result.fairness,
        "acceptance": result.acceptance,
        "data_as_of": datetime.now(UTC).isoformat(),
        "projection_run_id": run.id if run else "fixture",
    }


class TradeProposalRequest(BaseModel):
    created_by_roster_id: int
    sides_json: dict
    direction: str = "outgoing"


class TradeProposalStatusUpdate(BaseModel):
    status: str


@router.post("/leagues/{league_id}/trades/proposals")
def create_proposal(
    league_id: str,
    payload: TradeProposalRequest,
    user: AppUser = Depends(require_csrf),
    db: Session = Depends(get_db),
    idempotency_key: str = Depends(require_idempotency_key),
):
    proposal = TradeProposal(
        league_id=league_id,
        created_by_roster_id=payload.created_by_roster_id,
        sides_json=payload.sides_json,
        direction=payload.direction,
        status="offered",
    )
    db.add(proposal)
    db.commit()
    return {"proposal_id": proposal.id, "status": proposal.status}


@router.put("/leagues/{league_id}/trades/proposals/{proposal_id}/status")
def update_proposal_status(
    league_id: str,
    proposal_id: str,
    payload: TradeProposalStatusUpdate,
    user: AppUser = Depends(require_csrf),
    db: Session = Depends(get_db),
    idempotency_key: str = Depends(require_idempotency_key),
):
    proposal = db.query(TradeProposal).filter(TradeProposal.id == proposal_id).one()
    proposal.status = payload.status
    ManagerTendencyService(db).rebuild(league_id)
    db.commit()
    return {"proposal_id": proposal.id, "status": proposal.status}


@router.get("/leagues/{league_id}/trades/history")
def trade_history(league_id: str, user: AppUser = Depends(get_current_user), db: Session = Depends(get_db)):
    proposals = db.query(TradeProposal).filter(TradeProposal.league_id == league_id).all()
    return {
        "history": [{"id": p.id, "status": p.status, "sides": p.sides_json} for p in proposals],
        "data_as_of": datetime.now(UTC).isoformat(),
    }


@router.get("/leagues/{league_id}/managers/{roster_id}/tendencies")
def manager_tendencies(
    league_id: str,
    roster_id: int,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    features = ManagerTendencyService(db).get(league_id, roster_id)
    return {
        "league_id": league_id,
        "roster_id": roster_id,
        "sample_size": features.sample_size,
        "features": {
            "youth_preference": features.youth_preference,
            "pick_preference": features.pick_preference,
            "consolidation_bias": features.consolidation_bias,
            "avg_package_size": features.avg_package_size,
            "accept_rate": features.accept_rate,
        },
        "data_as_of": datetime.now(UTC).isoformat(),
    }
