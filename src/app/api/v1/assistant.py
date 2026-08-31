"""Assistant endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from src.app.api.deps import get_db, require_csrf, require_idempotency_key
from src.app.assistant.gateway import AssistantGateway
from src.app.assistant.validation import MAX_LEAGUE_ID_CHARS, MAX_WEEK, MIN_WEEK
from src.app.config import get_settings
from src.app.middleware.rate_limit import client_key, limiter
from src.app.persistence.models import AppUser

router = APIRouter()

#: Read once at import so the request schema advertises a fixed bound in the
#: OpenAPI document instead of a value that drifts per request.
MAX_MESSAGE_CHARS = get_settings().assistant_max_message_chars


class AssistantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    league_id: str | None = Field(default=None, min_length=1, max_length=MAX_LEAGUE_ID_CHARS)
    week: int = Field(default=1, ge=MIN_WEEK, le=MAX_WEEK)


@router.post("/assistant/responses")
def assistant_response(
    payload: AssistantRequest,
    request: Request,
    user: AppUser = Depends(require_csrf),
    db: Session = Depends(get_db),
    idempotency_key: str = Depends(require_idempotency_key),
):
    settings = get_settings()
    limiter.check(f"assistant:{client_key(request)}", limit=settings.assistant_rate_limit_per_minute)
    gateway = AssistantGateway(db)
    return gateway.respond(user, payload.message, league_id=payload.league_id, week=payload.week)
