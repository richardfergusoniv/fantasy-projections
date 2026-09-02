"""Research changed availability players during sync jobs."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from src.app.availability.research import (
    MODE_DISABLED,
    MODE_FIXTURE,
    ResearchUnavailable,
    build_provider,
    resolve_research_mode,
)
from src.app.availability.service import AvailabilityService, EvidenceRejected
from src.app.config import get_settings
from src.app.persistence.models import AvailabilityEvent

logger = logging.getLogger(__name__)


def research_changed_players(session: Session, *, limit: int = 25, mode: str | None = None) -> dict:
    settings = get_settings()
    availability = AvailabilityService(session)
    resolved_mode = mode or resolve_research_mode(settings)

    events = (
        session.query(AvailabilityEvent)
        .filter(AvailabilityEvent.cleared_at.is_(None))
        .order_by(AvailabilityEvent.active_from.desc())
        .limit(limit)
        .all()
    )

    if resolved_mode == MODE_DISABLED:
        return _unavailable(resolved_mode, "research_disabled", len(events))
    try:
        provider = build_provider(settings, mode=resolved_mode, session=session)
    except ResearchUnavailable as exc:
        logger.warning("injury_research_unavailable", extra={"mode": resolved_mode, "reason": str(exc)})
        return _unavailable(resolved_mode, str(exc), len(events))

    researched = 0
    rejected = 0
    for event in events:
        if availability.repo.evidence_for_player(event.player_id):
            continue
        try:
            result = provider.research(event.player_id)
        except ResearchUnavailable as exc:
            logger.warning("injury_research_unavailable", extra={"mode": resolved_mode, "reason": str(exc)})
            return _unavailable(resolved_mode, str(exc), len(events), researched=researched)
        for claim in result.claims:
            try:
                evidence = availability.add_evidence(claim)
            except EvidenceRejected as exc:
                rejected += 1
                logger.warning("injury_evidence_rejected", extra={"reason": str(exc)})
                continue
            event.evidence_ids = list(set((event.evidence_ids or []) + [evidence.id]))
            researched += 1
    session.flush()
    return {
        "researched": researched,
        "rejected": rejected,
        "active_events": len(events),
        "mode": resolved_mode,
        "synthetic": resolved_mode == MODE_FIXTURE,
        "status": "ok",
        "available": True,
    }


def _unavailable(mode: str, reason: str, active_events: int, *, researched: int = 0) -> dict:
    return {
        "researched": researched,
        "rejected": 0,
        "active_events": active_events,
        "mode": mode,
        "synthetic": False,
        "status": "unavailable",
        "available": False,
        "reason": reason,
    }
