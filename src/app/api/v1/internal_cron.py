"""Bearer-protected cron endpoints for external schedulers."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException

from src.app.config import get_settings
from src.app.jobs.scheduler import enqueue_due_slots, process_outbox, run_due

router = APIRouter(prefix="/internal/cron", include_in_schema=False)


@router.post("/run-due")
def cron_run_due(authorization: str | None = Header(default=None)):
    settings = get_settings()
    if not settings.cron_secret:
        raise HTTPException(status_code=503, detail="cron_not_configured")
    expected = f"Bearer {settings.cron_secret}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="unauthorized")
    if settings.long_jobs_external:
        enqueued = enqueue_due_slots()
        return {"status": "ok", "mode": "enqueue_only", "enqueued": enqueued}
    results = run_due()
    return {"status": "ok", "mode": "inline", "executed": len(results), "results": results}


@router.post("/process-outbox")
def cron_process_outbox(authorization: str | None = Header(default=None)):
    settings = get_settings()
    if not settings.cron_secret:
        raise HTTPException(status_code=503, detail="cron_not_configured")
    expected = f"Bearer {settings.cron_secret}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="unauthorized")
    results = process_outbox()
    return {"status": "ok", "processed": len(results), "results": results}
