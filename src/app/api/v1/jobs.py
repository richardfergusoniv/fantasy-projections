"""Job status endpoints."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.app.api.deps import get_current_user, get_db
from src.app.config import get_settings
from src.app.logging import redact_text
from src.app.persistence.models import AppUser, JobRun

router = APIRouter()

#: Job errors come from ``str(exc)`` on arbitrary internals: DSNs, file paths,
#: provider responses. Only a classified summary is exposed to the browser.
_ERROR_KINDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("database_unavailable", re.compile(r"operationalerror|could not connect|connection refused", re.I)),
    ("lock_contention", re.compile(r"advisory lock|deadlock|lock timeout", re.I)),
    ("upstream_unavailable", re.compile(r"httpx|timeout|status code|connecterror", re.I)),
    ("validation_failed", re.compile(r"validation|invalid|gate|contract", re.I)),
    ("not_found", re.compile(r"not found|no such", re.I)),
)

_MAX_DEBUG_ERROR_CHARS = 300


def summarize_job_error(raw: str | None) -> dict | None:
    """Classify a stored job error into a typed, non-leaking summary."""
    if not raw:
        return None
    kind = "unknown"
    for name, pattern in _ERROR_KINDS:
        if pattern.search(raw):
            kind = name
            break
    summary = {
        "code": "job_failed",
        "kind": kind,
        "message": "Job failed. See server logs for the full error.",
    }
    if get_settings().is_development:
        summary["debug_message"] = redact_text(raw)[:_MAX_DEBUG_ERROR_CHARS]
    return summary


@router.get("/jobs/{job_id}")
def get_job(job_id: str, user: AppUser = Depends(get_current_user), db: Session = Depends(get_db)):
    job = db.query(JobRun).filter(JobRun.id == job_id).one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    error_summary = summarize_job_error(job.error)
    return {
        "id": job.id,
        "job_name": job.job_name,
        "status": job.status,
        "correlation_id": job.correlation_id,
        "started_at": job.started_at.isoformat(),
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "error": error_summary["code"] if error_summary else None,
        "error_summary": error_summary,
        "metadata": job.metadata_json,
    }
