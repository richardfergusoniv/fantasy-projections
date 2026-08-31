"""Operations dashboard API."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.app.api.deps import get_current_user, get_db, require_csrf, require_idempotency_key
from src.app.config import get_settings
from src.app.jobs.runner import JobRunner
from src.app.persistence.models import (
    ActiveProjectionPointer,
    AppUser,
    AssistantAudit,
    InjuryEvidence,
    JobRun,
    League,
    ProjectionRun,
    PromotionEvent,
    SourceSnapshot,
)

router = APIRouter()

HORIZONS = ("weekly", "ros", "dynasty", "preseason")

#: Beyond this, the corresponding feed is reported as a degraded dependency.
DATA_STALE_AFTER_HOURS = 36.0
EVIDENCE_STALE_AFTER_HOURS = 72.0


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _age_hours(value: datetime | None, now: datetime) -> float | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return round((now - value).total_seconds() / 3600.0, 3)


def _resolve_season_week(db: Session) -> dict:
    from src.app.jobs.handlers import resolve_season_week

    return resolve_season_week(db).to_dict()


def _pointer_runs(db: Session, season: int, week: int | None) -> dict[str, ProjectionRun | None]:
    pointers = (
        db.query(ActiveProjectionPointer)
        .filter(ActiveProjectionPointer.season == season)
        .all()
    )
    by_mode: dict[str, ActiveProjectionPointer] = {}
    for pointer in pointers:
        if pointer.mode == "weekly" and pointer.week != week:
            continue
        by_mode[pointer.mode] = pointer
    runs: dict[str, ProjectionRun | None] = {}
    for horizon in HORIZONS:
        pointer = by_mode.get(horizon)
        runs[horizon] = (
            db.query(ProjectionRun).filter(ProjectionRun.id == pointer.run_id).one_or_none()
            if pointer is not None
            else None
        )
    return runs


def _artifact_store_health() -> dict:
    """Cheap read/write probe. Never raises — a broken store is a status, not a 500."""
    settings = get_settings()
    backend = settings.artifact_backend
    try:
        if backend != "local":
            return {
                "backend": backend,
                "status": "unknown",
                "detail": "probe_skipped_remote_backend",
                "writable": None,
                "readable": None,
            }
        from src.app.artifacts.store import get_artifact_store

        store = get_artifact_store()
        uri = store.put_json({"probe": "operations_status"})
        echoed = store.get_json(uri)
        healthy = echoed == {"probe": "operations_status"}
        return {
            "backend": backend,
            "status": "healthy" if healthy else "degraded",
            "detail": "round_trip_ok" if healthy else "round_trip_mismatch",
            "writable": True,
            "readable": healthy,
        }
    except Exception as exc:  # noqa: BLE001 — status endpoints must not throw
        return {
            "backend": backend,
            "status": "degraded",
            "detail": f"{type(exc).__name__}",
            "writable": False,
            "readable": False,
        }


def _scheduler_health(db: Session, now: datetime) -> dict:
    from src.app.jobs.scheduler import due_slots, last_completed_slot, last_successful_runs, next_due_job

    upcoming = next_due_job(now)
    last_runs = last_successful_runs(db)
    pending = due_slots(now, last_runs)
    return {
        "next_due_job": (
            {
                "job_name": upcoming.job_name,
                "scheduled_at": _iso(upcoming.scheduled_at),
                "slot_key": upcoming.slot_key,
            }
            if upcoming is not None
            else None
        ),
        "last_completed_slot": last_completed_slot(db, now),
        "due_now": [
            {"job_name": slot.job_name, "scheduled_at": _iso(slot.scheduled_at)}
            for slot in pending
        ],
        "timezone": "America/Los_Angeles",
    }


@router.get("/operations/status")
def operations_status(user: AppUser = Depends(get_current_user), db: Session = Depends(get_db)):
    settings = get_settings()
    now = datetime.now(UTC)
    season_week = _resolve_season_week(db)
    season = season_week["season"]
    week = season_week["week"]

    latest_job = db.query(JobRun).order_by(JobRun.started_at.desc()).first()
    last_successful_job = (
        db.query(JobRun)
        .filter(JobRun.status == "succeeded")
        .order_by(JobRun.started_at.desc())
        .first()
    )
    last_failed_job = (
        db.query(JobRun)
        .filter(JobRun.status == "failed")
        .order_by(JobRun.started_at.desc())
        .first()
    )
    latest_source = db.query(SourceSnapshot).order_by(SourceSnapshot.fetched_at.desc()).first()
    latest_evidence = db.query(InjuryEvidence).order_by(InjuryEvidence.fetched_at.desc()).first()
    latest_promotion = db.query(PromotionEvent).order_by(PromotionEvent.created_at.desc()).first()

    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    estimated_monthly = sum(
        row.estimated_cost_usd or 0.0
        for row in db.query(AssistantAudit).filter(AssistantAudit.created_at >= month_start).all()
    )

    failed_promotions = (
        db.query(PromotionEvent)
        .filter(PromotionEvent.promoted.is_(False))
        .order_by(PromotionEvent.created_at.desc())
        .limit(10)
        .all()
    )
    failed_gates: list[str] = []
    recent_gate_failures: list[dict] = []
    for event in failed_promotions:
        validation = event.validation_json or {}
        gates = validation.get("gates", {})
        failures: list[str] = []
        if isinstance(gates, dict):
            for name, result in gates.items():
                if isinstance(result, dict):
                    failures.extend(f"{name}:{item}" for item in result.get("failures", []))
                elif name == "failures" and isinstance(result, list):
                    failures.extend(str(item) for item in result)
        failed_gates.extend(failures)
        recent_gate_failures.append(
            {
                "mode": event.mode,
                "candidate_run_id": event.candidate_run_id,
                "reason": validation.get("reason"),
                "failures": failures,
                "created_at": _iso(event.created_at),
            }
        )

    last_rollback_event = (
        db.query(PromotionEvent)
        .filter(PromotionEvent.mode.isnot(None))
        .order_by(PromotionEvent.created_at.desc())
        .limit(50)
        .all()
    )
    last_rollback = None
    for event in last_rollback_event:
        if (event.validation_json or {}).get("derivation") == "rollback":
            last_rollback = {
                "mode": event.mode,
                "restored_run_id": event.candidate_run_id,
                "superseded_run_id": event.previous_run_id,
                "reason": (event.validation_json or {}).get("reason"),
                "created_at": _iso(event.created_at),
            }
            break

    runs = _pointer_runs(db, season, week)
    active_releases = {horizon: (run.id if run else None) for horizon, run in runs.items()}
    modes_by_horizon = {
        horizon: (run.artifact_mode if run else None) for horizon, run in runs.items()
    }
    league_ids = [row.league_id for row in db.query(League).filter(League.season == season).all()]
    active_releases_by_league = {league_id: dict(active_releases) for league_id in league_ids}

    from src.app.projections.weekly_v2_bridge import weekly_v2_readiness

    readiness = weekly_v2_readiness(season, week)
    artifact_store = _artifact_store_health()
    scheduler = _scheduler_health(db, now)

    data_age = _age_hours(latest_source.fetched_at if latest_source else None, now)
    evidence_age = _age_hours(latest_evidence.fetched_at if latest_evidence else None, now)

    dependencies = [
        {
            "name": "sleeper",
            "status": (
                "unknown"
                if latest_source is None
                else "degraded"
                if latest_source.health_verdict != "healthy"
                or (data_age is not None and data_age > DATA_STALE_AFTER_HOURS)
                else "healthy"
            ),
            "detail": (
                f"{settings.sleeper_mode}:"
                f"{latest_source.health_verdict if latest_source else 'no_snapshot'}"
            ),
            "mode": settings.sleeper_mode,
        },
        {
            "name": "injury_evidence",
            "status": (
                "unknown"
                if latest_evidence is None
                else "degraded"
                if evidence_age is not None and evidence_age > EVIDENCE_STALE_AFTER_HOURS
                else "healthy"
            ),
            "detail": "stale" if (evidence_age or 0) > EVIDENCE_STALE_AFTER_HOURS else "fresh",
        },
        {
            "name": "artifact_store",
            "status": artifact_store["status"],
            "detail": artifact_store["detail"],
        },
        {
            "name": "weekly_v2_artifacts",
            "status": "healthy" if readiness.is_trained else "degraded",
            "detail": readiness.state,
        },
        {
            "name": "assistant_llm",
            "status": "healthy" if settings.openai_api_key else "degraded",
            "detail": "configured" if settings.openai_api_key else "not_configured",
        },
    ]
    degraded_dependencies = [dep for dep in dependencies if dep["status"] != "healthy"]

    return {
        "generated_at": now.isoformat(),
        "season": season,
        "week": week,
        "season_week_source": season_week["source"],
        "data_as_of": _iso(latest_source.fetched_at if latest_source else None) or now.isoformat(),
        "last_sync_at": _iso(latest_source.fetched_at) if latest_source else None,
        "active_releases": active_releases,
        "active_releases_by_league": active_releases_by_league,
        "active_projection_run_id": active_releases["weekly"],
        "modes": {
            # Fixture data must never be mistakable for live league data.
            "sleeper_source": settings.sleeper_mode,
            "weekly_v2_state": readiness.state,
            "weekly_v2_model_version": readiness.model_version,
            "weekly_v2_manifest_uri": readiness.manifest_uri,
            "weekly_v2_reasons": list(readiness.reasons),
            "auto_publish_allowed": readiness.is_trained,
            "by_horizon": modes_by_horizon,
        },
        "jobs": {
            "latest": _job_payload(latest_job),
            "last_successful": _job_payload(last_successful_job),
            "last_failed": _job_payload(last_failed_job),
        },
        "last_successful_job_at": _iso(
            (last_successful_job.finished_at or last_successful_job.started_at)
            if last_successful_job
            else None
        ),
        "last_failed_job_at": _iso(
            (last_failed_job.finished_at or last_failed_job.started_at) if last_failed_job else None
        ),
        "freshness": {
            "data": {
                "as_of": _iso(latest_source.fetched_at) if latest_source else None,
                "age_hours": data_age,
                "endpoint": latest_source.endpoint if latest_source else None,
                "stale": data_age is None or data_age > DATA_STALE_AFTER_HOURS,
            },
            "evidence": {
                "as_of": _iso(latest_evidence.fetched_at) if latest_evidence else None,
                "age_hours": evidence_age,
                "stale": evidence_age is None or evidence_age > EVIDENCE_STALE_AFTER_HOURS,
            },
        },
        "scheduler": scheduler,
        "artifact_store": artifact_store,
        "recent_gate_failures": recent_gate_failures,
        "failed_gates": failed_gates,
        "last_rollback": last_rollback,
        "dependencies": dependencies,
        "degraded_dependencies": degraded_dependencies,
        "latest_job": _job_payload(latest_job),
        "latest_source": {
            "endpoint": latest_source.endpoint,
            "fetched_at": _iso(latest_source.fetched_at),
            "health_verdict": latest_source.health_verdict,
        }
        if latest_source
        else None,
        "latest_promotion": {
            "mode": latest_promotion.mode,
            "promoted": latest_promotion.promoted,
            "created_at": _iso(latest_promotion.created_at),
        }
        if latest_promotion
        else None,
        "assistant_cost_usd_month": estimated_monthly,
        "estimated_month_cost_usd": estimated_monthly,
        "openai_configured": bool(settings.openai_api_key),
    }


def _job_payload(job: JobRun | None) -> dict | None:
    if job is None:
        return None
    return {
        "id": job.id,
        "name": job.job_name,
        "status": job.status,
        "attempt": job.attempt,
        "started_at": _iso(job.started_at),
        "finished_at": _iso(job.finished_at),
        "error": job.error,
    }


@router.post("/operations/jobs/{job_name}/run")
def run_job_now(
    job_name: str,
    user: AppUser = Depends(require_csrf),
    db: Session = Depends(get_db),
    idempotency_key: str = Depends(require_idempotency_key),
):
    from src.app.jobs.handlers import JOB_HANDLERS

    handler = JOB_HANDLERS.get(job_name)
    if handler is None:
        return {"status": "error", "detail": "unknown job"}
    runner = JobRunner(db)

    def _body() -> dict:
        # Operator-triggered, so artifact readiness downgrades to a warning and
        # the resulting run is labelled with the artifact mode it actually used.
        return handler(db, automatic=False)

    job = runner.run(job_name, _body, idempotency_key=idempotency_key)
    return {
        "job_id": job.id,
        "status": job.status,
        "attempt": job.attempt,
        "metadata": job.metadata_json,
    }


@router.post("/operations/projections/rollback")
def rollback_projection(
    mode: str = "weekly",
    season: int | None = None,
    week: int | None = None,
    user: AppUser = Depends(require_csrf),
    db: Session = Depends(get_db),
    idempotency_key: str = Depends(require_idempotency_key),
):
    from src.app.releases.rollback import ProjectionRollbackService

    resolved = _resolve_season_week(db)
    season = resolved["season"] if season is None else season
    if week is None and mode == "weekly":
        week = resolved["week"]

    service = ProjectionRollbackService(db)
    restored = service.rollback(mode, season, week)
    if restored is None:
        return {"status": "unchanged", "detail": "no previous projection pointer"}
    return {
        "status": "rolled_back",
        "mode": mode,
        "season": season,
        "week": week,
        "active_projection_run_id": restored,
    }
