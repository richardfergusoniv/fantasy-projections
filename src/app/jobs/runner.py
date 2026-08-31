"""PostgreSQL advisory locks and job orchestration."""

from __future__ import annotations

import hashlib
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Generator

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.app.logging import bind_correlation_id, get_logger
from src.app.persistence.models import JobRun, utcnow

logger = get_logger(__name__)

ADVISORY_LOCK_SQL = "SELECT pg_try_advisory_lock(:key)"
ADVISORY_UNLOCK_SQL = "SELECT pg_advisory_unlock(:key)"

#: A ``running`` row older than this is assumed to belong to a crashed worker and
#: may be recovered by the next attempt. Override with ``JOB_STALE_AFTER_SECONDS``.
DEFAULT_STALE_AFTER = timedelta(hours=2)

#: Handler results carrying one of these ``status`` values are recorded with that
#: status instead of ``succeeded``, so a deliberately deferred run is not
#: indistinguishable from a run that did the work.
NON_SUCCESS_RESULT_STATUSES = frozenset({"postponed", "blocked"})


def _lock_key(job_name: str) -> int:
    """Deterministic 64-bit advisory-lock key for ``job_name``.

    ``hash()`` is seeded per process for ``str`` (PYTHONHASHSEED randomization),
    so two workers computed different keys for the same job and both "acquired"
    the lock. A SHA-256 prefix is stable across processes, hosts and releases.
    """
    digest = hashlib.sha256(job_name.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big", signed=True)


def advisory_lock_statement(job_name: str) -> tuple[str, dict[str, int]]:
    """SQL and bind parameters used to take the PostgreSQL advisory lock."""
    return ADVISORY_LOCK_SQL, {"key": _lock_key(job_name)}


def advisory_unlock_statement(job_name: str) -> tuple[str, dict[str, int]]:
    """SQL and bind parameters used to release the PostgreSQL advisory lock."""
    return ADVISORY_UNLOCK_SQL, {"key": _lock_key(job_name)}


@contextmanager
def advisory_lock(session: Session, job_name: str) -> Generator[None]:
    statement, params = advisory_lock_statement(job_name)
    acquired = session.execute(text(statement), params).scalar()
    if not acquired:
        raise RuntimeError(f"Could not acquire advisory lock for {job_name}")
    try:
        yield
    finally:
        unlock_statement, unlock_params = advisory_unlock_statement(job_name)
        session.execute(text(unlock_statement), unlock_params)


@contextmanager
def sqlite_job_guard(_session: Session, job_name: str) -> Generator[None]:
    """Best-effort single-process guard for SQLite tests."""
    yield


def job_lock(session: Session, job_name: str) -> Generator[None]:
    bind = session.get_bind()
    if bind.dialect.name == "postgresql":
        return advisory_lock(session, job_name)
    return sqlite_job_guard(session, job_name)


def _default_stale_after() -> timedelta:
    raw = os.getenv("JOB_STALE_AFTER_SECONDS")
    if not raw:
        return DEFAULT_STALE_AFTER
    try:
        return timedelta(seconds=float(raw))
    except ValueError:
        return DEFAULT_STALE_AFTER


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True)
class StartDecision:
    job: JobRun
    should_execute: bool
    reason: str


class JobRunner:
    def __init__(
        self,
        session: Session,
        *,
        session_factory: Callable[[], Session] | None = None,
        stale_after: timedelta | None = None,
    ) -> None:
        self.session = session
        self.stale_after = stale_after or _default_stale_after()
        self._session_factory = session_factory

    # ------------------------------------------------------------------ start

    def begin(
        self,
        job_name: str,
        *,
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
    ) -> StartDecision:
        """Claim a job slot, deciding whether the body should actually execute."""
        cid = bind_correlation_id(correlation_id)
        if idempotency_key:
            existing = (
                self.session.query(JobRun)
                .filter(JobRun.idempotency_key == idempotency_key)
                .one_or_none()
            )
            if existing is not None:
                if existing.status == "succeeded":
                    logger.info(
                        "job_deduplicated",
                        job_name=job_name,
                        job_id=existing.id,
                        idempotency_key=idempotency_key,
                    )
                    return StartDecision(existing, False, "duplicate_success")
                if existing.status == "running" and not self._is_stale(existing):
                    logger.info("job_already_running", job_name=job_name, job_id=existing.id)
                    return StartDecision(existing, False, "in_progress")
                reason = "stale_recovery" if existing.status == "running" else "retry"
                self._reclaim(existing, cid)
                logger.info(
                    "job_retry",
                    job_name=job_name,
                    job_id=existing.id,
                    attempt=existing.attempt,
                    reason=reason,
                )
                return StartDecision(existing, True, reason)
        job = JobRun(
            job_name=job_name,
            correlation_id=cid,
            idempotency_key=idempotency_key,
            status="running",
            attempt=1,
            started_at=utcnow(),
        )
        self.session.add(job)
        self.session.flush()
        logger.info("job_started", job_name=job_name, job_id=job.id)
        return StartDecision(job, True, "new")

    def start(
        self,
        job_name: str,
        *,
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
    ) -> JobRun:
        return self.begin(
            job_name, idempotency_key=idempotency_key, correlation_id=correlation_id
        ).job

    def _is_stale(self, job: JobRun) -> bool:
        started = _as_utc(job.started_at)
        if started is None:
            return True
        return (utcnow() - started) > self.stale_after

    def _reclaim(self, job: JobRun, correlation_id: str) -> None:
        job.attempt = (job.attempt or 1) + 1
        job.status = "running"
        job.error = None
        job.finished_at = None
        job.duration_ms = None
        job.started_at = utcnow()
        job.correlation_id = correlation_id
        self.session.add(job)
        self.session.flush()

    # ----------------------------------------------------------------- finish

    def finish(
        self,
        job: JobRun,
        *,
        status: str = "succeeded",
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JobRun:
        finished = datetime.now(UTC)
        started = _as_utc(job.started_at) or finished
        job.status = status
        job.error = error
        job.finished_at = finished
        job.duration_ms = int((finished - started).total_seconds() * 1000)
        if metadata:
            job.metadata_json = {**(job.metadata_json or {}), **metadata}
        self.session.add(job)
        self.session.flush()
        logger.info("job_finished", job_name=job.job_name, job_id=job.id, status=status)
        return job

    # -------------------------------------------------------------------- run

    def run(
        self,
        job_name: str,
        fn,
        *,
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
    ) -> JobRun:
        with job_lock(self.session, job_name):
            decision = self.begin(
                job_name, idempotency_key=idempotency_key, correlation_id=correlation_id
            )
            job = decision.job
            if not decision.should_execute:
                return job
            started = time.perf_counter()
            try:
                result = fn()
                metadata = result if isinstance(result, dict) else {}
                status = "succeeded"
                declared = metadata.get("status") if isinstance(metadata, dict) else None
                if declared in NON_SUCCESS_RESULT_STATUSES:
                    status = declared
                return self.finish(job, status=status, metadata=metadata)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "job_failed",
                    job_name=job_name,
                    job_id=job.id,
                    error=str(exc),
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                )
                self._record_durable_failure(job, exc)
                raise

    # --------------------------------------------------------- durable audit

    def _record_durable_failure(self, job: JobRun, exc: BaseException) -> None:
        """Persist the failure row even though the handler's writes are discarded.

        The failure row used to be written into the same transaction that the
        outer session context then rolled back, so a crashed job left no audit
        trail at all. Snapshot the row, roll the work back, then commit the
        failure record on a fresh short-lived session.
        """
        snapshot = self._failure_snapshot(job, exc)
        try:
            self.session.rollback()
        except Exception:  # noqa: BLE001
            logger.warning("job_failure_rollback_failed", job_id=snapshot["id"])
        session = self._new_session()
        try:
            existing = session.get(JobRun, snapshot["id"])
            if existing is None:
                session.add(JobRun(**snapshot))
            else:
                for field, value in snapshot.items():
                    setattr(existing, field, value)
            session.commit()
            logger.info("job_failure_recorded", job_id=snapshot["id"], job_name=snapshot["job_name"])
        except Exception:  # noqa: BLE001
            session.rollback()
            logger.error("job_failure_record_unpersisted", job_id=snapshot["id"])
        finally:
            session.close()

    def _failure_snapshot(self, job: JobRun, exc: BaseException) -> dict[str, Any]:
        finished = datetime.now(UTC)
        started = _as_utc(job.started_at) or finished
        error = f"{type(exc).__name__}: {exc}"
        return {
            "id": job.id,
            "job_name": job.job_name,
            "correlation_id": job.correlation_id,
            "idempotency_key": job.idempotency_key,
            "status": "failed",
            "attempt": job.attempt or 1,
            "started_at": started,
            "finished_at": finished,
            "duration_ms": int((finished - started).total_seconds() * 1000),
            "error": error[:4000],
            "metadata_json": {**(job.metadata_json or {}), "failure": error[:512]},
        }

    def _new_session(self) -> Session:
        if self._session_factory is not None:
            return self._session_factory()
        from src.app.persistence.database import SessionLocal

        return SessionLocal()
