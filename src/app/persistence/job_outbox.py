"""Durable job outbox and lease management for serverless-safe scheduling."""

from __future__ import annotations

import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.app.persistence.models import JobLease, JobOutbox, utcnow

DEFAULT_LEASE_SECONDS = 300
OUTBOX_CLAIM_LEASE_SECONDS = 600


def _holder_id() -> str:
    return secrets.token_hex(8)


class JobLeaseManager:
    def __init__(self, session: Session, *, lease_seconds: int = DEFAULT_LEASE_SECONDS) -> None:
        self.session = session
        self.lease_seconds = lease_seconds

    def acquire(self, job_name: str) -> bool:
        holder = _holder_id()
        now = utcnow()
        lease_until = now + timedelta(seconds=self.lease_seconds)
        bind = self.session.get_bind()
        if bind.dialect.name != "postgresql":
            return True
        result = self.session.execute(
            text(
                """
                INSERT INTO job_lease (job_name, holder_id, lease_until, updated_at)
                VALUES (:job_name, :holder_id, :lease_until, :updated_at)
                ON CONFLICT (job_name) DO UPDATE
                SET holder_id = EXCLUDED.holder_id,
                    lease_until = EXCLUDED.lease_until,
                    updated_at = EXCLUDED.updated_at
                WHERE job_lease.lease_until < :now
                """
            ),
            {
                "job_name": job_name,
                "holder_id": holder,
                "lease_until": lease_until,
                "updated_at": now,
                "now": now,
            },
        )
        self.session.flush()
        self._holder = holder
        return result.rowcount > 0

    def release(self, job_name: str) -> None:
        bind = self.session.get_bind()
        if bind.dialect.name != "postgresql":
            return
        holder = getattr(self, "_holder", None)
        if holder is None:
            return
        self.session.execute(
            text(
                "DELETE FROM job_lease WHERE job_name = :job_name AND holder_id = :holder_id"
            ),
            {"job_name": job_name, "holder_id": holder},
        )


class JobOutboxService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def enqueue(
        self,
        job_name: str,
        *,
        idempotency_key: str,
        scheduled_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JobOutbox:
        existing = (
            self.session.query(JobOutbox)
            .filter(JobOutbox.idempotency_key == idempotency_key)
            .one_or_none()
        )
        if existing is not None:
            return existing
        row = JobOutbox(
            id=str(uuid.uuid4()),
            job_name=job_name,
            idempotency_key=idempotency_key,
            status="queued",
            scheduled_at=scheduled_at,
            metadata_json=dict(metadata or {}),
        )
        self.session.add(row)
        self.session.flush()
        return row

    def claim_next(self, *, job_name: str | None = None) -> JobOutbox | None:
        now = utcnow()
        query = (
            self.session.query(JobOutbox)
            .filter(JobOutbox.status.in_(("queued", "failed")))
            .order_by(JobOutbox.created_at.asc())
        )
        if job_name is not None:
            query = query.filter(JobOutbox.job_name == job_name)
        for row in query.limit(20):
            if row.scheduled_at is not None and row.scheduled_at > now:
                continue
            holder = _holder_id()
            row.status = "claimed"
            row.holder_id = holder
            row.claimed_at = now
            row.attempt = (row.attempt or 0) + 1
            self.session.add(row)
            self.session.flush()
            return row
        return None

    def mark_running(self, row: JobOutbox) -> None:
        row.status = "running"
        row.started_at = utcnow()
        self.session.add(row)
        self.session.flush()

    def mark_succeeded(self, row: JobOutbox, *, metadata: dict[str, Any] | None = None) -> None:
        row.status = "succeeded"
        row.finished_at = utcnow()
        if metadata:
            row.metadata_json = {**(row.metadata_json or {}), **metadata}
        self.session.add(row)
        self.session.flush()

    def mark_failed(self, row: JobOutbox, error: str) -> None:
        row.status = "failed"
        row.finished_at = utcnow()
        row.error = error[:4000]
        self.session.add(row)
        self.session.flush()

    def recover_stale_running(self, *, stale_after: timedelta) -> int:
        cutoff = utcnow() - stale_after
        rows = (
            self.session.query(JobOutbox)
            .filter(JobOutbox.status.in_(("claimed", "running")), JobOutbox.started_at < cutoff)
            .all()
        )
        for row in rows:
            row.status = "queued"
            row.holder_id = None
            row.claimed_at = None
            row.started_at = None
            self.session.add(row)
        self.session.flush()
        return len(rows)
