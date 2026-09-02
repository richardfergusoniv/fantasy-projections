"""Cron route path preservation tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from src.app.config import get_settings
from src.app.factory import create_app
from src.app.persistence.database import get_session, init_db
from src.app.persistence.job_outbox import JobOutboxService


def _cron_client(monkeypatch) -> TestClient:
    monkeypatch.setenv("CRON_SECRET", "test-cron-secret")
    monkeypatch.setenv("TRUSTED_HOSTS", "*")
    get_settings.cache_clear()
    init_db()
    return TestClient(create_app())


def test_cron_run_due_route_path(monkeypatch):
    client = _cron_client(monkeypatch)

    response = client.post(
        "/api/internal/cron/run-due",
        headers={"Authorization": "Bearer test-cron-secret"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "results" in body or "enqueued" in body


def test_cron_run_due_rejects_missing_auth(monkeypatch):
    client = _cron_client(monkeypatch)

    response = client.post("/api/internal/cron/run-due")
    assert response.status_code == 401


def test_recover_stale_running_requeues_old_claims(monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "test-cron-secret")
    get_settings.cache_clear()
    init_db()
    with get_session() as session:
        outbox = JobOutboxService(session)
        row = outbox.enqueue("daily_refresh", idempotency_key="stale-test")
        outbox.claim_next()
        outbox.mark_running(row)
        row.started_at = datetime.now(UTC) - timedelta(hours=3)
        session.add(row)
        session.flush()
        recovered = outbox.recover_stale_running(stale_after=timedelta(hours=2))
        assert recovered == 1
        assert row.status == "queued"
        assert row.holder_id is None


def test_recover_stale_running_leaves_fresh_jobs(monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "test-cron-secret")
    get_settings.cache_clear()
    init_db()
    with get_session() as session:
        outbox = JobOutboxService(session)
        row = outbox.enqueue("daily_refresh", idempotency_key="fresh-test")
        outbox.claim_next()
        outbox.mark_running(row)
        recovered = outbox.recover_stale_running(stale_after=timedelta(hours=2))
        assert recovered == 0
        assert row.status == "running"


def test_enqueue_is_idempotent_for_same_slot(monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "test-cron-secret")
    get_settings.cache_clear()
    init_db()
    with get_session() as session:
        outbox = JobOutboxService(session)
        first = outbox.enqueue("daily_refresh", idempotency_key="slot-1")
        second = outbox.enqueue("daily_refresh", idempotency_key="slot-1")
        assert first.id == second.id


def test_process_outbox_empty_queue(monkeypatch):
    from src.app.jobs.scheduler import process_outbox

    monkeypatch.setenv("CRON_SECRET", "test-cron-secret")
    get_settings.cache_clear()
    init_db()
    results = process_outbox(max_jobs=3)
    assert results == []
