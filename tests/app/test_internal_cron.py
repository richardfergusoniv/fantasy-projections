"""Cron route path preservation tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.app.config import get_settings
from src.app.factory import create_app
from src.app.persistence.database import init_db


def test_cron_run_due_route_path(monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "test-cron-secret")
    monkeypatch.setenv("TRUSTED_HOSTS", "*")
    get_settings.cache_clear()
    init_db()
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/internal/cron/run-due",
        headers={"Authorization": "Bearer test-cron-secret"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "results" in body or "enqueued" in body


def test_cron_run_due_rejects_missing_auth(monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "test-cron-secret")
    monkeypatch.setenv("TRUSTED_HOSTS", "*")
    get_settings.cache_clear()
    init_db()
    app = create_app()
    client = TestClient(app)

    response = client.post("/api/internal/cron/run-due")
    assert response.status_code == 401
