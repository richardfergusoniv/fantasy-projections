"""Application tests."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ["APP_ENV"] = "test"
os.environ["APP_ENABLE_DEV_AUTH"] = "true"
os.environ["TEST_DATABASE_URL"] = "sqlite+pysqlite:///:memory:?cache=shared"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_ENABLE_DEV_AUTH", "true")
    from src.app.config import get_settings
    from src.app.factory import create_app
    from src.app.persistence.database import init_db
    from src.app.seed import seed_development_data
    from src.app.persistence.database import get_session

    get_settings.cache_clear()
    init_db()
    with get_session() as session:
        seed_development_data(session, email="owner@example.com")
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_live(client: TestClient):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_magic_link_and_me(client: TestClient):
    response = client.post("/api/v1/auth/magic-link", json={"email": "owner@example.com"})
    assert response.status_code == 200
    dev_link = response.json()["development_link"]
    token = dev_link.split("token=")[-1]
    verify = client.post("/api/v1/auth/verify", json={"token": token})
    assert verify.status_code == 200
    me = client.get("/api/v1/me")
    assert me.status_code == 200
    assert me.json()["email"] == "owner@example.com"


def test_list_leagues_authenticated(client: TestClient):
    client.post("/api/v1/auth/magic-link", json={"email": "owner@example.com"})
    token = client.post("/api/v1/auth/magic-link", json={"email": "owner@example.com"}).json()["development_link"].split("token=")[-1]
    client.post("/api/v1/auth/verify", json={"token": token})
    leagues = client.get("/api/v1/leagues")
    assert leagues.status_code == 200
    assert len(leagues.json()["leagues"]) == 6
