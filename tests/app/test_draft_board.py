"""Draft board and rate limit tests."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("APP_ENABLE_DEV_AUTH", "true")
os.environ.setdefault("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:?cache=shared")


def test_draft_board_returns_release_players(db_session):
    from src.app.decisions.draft_board import DraftBoardService

    service = DraftBoardService()
    board = service.load_board(2026, limit=20)
    if not board["entries"]:
        pytest.skip("no release bundle available")
    assert len(board["entries"]) <= 20
    assert board["entries"][0]["rank"] == 1
    assert board["entries"][0]["name"]


def test_rate_limit_blocks_excess_auth_requests():
    from src.app.middleware.rate_limit import RateLimiter

    limiter = RateLimiter()
    for _ in range(3):
        limiter.check("auth:test", limit=3)
    with pytest.raises(Exception) as exc:
        limiter.check("auth:test", limit=3)
    assert exc.value.status_code == 429


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_ENABLE_DEV_AUTH", "true")
    monkeypatch.setenv("AUTH_RATE_LIMIT_PER_MINUTE", "100")
    from src.app.config import get_settings
    from src.app.factory import create_app
    from src.app.persistence.database import get_session, init_db
    from src.app.seed import seed_development_data

    get_settings.cache_clear()
    init_db()
    with get_session() as session:
        seed_development_data(session, email="owner@example.com")
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_draft_board_endpoint(client: TestClient):
    token = client.post("/api/v1/auth/magic-link", json={"email": "owner@example.com"}).json()[
        "development_link"
    ].split("token=")[-1]
    client.post("/api/v1/auth/verify", json={"token": token})
    response = client.get("/api/v1/leagues/fixture-standard/draft/board")
    assert response.status_code == 200
    body = response.json()
    if body.get("entries"):
        assert body["entries"][0]["player_id"]
