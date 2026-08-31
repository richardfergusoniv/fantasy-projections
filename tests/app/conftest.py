"""Shared fixtures for app tests."""

from __future__ import annotations

import pytest


@pytest.fixture()
def db_session(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:?cache=shared")
    from src.app.config import get_settings
    from src.app.persistence.database import SessionLocal, init_db

    get_settings.cache_clear()
    init_db()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    finally:
        session.close()
