"""Repository-wide pytest configuration."""

from __future__ import annotations

import os
import uuid

import pytest


def pytest_configure(config) -> None:
    """Establish isolated test defaults before any Settings() is constructed."""
    os.environ.setdefault("APP_ENV", "test")
    os.environ.setdefault("APP_ENABLE_DEV_AUTH", "true")
    os.environ.setdefault("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:")
    os.environ.setdefault("TRUSTED_HOSTS", "testserver,localhost,127.0.0.1,*")
    os.environ.setdefault("APP_ALLOWED_EMAIL", "owner@example.com")
    os.environ.setdefault("EMAIL_PROVIDER", "development")
    os.environ.setdefault("SLEEPER_USE_FIXTURES", "true")
    os.environ.setdefault("INJURY_RESEARCH_MODE", "fixture")


@pytest.fixture(autouse=True)
def _isolated_test_database(monkeypatch):
    """Give each test its own in-memory SQLite database."""
    from src.app.config import get_settings
    from src.app.persistence.database import reset_engine

    db_url = f"sqlite+pysqlite:///file:test_{uuid.uuid4().hex}?mode=memory&cache=shared&uri=true"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("TEST_DATABASE_URL", db_url)
    get_settings.cache_clear()
    reset_engine()
    yield
    reset_engine()
