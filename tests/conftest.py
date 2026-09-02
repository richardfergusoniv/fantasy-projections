"""Repository-wide pytest configuration."""

from __future__ import annotations

import os
import uuid

import pytest


def _polars_runtime_available() -> bool:
    try:
        import polars as pl

        pl.DataFrame({"x": [1]})
        return True
    except Exception:
        return False


POLARS_AVAILABLE = _polars_runtime_available()


def pytest_configure(config) -> None:
    """Establish isolated test defaults before any Settings() is constructed."""
    os.environ.pop("MIGRATION_DATABASE_URL", None)
    os.environ.setdefault("APP_ENV", "test")
    os.environ.setdefault("APP_ENABLE_DEV_AUTH", "true")
    os.environ.setdefault("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:")
    os.environ.setdefault("TRUSTED_HOSTS", "testserver,localhost,127.0.0.1,*")
    os.environ.setdefault("APP_ALLOWED_EMAIL", "owner@example.com")
    os.environ.setdefault("EMAIL_PROVIDER", "development")
    os.environ.setdefault("SLEEPER_USE_FIXTURES", "true")
    os.environ.setdefault("INJURY_RESEARCH_MODE", "fixture")
    os.environ.setdefault("WEEKLY_RND_ENABLED", "false")


def pytest_collection_modifyitems(config, items) -> None:
    if POLARS_AVAILABLE:
        return
    skip_polars = pytest.mark.skip(reason="polars runtime unavailable on this platform")
    for item in items:
        if "weekly" in item.nodeid and "polars" in (item.module.__doc__ or ""):
            item.add_marker(skip_polars)
        if "weekly_event_cohort" in item.nodeid or "weekly_v2_tuning" in item.nodeid:
            item.add_marker(skip_polars)


@pytest.fixture(autouse=True)
def _isolated_test_database(monkeypatch):
    """Give each test its own in-memory SQLite database."""
    from src.app.config import get_settings
    from src.app.persistence.database import reset_engine

    db_url = f"sqlite+pysqlite:///file:test_{uuid.uuid4().hex}?mode=memory&cache=shared&uri=true"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.delenv("MIGRATION_DATABASE_URL", raising=False)
    monkeypatch.setenv("TEST_DATABASE_URL", db_url)
    get_settings.cache_clear()
    reset_engine()
    yield
    reset_engine()
