"""Concurrent reads against a file-backed database must not corrupt each other.

FastAPI runs synchronous endpoints in a thread pool, so two requests overlap
routinely — the PWA itself issues `/leagues` and `/operations/status` together on
first paint. The engine used to hand every thread the *same* SQLite connection
(`StaticPool`), so those two requests interleaved on one cursor and whichever
lost the race failed with `IndexError: tuple index out of range` from
SQLAlchemy's result processor, returning a 500.

This reproduces that shape: a real file database (the README's default), several
threads, mixed read endpoints, and an assertion that every response succeeded.
"""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("APP_ENABLE_DEV_AUTH", "true")

READ_PATHS = [
    "/api/v1/leagues",
    "/api/v1/operations/status",
    "/api/v1/leagues/fixture-standard/rules",
    "/api/v1/leagues/fixture-standard/rosters",
    "/api/v1/leagues/fixture-superflex/rules",
    "/api/v1/me",
]


@pytest.fixture()
def file_backed_client(tmp_path: Path, monkeypatch):
    """A client on a real SQLite *file*, not the shared in-memory database."""
    import src.app.persistence.database as database
    from src.app.config import get_settings
    from src.app.factory import create_app
    from src.app.middleware.rate_limit import limiter
    from src.app.seed import seed_development_data

    db_path = tmp_path / f"concurrency-{uuid.uuid4().hex}.db"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_ENABLE_DEV_AUTH", "true")
    monkeypatch.setenv("TEST_DATABASE_URL", f"sqlite+pysqlite:///{db_path.as_posix()}")

    previous_engine = database._engine
    previous_sessionmaker = database._SessionLocal
    database._engine = None
    database._SessionLocal = None
    get_settings.cache_clear()
    limiter.reset()

    try:
        database.init_db()
        with database.get_session() as session:
            seed_development_data(session, email="owner@example.com")

        client = TestClient(create_app())
        link = client.post(
            "/api/v1/auth/magic-link", json={"email": "owner@example.com"}
        ).json()["development_link"]
        client.post("/api/v1/auth/verify", json={"token": link.split("token=")[-1]})
        yield client
    finally:
        # Restore the shared in-memory engine for the rest of the session.
        if database._engine is not None:
            database._engine.dispose()
        database._engine = previous_engine
        database._SessionLocal = previous_sessionmaker
        get_settings.cache_clear()
        limiter.reset()


def test_overlapping_reads_all_succeed(file_backed_client: TestClient):
    requests = READ_PATHS * 6

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(file_backed_client.get, requests))

    failures = [
        (path, response.status_code, response.text[:200])
        for path, response in zip(requests, responses, strict=True)
        if response.status_code != 200
    ]
    assert not failures, failures


def test_a_read_succeeds_while_a_job_is_writing(file_backed_client: TestClient):
    """A refresh holding the write lock must not turn reads into errors."""
    csrf = file_backed_client.post(
        "/api/v1/auth/magic-link", json={"email": "owner@example.com"}
    ).json()["development_link"]
    verify = file_backed_client.post(
        "/api/v1/auth/verify", json={"token": csrf.split("token=")[-1]}
    )
    headers = {
        "X-CSRF-Token": verify.json()["csrf_token"],
        "Idempotency-Key": uuid.uuid4().hex,
    }

    with ThreadPoolExecutor(max_workers=4) as pool:
        sync = pool.submit(
            file_backed_client.post, "/api/v1/sync", headers=headers
        )
        reads = [pool.submit(file_backed_client.get, path) for path in READ_PATHS]
        sync_response = sync.result()
        read_responses = [future.result() for future in reads]

    assert sync_response.status_code == 200, sync_response.text
    assert [r.status_code for r in read_responses] == [200] * len(READ_PATHS)
