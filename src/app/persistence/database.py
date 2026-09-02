"""SQLAlchemy engine and session management."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, StaticPool

from src.app.config import get_settings
from src.app.persistence.base import Base

_engine: Engine | None = None
_job_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None
_JobSessionLocal: sessionmaker[Session] | None = None


#: Seconds a SQLite connection waits for a write lock before raising. File-backed
#: SQLite serializes writers, and the default (5s) is short enough that a long
#: job overlapping a request surfaces as "database is locked".
SQLITE_BUSY_TIMEOUT_SECONDS = 30


def _is_memory_sqlite(url: str) -> bool:
    return url.startswith("sqlite") and ":memory:" in url


def _build_engine(url: str, *, serverless: bool = False) -> Engine:
    connect_args: dict[str, object] = {}
    pool_kwargs: dict[str, object] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        connect_args["timeout"] = SQLITE_BUSY_TIMEOUT_SECONDS
        if _is_memory_sqlite(url):
            pool_kwargs["poolclass"] = StaticPool
    elif serverless:
        pool_kwargs["poolclass"] = NullPool
    engine = create_engine(
        url,
        future=True,
        pool_pre_ping=True,
        connect_args=connect_args,
        **pool_kwargs,
    )
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            if not _is_memory_sqlite(url):
                cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
    return engine


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        _engine = _build_engine(settings.sqlalchemy_url, serverless=settings.use_serverless_db_pool)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def get_job_engine() -> Engine:
    global _job_engine, _JobSessionLocal
    if _job_engine is None:
        settings = get_settings()
        _job_engine = _build_engine(settings.sqlalchemy_job_url, serverless=False)
        _JobSessionLocal = sessionmaker(bind=_job_engine, autoflush=False, autocommit=False)
    return _job_engine


def JobSessionLocal() -> Session:
    get_job_engine()
    assert _JobSessionLocal is not None
    return _JobSessionLocal()


@contextmanager
def get_job_session() -> Generator[Session]:
    session = JobSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def current_alembic_revision() -> str | None:
    try:
        with get_engine().connect() as conn:
            if conn.dialect.name == "sqlite":
                row = conn.execute(text("SELECT version_num FROM alembic_version")).fetchone()
            else:
                row = conn.execute(text("SELECT version_num FROM alembic_version")).fetchone()
            return str(row[0]) if row else None
    except Exception:
        return None


def assert_expected_revision() -> None:
    settings = get_settings()
    if settings.app_env != "production" or not settings.expected_alembic_revision:
        return
    current = current_alembic_revision()
    if current != settings.expected_alembic_revision:
        raise RuntimeError(
            f"Database revision {current!r} does not match expected "
            f"{settings.expected_alembic_revision!r}"
        )


def SessionLocal() -> Session:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal()


@contextmanager
def get_session() -> Generator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    from src.app.persistence import models  # noqa: F401

    Base.metadata.create_all(bind=get_engine())


def reset_engine() -> None:
    """Drop cached engine/session factory so a new DATABASE_URL can take effect."""
    global _engine, _SessionLocal, _job_engine, _JobSessionLocal
    if _engine is not None:
        _engine.dispose()
    if _job_engine is not None:
        _job_engine.dispose()
    _engine = None
    _SessionLocal = None
    _job_engine = None
    _JobSessionLocal = None
