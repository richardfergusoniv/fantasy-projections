"""SQLAlchemy engine and session management."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.app.config import get_settings
from src.app.persistence.base import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


#: Seconds a SQLite connection waits for a write lock before raising. File-backed
#: SQLite serializes writers, and the default (5s) is short enough that a long
#: job overlapping a request surfaces as "database is locked".
SQLITE_BUSY_TIMEOUT_SECONDS = 30


def _is_memory_sqlite(url: str) -> bool:
    return url.startswith("sqlite") and ":memory:" in url


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        url = settings.sqlalchemy_url
        connect_args: dict[str, object] = {}
        pool_kwargs: dict[str, object] = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            connect_args["timeout"] = SQLITE_BUSY_TIMEOUT_SECONDS
            if _is_memory_sqlite(url):
                # An in-memory database only exists for as long as its
                # connection, so every session has to share one. That is safe
                # here because it is the test configuration.
                pool_kwargs["poolclass"] = StaticPool
            # A file-backed database deliberately gets the normal pool. Sharing
            # one connection across FastAPI's request threadpool let two
            # concurrent requests interleave on the same cursor, which surfaced
            # as `IndexError: tuple index out of range` from the result
            # processor and a 500 on whichever request lost the race.
        _engine = create_engine(
            url,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
            **pool_kwargs,
        )
        if url.startswith("sqlite"):

            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                # WAL lets a reader run while a writer holds the write lock,
                # which is what a request concurrent with a job needs. It is a
                # no-op for :memory:.
                if not _is_memory_sqlite(url):
                    cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()

        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


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
