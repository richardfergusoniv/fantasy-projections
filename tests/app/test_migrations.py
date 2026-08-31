"""Alembic migration smoke tests.

These run against SQLite because that is what the environment provides. The
schema they assert is the one PostgreSQL gets: revision ``d4a1f6c28b57`` brings
the SQLite path up to parity precisely so these assertions mean something. Where
a behaviour genuinely cannot be proven on SQLite it is called out rather than
implied.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.exc import IntegrityError

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Composed incremental run ids are ~43 characters. The initial migration typed
#: these columns ``VARCHAR(36)``, which SQLite ignores and PostgreSQL rejects.
LONG_RUN_ID = "weekly-2026-w01-538cf955e04c-inc-7304e071a8"
RUN_ID_COLUMNS = {
    "projection_run": "id",
    "player_projection": "run_id",
    "simulation_partition": "run_id",
    "active_projection_pointer": "run_id",
    "promotion_event": "candidate_run_id",
    "decision_snapshot": "projection_run_id",
    "trade_evaluation": "projection_run_id",
}


def _migrated_engine(tmp_path: Path, monkeypatch, *, revision: str = "head"):
    from alembic import command
    from alembic.config import Config

    db_path = tmp_path / f"migrate-{uuid.uuid4().hex}.db"
    database_url = f"sqlite+pysqlite:///{db_path.as_posix()}"
    # monkeypatch, not os.environ: a leaked DATABASE_URL would silently
    # repoint every later test in the session at this temporary database.
    monkeypatch.setenv("DATABASE_URL", database_url)

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(cfg, revision)

    engine = create_engine(database_url)

    # The application engine turns this on for every connection; migrations run
    # with it off (which is what makes the SQLite table rebuild safe), so the
    # test has to enable it to observe the constraints at all.
    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine, cfg


def test_alembic_upgrade_head(tmp_path: Path, monkeypatch):
    engine, _cfg = _migrated_engine(tmp_path, monkeypatch)
    tables = set(inspect(engine).get_table_names())
    assert "app_user" in tables
    assert "projection_run" in tables
    assert "active_projection_pointer" in tables


def test_run_id_columns_are_wide_enough_for_composed_run_ids(tmp_path: Path, monkeypatch):
    """A 43-character incremental run id must fit every column that stores one."""
    engine, _cfg = _migrated_engine(tmp_path, monkeypatch)
    inspector = inspect(engine)

    for table, column in RUN_ID_COLUMNS.items():
        info = next(col for col in inspector.get_columns(table) if col["name"] == column)
        length = getattr(info["type"], "length", None)
        assert length is not None and length >= len(LONG_RUN_ID), (
            f"{table}.{column} is {length} chars; a composed run id is "
            f"{len(LONG_RUN_ID)} and PostgreSQL would reject the insert"
        )


def test_decision_snapshot_cannot_reference_a_missing_projection_run(
    tmp_path: Path, monkeypatch
):
    """Referential integrity, not just a column that happens to hold a string."""
    engine, _cfg = _migrated_engine(tmp_path, monkeypatch)

    with engine.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(
                text(
                    "INSERT INTO decision_snapshot "
                    "(id, kind, league_id, week, projection_run_id, result_json, created_at) "
                    "VALUES (:id, 'lineup', 'fixture-standard', 1, :run, '{}', :now)"
                ),
                {
                    "id": uuid.uuid4().hex,
                    "run": "run-that-does-not-exist",
                    "now": datetime.now(UTC).isoformat(),
                },
            )


def test_a_real_run_id_round_trips_through_the_pointer(tmp_path: Path, monkeypatch):
    engine, _cfg = _migrated_engine(tmp_path, monkeypatch)
    now = datetime.now(UTC).isoformat()

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO projection_run "
                "(id, mode, season, week, as_of, model_version, input_hash, status) "
                "VALUES (:id, 'weekly', 2026, 1, :now, 'v2', 'hash', 'active')"
            ),
            {"id": LONG_RUN_ID, "now": now},
        )
        conn.execute(
            text(
                "INSERT INTO active_projection_pointer "
                "(id, mode, season, week, run_id, activated_at) "
                "VALUES (:id, 'weekly', 2026, 1, :run, :now)"
            ),
            {"id": uuid.uuid4().hex, "run": LONG_RUN_ID, "now": now},
        )
        stored = conn.execute(
            text("SELECT run_id FROM active_projection_pointer")
        ).scalar_one()

    assert stored == LONG_RUN_ID


def test_only_one_season_long_pointer_may_be_active_per_mode(tmp_path: Path, monkeypatch):
    """A NULL week must not let two ROS pointers both claim to be active."""
    engine, _cfg = _migrated_engine(tmp_path, monkeypatch)
    now = datetime.now(UTC).isoformat()

    with engine.begin() as conn:
        for run_id in ("ros-a", "ros-b"):
            conn.execute(
                text(
                    "INSERT INTO projection_run "
                    "(id, mode, season, week, as_of, model_version, input_hash, status) "
                    "VALUES (:id, 'ros', 2026, NULL, :now, 'v2', 'hash', 'active')"
                ),
                {"id": run_id, "now": now},
            )
        conn.execute(
            text(
                "INSERT INTO active_projection_pointer "
                "(id, mode, season, week, run_id, activated_at) "
                "VALUES (:id, 'ros', 2026, NULL, 'ros-a', :now)"
            ),
            {"id": uuid.uuid4().hex, "now": now},
        )
        with pytest.raises(IntegrityError):
            conn.execute(
                text(
                    "INSERT INTO active_projection_pointer "
                    "(id, mode, season, week, run_id, activated_at) "
                    "VALUES (:id, 'ros', 2026, NULL, 'ros-b', :now)"
                ),
                {"id": uuid.uuid4().hex, "now": now},
            )


def test_downgrade_and_reupgrade_round_trips(tmp_path: Path, monkeypatch):
    """A rollback of the newest revision must not strand the database."""
    from alembic import command

    engine, cfg = _migrated_engine(tmp_path, monkeypatch)
    command.downgrade(cfg, "-1")
    command.upgrade(cfg, "head")

    tables = set(inspect(engine).get_table_names())
    assert "projection_run" in tables
    inspector = inspect(engine)
    info = next(
        col for col in inspector.get_columns("projection_run") if col["name"] == "id"
    )
    assert getattr(info["type"], "length", None) >= len(LONG_RUN_ID)
