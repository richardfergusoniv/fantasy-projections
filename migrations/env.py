"""Alembic migration environment."""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from src.app.persistence.base import Base
from src.app.persistence import models  # noqa: F401

config = context.config
if config.config_file_name is not None:
    # `disable_existing_loggers` defaults to True, which would silently disable
    # every `logging.getLogger(__name__)` created when `src.app.*` was imported.
    # Running migrations in-process (app-migrate, tests) must not mute the app.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        config.get_main_option("sqlalchemy.url", "postgresql+psycopg://fantasy:fantasy@localhost:5432/fantasy_app"),
    )


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {}) or {}
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        # `render_as_batch` makes autogenerate emit SQLite-compatible
        # copy-and-rebuild operations, so a future schema change does not
        # silently produce a migration SQLite cannot apply.
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
