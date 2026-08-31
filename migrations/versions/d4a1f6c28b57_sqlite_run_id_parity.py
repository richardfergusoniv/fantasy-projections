"""SQLite parity for run-id widths and projection-run foreign keys

Revision ID: d4a1f6c28b57
Revises: b7c41d92f0aa
Create Date: 2026-08-31

Why this exists
---------------
``b7c41d92f0aa`` widened the run-id columns and added the missing references
into ``projection_run`` on PostgreSQL only, because plain ``ALTER TABLE`` cannot
express either change on SQLite. That left two problems:

* ``uv run alembic check`` — a documented verification step — failed against a
  SQLite database, reporting nine type differences and five missing foreign
  keys. A failing schema-drift check cannot distinguish real drift from this
  known gap.
* The SQLite test path never enforced the ``projection_run`` references, so no
  test could prove that a decision, trade evaluation, or promotion event is
  unable to point at a run that does not exist.

``batch_alter_table`` performs SQLite's copy-and-rebuild, so both databases end
up with the same schema and the drift check is meaningful on either. On
PostgreSQL this revision is a no-op: ``b7c41d92f0aa`` already made those
changes.

Server defaults remain PostgreSQL-only, as documented in ``b7c41d92f0aa``: a
SQLite database created from the ORM metadata already carries them, and
rebuilding forty-five columns to add them would buy nothing.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d4a1f6c28b57"
down_revision: str | Sequence[str] | None = "b7c41d92f0aa"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUN_ID_LEN = 128
OLD_RUN_ID_LEN = 36

#: table -> ((column, nullable), ...) for every column holding a run id.
RUN_ID_COLUMNS: dict[str, tuple[tuple[str, bool], ...]] = {
    "projection_run": (("id", False),),
    "player_projection": (("run_id", False),),
    "simulation_partition": (("run_id", False),),
    "active_projection_pointer": (("run_id", False), ("previous_run_id", True)),
    "promotion_event": (("candidate_run_id", False), ("previous_run_id", True)),
    "decision_snapshot": (("projection_run_id", False),),
    "trade_evaluation": (("projection_run_id", False),),
}

#: table -> ((constraint name, column), ...) missing from the SQLite schema.
RUN_FOREIGN_KEYS: dict[str, tuple[tuple[str, str], ...]] = {
    "decision_snapshot": (("fk_decision_snapshot_projection_run", "projection_run_id"),),
    "trade_evaluation": (("fk_trade_evaluation_projection_run", "projection_run_id"),),
    "promotion_event": (
        ("fk_promotion_event_candidate_run", "candidate_run_id"),
        ("fk_promotion_event_previous_run", "previous_run_id"),
    ),
    "active_projection_pointer": (
        ("fk_active_pointer_previous_run", "previous_run_id"),
    ),
}

#: Partial unique index that batch reflection cannot round-trip, so it is
#: dropped and recreated explicitly around the rebuild.
PARTIAL_INDEX = "uq_active_pointer_season_long"


def _is_sqlite() -> bool:
    return op.get_bind().dialect.name == "sqlite"


def _create_partial_index() -> None:
    op.create_index(
        PARTIAL_INDEX,
        "active_projection_pointer",
        ["mode", "season"],
        unique=True,
        sqlite_where=sa.text("week IS NULL"),
        postgresql_where=sa.text("week IS NULL"),
    )


def upgrade() -> None:
    if not _is_sqlite():
        return

    op.drop_index(PARTIAL_INDEX, table_name="active_projection_pointer")
    for table, columns in RUN_ID_COLUMNS.items():
        with op.batch_alter_table(table) as batch:
            for column, nullable in columns:
                batch.alter_column(
                    column,
                    existing_type=sa.String(length=OLD_RUN_ID_LEN),
                    type_=sa.String(length=RUN_ID_LEN),
                    existing_nullable=nullable,
                )
            for name, column in RUN_FOREIGN_KEYS.get(table, ()):
                batch.create_foreign_key(name, "projection_run", [column], ["id"])
    _create_partial_index()


def downgrade() -> None:
    if not _is_sqlite():
        return

    op.drop_index(PARTIAL_INDEX, table_name="active_projection_pointer")
    for table, columns in reversed(list(RUN_ID_COLUMNS.items())):
        with op.batch_alter_table(table) as batch:
            for name, _column in RUN_FOREIGN_KEYS.get(table, ()):
                batch.drop_constraint(name, type_="foreignkey")
            for column, nullable in columns:
                batch.alter_column(
                    column,
                    existing_type=sa.String(length=RUN_ID_LEN),
                    type_=sa.String(length=OLD_RUN_ID_LEN),
                    existing_nullable=nullable,
                )
    _create_partial_index()
