"""publication integrity: uniqueness, referential integrity, server defaults

Revision ID: b7c41d92f0aa
Revises: e53ebac3a6e5
Create Date: 2026-08-30

Portability
-----------
Everything in the "portable" section runs identically on PostgreSQL (the
production target) and SQLite (the test target): ``ADD COLUMN`` and
``CREATE UNIQUE INDEX`` — including partial indexes — are supported by both.

The "PostgreSQL only" section covers changes SQLite's ``ALTER TABLE`` cannot
express without a full table rebuild:

* widening the run-id columns from ``VARCHAR(36)`` to ``VARCHAR(128)``. Composed
  run ids such as ``weekly-2026-w01-<hash>-inc-<hash>`` are 43 characters, which
  SQLite silently accepts and PostgreSQL rejects. SQLite does not enforce
  ``VARCHAR`` length, so skipping it there changes nothing observable.
* ``ALTER COLUMN ... SET DEFAULT``. The ORM metadata carries the same
  ``server_default`` values, so a SQLite database created with
  ``Base.metadata.create_all()`` already has them; only the SQLite *migration*
  path lacks them.
* ``ADD CONSTRAINT ... FOREIGN KEY``. SQLite would need a table rebuild.

A SQLite run of this migration therefore proves the portable half only.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7c41d92f0aa"
down_revision: str | Sequence[str] | None = "e53ebac3a6e5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RUN_ID_LEN = 128

#: ``(table, column)`` pairs holding a projection run id.
RUN_ID_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("projection_run", "id", False),
    ("player_projection", "run_id", False),
    ("simulation_partition", "run_id", False),
    ("active_projection_pointer", "run_id", False),
    ("active_projection_pointer", "previous_run_id", True),
    ("promotion_event", "candidate_run_id", False),
    ("promotion_event", "previous_run_id", True),
    ("decision_snapshot", "projection_run_id", False),
    ("trade_evaluation", "projection_run_id", False),
)

#: NOT NULL columns whose only default lived in Python, so any non-ORM insert
#: (psql, a data-loader, another service) violated the constraint.
SERVER_DEFAULTS: tuple[tuple[str, str, sa.types.TypeEngine, str], ...] = (
    ("app_user", "created_at", sa.DateTime(timezone=True), "now()"),
    ("magic_link_token", "created_at", sa.DateTime(timezone=True), "now()"),
    ("session_record", "created_at", sa.DateTime(timezone=True), "now()"),
    ("league", "status", sa.String(32), "'active'"),
    ("league", "raw_json", sa.JSON(), "'{}'"),
    ("league_draft_rule", "confirmed_at", sa.DateTime(timezone=True), "now()"),
    ("league_rule_snapshot", "raw_json", sa.JSON(), "'{}'"),
    ("league_rule_snapshot", "normalized_json", sa.JSON(), "'{}'"),
    ("roster_snapshot", "players", sa.JSON(), "'[]'"),
    ("roster_snapshot", "starters", sa.JSON(), "'[]'"),
    ("roster_snapshot", "reserve", sa.JSON(), "'[]'"),
    ("league_transaction", "payload", sa.JSON(), "'{}'"),
    ("player_status_snapshot", "raw_json", sa.JSON(), "'{}'"),
    ("injury_evidence", "claim_json", sa.JSON(), "'{}'"),
    ("injury_evidence", "confidence", sa.Float(), "0.5"),
    ("availability_event", "evidence_ids", sa.JSON(), "'[]'"),
    ("availability_event", "policy_json", sa.JSON(), "'{}'"),
    ("projection_run", "status", sa.String(32), "'candidate'"),
    ("player_projection", "mean_json", sa.JSON(), "'{}'"),
    ("player_projection", "quantiles_json", sa.JSON(), "'{}'"),
    ("active_projection_pointer", "activated_at", sa.DateTime(timezone=True), "now()"),
    ("decision_snapshot", "result_json", sa.JSON(), "'{}'"),
    ("decision_snapshot", "created_at", sa.DateTime(timezone=True), "now()"),
    ("manager_state", "probabilities_json", sa.JSON(), "'{}'"),
    ("manager_state", "features_json", sa.JSON(), "'{}'"),
    ("trade_proposal", "created_at", sa.DateTime(timezone=True), "now()"),
    ("trade_proposal", "sides_json", sa.JSON(), "'{}'"),
    ("trade_proposal", "status", sa.String(32), "'offered'"),
    ("trade_evaluation", "objective_json", sa.JSON(), "'{}'"),
    ("trade_evaluation", "fairness_json", sa.JSON(), "'{}'"),
    ("trade_evaluation", "acceptance_json", sa.JSON(), "'{}'"),
    ("manager_tendency", "sample_size", sa.Integer(), "0"),
    ("manager_tendency", "features_json", sa.JSON(), "'{}'"),
    ("job_run", "status", sa.String(32), "'running'"),
    ("job_run", "attempt", sa.Integer(), "1"),
    ("job_run", "started_at", sa.DateTime(timezone=True), "now()"),
    ("job_run", "metadata_json", sa.JSON(), "'{}'"),
    ("source_snapshot", "health_verdict", sa.String(32), "'healthy'"),
    ("source_snapshot", "is_complete", sa.Boolean(), "true"),
    ("promotion_event", "promoted", sa.Boolean(), "false"),
    ("promotion_event", "validation_json", sa.JSON(), "'{}'"),
    ("promotion_event", "created_at", sa.DateTime(timezone=True), "now()"),
    ("assistant_audit", "tools_called", sa.JSON(), "'[]'"),
    ("assistant_audit", "source_ids", sa.JSON(), "'[]'"),
    ("assistant_audit", "token_usage", sa.JSON(), "'{}'"),
    ("assistant_audit", "created_at", sa.DateTime(timezone=True), "now()"),
)

#: ``(name, source table, source column, nullable)`` for the missing references
#: into ``projection_run``. Every writer in the application creates the run row
#: before the referencing row, so these are safe to enforce.
RUN_FOREIGN_KEYS: tuple[tuple[str, str, str], ...] = (
    ("fk_decision_snapshot_projection_run", "decision_snapshot", "projection_run_id"),
    ("fk_trade_evaluation_projection_run", "trade_evaluation", "projection_run_id"),
    ("fk_promotion_event_candidate_run", "promotion_event", "candidate_run_id"),
    ("fk_promotion_event_previous_run", "promotion_event", "previous_run_id"),
    ("fk_active_pointer_previous_run", "active_projection_pointer", "previous_run_id"),
)


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    # ------------------------------------------------------------- portable
    op.add_column(
        "projection_run",
        sa.Column("artifact_mode", sa.String(length=32), nullable=True, server_default="derived"),
    )
    op.create_index(
        "uq_player_projection_run_player",
        "player_projection",
        ["run_id", "player_id"],
        unique=True,
    )
    op.create_index(
        "uq_simulation_partition_run_key",
        "simulation_partition",
        ["run_id", "partition_key"],
        unique=True,
    )
    # ``uq_active_pointer`` (mode, season, week) does not constrain season-long
    # horizons because NULL weeks compare as distinct, so multiple rows could
    # claim to be the active ROS/dynasty/preseason pointer at once. A partial
    # unique index over the NULL-week rows closes that without a sentinel week.
    op.create_index(
        "uq_active_pointer_season_long",
        "active_projection_pointer",
        ["mode", "season"],
        unique=True,
        sqlite_where=sa.text("week IS NULL"),
        postgresql_where=sa.text("week IS NULL"),
    )

    if not _is_postgres():
        return

    # -------------------------------------------------------- PostgreSQL only
    for table, column, nullable in RUN_ID_COLUMNS:
        op.alter_column(
            table,
            column,
            existing_type=sa.String(length=36),
            type_=sa.String(length=RUN_ID_LEN),
            existing_nullable=nullable,
        )
    for table, column, column_type, default in SERVER_DEFAULTS:
        op.alter_column(
            table, column, existing_type=column_type, server_default=sa.text(default)
        )
    for name, table, column in RUN_FOREIGN_KEYS:
        op.create_foreign_key(name, table, "projection_run", [column], ["id"])


def downgrade() -> None:
    if _is_postgres():
        for name, table, _column in reversed(RUN_FOREIGN_KEYS):
            op.drop_constraint(name, table, type_="foreignkey")
        for table, column, column_type, _default in reversed(SERVER_DEFAULTS):
            op.alter_column(table, column, existing_type=column_type, server_default=None)
        for table, column, nullable in reversed(RUN_ID_COLUMNS):
            op.alter_column(
                table,
                column,
                existing_type=sa.String(length=RUN_ID_LEN),
                type_=sa.String(length=36),
                existing_nullable=nullable,
            )

    op.drop_index("uq_active_pointer_season_long", table_name="active_projection_pointer")
    op.drop_index("uq_simulation_partition_run_key", table_name="simulation_partition")
    op.drop_index("uq_player_projection_run_player", table_name="player_projection")
    op.drop_column("projection_run", "artifact_mode")
