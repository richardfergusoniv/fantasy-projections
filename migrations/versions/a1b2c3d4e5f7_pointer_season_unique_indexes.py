"""Align pointer season indexes with SQLAlchemy unique index metadata.

Revision ID: a1b2c3d4e5f7
Revises: f1e2d3c4b5a6
Create Date: 2026-09-02
"""

from __future__ import annotations

from alembic import op

revision = "a1b2c3d4e5f7"
down_revision = "f1e2d3c4b5a6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect == "postgresql":
        op.drop_constraint("release_pointer_season_key", "release_pointer", type_="unique")
        op.drop_constraint(
            "status_overlay_pointer_season_key",
            "status_overlay_pointer",
            type_="unique",
        )
    op.drop_index("ix_release_pointer_season", table_name="release_pointer")
    op.create_index(
        "ix_release_pointer_season",
        "release_pointer",
        ["season"],
        unique=True,
    )
    op.drop_index("ix_status_overlay_pointer_season", table_name="status_overlay_pointer")
    op.create_index(
        "ix_status_overlay_pointer_season",
        "status_overlay_pointer",
        ["season"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_status_overlay_pointer_season", table_name="status_overlay_pointer")
    op.create_index(
        "ix_status_overlay_pointer_season",
        "status_overlay_pointer",
        ["season"],
        unique=False,
    )
    op.drop_index("ix_release_pointer_season", table_name="release_pointer")
    op.create_index(
        "ix_release_pointer_season",
        "release_pointer",
        ["season"],
        unique=False,
    )
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.create_unique_constraint(
            "status_overlay_pointer_season_key",
            "status_overlay_pointer",
            ["season"],
        )
        op.create_unique_constraint(
            "release_pointer_season_key",
            "release_pointer",
            ["season"],
        )
