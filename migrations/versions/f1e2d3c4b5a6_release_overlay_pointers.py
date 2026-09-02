"""Release/overlay pointers, job lease/outbox, and rate-limit buckets.

Revision ID: f1e2d3c4b5a6
Revises: d4a1f6c28b57
Create Date: 2026-09-01
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "f1e2d3c4b5a6"
down_revision = "d4a1f6c28b57"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "release_pointer",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("season", sa.Integer(), nullable=False),
        sa.Column("namespace", sa.String(length=128), nullable=False),
        sa.Column("release_id", sa.String(length=64), nullable=False),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("manifest_storage_uri", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
        sa.Column("pointer_json", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_release_pointer_season", "release_pointer", ["season"], unique=True)

    op.create_table(
        "release_pointer_history",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("season", sa.Integer(), nullable=False),
        sa.Column("pointer_json", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("reason", sa.String(length=64), server_default="promote", nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_release_pointer_history_season", "release_pointer_history", ["season"])

    op.create_table(
        "status_overlay_pointer",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("season", sa.Integer(), nullable=False),
        sa.Column("overlay_hash", sa.String(length=64), nullable=False),
        sa.Column("base_release_id", sa.String(length=64), nullable=False),
        sa.Column("base_manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("adjustment_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("algorithm_version", sa.String(length=64), nullable=False),
        sa.Column("pointer_json", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_status_overlay_pointer_season",
        "status_overlay_pointer",
        ["season"],
        unique=True,
    )

    op.create_table(
        "status_overlay_pointer_history",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("season", sa.Integer(), nullable=False),
        sa.Column("pointer_json", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("reason", sa.String(length=64), server_default="promote", nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_status_overlay_pointer_history_season", "status_overlay_pointer_history", ["season"])

    op.create_table(
        "job_lease",
        sa.Column("job_name", sa.String(length=64), nullable=False),
        sa.Column("holder_id", sa.String(length=64), nullable=False),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("job_name"),
    )

    op.create_table(
        "job_outbox",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("job_name", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="queued", nullable=False),
        sa.Column("holder_id", sa.String(length=64), nullable=True),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempt", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key"),
    )
    op.create_index("ix_job_outbox_job_name", "job_outbox", ["job_name"])

    op.create_table(
        "rate_limit_bucket",
        sa.Column("bucket_key", sa.String(length=256), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("bucket_key"),
    )


def downgrade() -> None:
    op.drop_table("rate_limit_bucket")
    op.drop_index("ix_job_outbox_job_name", table_name="job_outbox")
    op.drop_table("job_outbox")
    op.drop_table("job_lease")
    op.drop_index("ix_status_overlay_pointer_history_season", table_name="status_overlay_pointer_history")
    op.drop_table("status_overlay_pointer_history")
    op.drop_index("ix_status_overlay_pointer_season", table_name="status_overlay_pointer")
    op.drop_table("status_overlay_pointer")
    op.drop_index("ix_release_pointer_history_season", table_name="release_pointer_history")
    op.drop_table("release_pointer_history")
    op.drop_index("ix_release_pointer_season", table_name="release_pointer")
    op.drop_table("release_pointer")
