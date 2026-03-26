"""add scheduled sync tables

Revision ID: f4a5b6c7d8e9
Revises: b2d3e4f5a6c7
Create Date: 2026-03-26 15:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f4a5b6c7d8e9"
down_revision: Union[str, None] = "b2d3e4f5a6c7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "scheduled_sync_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("sync_type", sa.String(length=20), nullable=False),
        sa.Column("sync_action", sa.String(length=20), nullable=False),
        sa.Column("index_symbol", sa.String(length=20), nullable=True),
        sa.Column("symbols", sa.JSON(), nullable=False),
        sa.Column("date_from", sa.Date(), nullable=True),
        sa.Column("date_to", sa.Date(), nullable=True),
        sa.Column("auto_repair", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("starts_at", sa.DateTime(), nullable=False),
        sa.Column("interval_value", sa.Integer(), nullable=False),
        sa.Column("interval_unit", sa.String(length=20), nullable=False),
        sa.Column("timezone", sa.String(length=64), nullable=False, server_default="Asia/Ho_Chi_Minh"),
        sa.Column("max_retries", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_run_at", sa.DateTime(), nullable=False),
        sa.Column("last_run_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_scheduled_sync_jobs_enabled_next_run",
        "scheduled_sync_jobs",
        ["enabled", "next_run_at"],
    )
    op.create_index(
        "ix_scheduled_sync_jobs_sync_type_action",
        "scheduled_sync_jobs",
        ["sync_type", "sync_action"],
    )

    op.create_table(
        "scheduled_sync_job_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("scheduled_for", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("error", sa.String(length=1000), nullable=True),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["scheduled_sync_jobs.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_scheduled_sync_job_runs_status_scheduled_for",
        "scheduled_sync_job_runs",
        ["status", "scheduled_for"],
    )
    op.create_index(
        "ix_scheduled_sync_job_runs_job_scheduled_for",
        "scheduled_sync_job_runs",
        ["job_id", "scheduled_for"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_scheduled_sync_job_runs_job_scheduled_for",
        table_name="scheduled_sync_job_runs",
    )
    op.drop_index(
        "ix_scheduled_sync_job_runs_status_scheduled_for",
        table_name="scheduled_sync_job_runs",
    )
    op.drop_table("scheduled_sync_job_runs")

    op.drop_index(
        "ix_scheduled_sync_jobs_sync_type_action",
        table_name="scheduled_sync_jobs",
    )
    op.drop_index(
        "ix_scheduled_sync_jobs_enabled_next_run",
        table_name="scheduled_sync_jobs",
    )
    op.drop_table("scheduled_sync_jobs")
