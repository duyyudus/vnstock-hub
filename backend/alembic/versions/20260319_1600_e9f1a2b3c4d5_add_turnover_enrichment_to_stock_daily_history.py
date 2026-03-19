"""add turnover enrichment to stock daily history

Revision ID: e9f1a2b3c4d5
Revises: d8e9f0a1b2c3
Create Date: 2026-03-19 16:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = "e9f1a2b3c4d5"
down_revision = "d8e9f0a1b2c3"
branch_labels = None
depends_on = None


def _column_names(inspector, table_name: str) -> set[str]:
    return {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "stock_daily_history" not in inspector.get_table_names():
        return

    existing_columns = _column_names(inspector, "stock_daily_history")
    extra_columns = [
        sa.Column("matched_volume", sa.BigInteger(), nullable=True),
        sa.Column("matched_value", sa.Float(), nullable=True),
        sa.Column("deal_volume", sa.BigInteger(), nullable=True),
        sa.Column("deal_value", sa.Float(), nullable=True),
        sa.Column("total_volume", sa.BigInteger(), nullable=True),
        sa.Column("total_value", sa.Float(), nullable=True),
    ]

    for column in extra_columns:
        if column.name not in existing_columns:
            op.add_column("stock_daily_history", column)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "stock_daily_history" not in inspector.get_table_names():
        return

    existing_columns = _column_names(inspector, "stock_daily_history")
    for column_name in [
        "total_value",
        "total_volume",
        "deal_value",
        "deal_volume",
        "matched_value",
        "matched_volume",
    ]:
        if column_name in existing_columns:
            op.drop_column("stock_daily_history", column_name)
