"""rename_price_sync_bootstrap_columns

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4f7a8
Create Date: 2026-02-12 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = 'a1b2c3d4f7a8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _rename_index(
    bind,
    table_name: str,
    old_index_name: str,
    new_index_name: str,
    target_column: str,
) -> None:
    inspector = sa.inspect(bind)
    existing_indexes = {idx['name'] for idx in inspector.get_indexes(table_name)}
    existing_columns = {column["name"] for column in inspector.get_columns(table_name)}

    if old_index_name in existing_indexes and new_index_name not in existing_indexes:
        if bind.dialect.name == "postgresql":
            op.execute(sa.text(f"ALTER INDEX {old_index_name} RENAME TO {new_index_name}"))
        else:
            op.drop_index(old_index_name, table_name=table_name)
            op.create_index(new_index_name, table_name, [target_column])
        return

    if old_index_name in existing_indexes and new_index_name in existing_indexes:
        op.drop_index(old_index_name, table_name=table_name)
        return

    if (
        old_index_name not in existing_indexes
        and new_index_name not in existing_indexes
        and target_column in existing_columns
    ):
        op.create_index(new_index_name, table_name, [target_column])


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_name = "stock_price_sync_state"

    if table_name not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns(table_name)}

    if "bootstrap_status" in existing_columns and "sync_status" not in existing_columns:
        op.alter_column(table_name, "bootstrap_status", new_column_name="sync_status")
    if "bootstrap_started_at" in existing_columns and "sync_started_at" not in existing_columns:
        op.alter_column(table_name, "bootstrap_started_at", new_column_name="sync_started_at")
    if "bootstrap_completed_at" in existing_columns and "sync_completed_at" not in existing_columns:
        op.alter_column(table_name, "bootstrap_completed_at", new_column_name="sync_completed_at")

    _rename_index(
        bind=bind,
        table_name=table_name,
        old_index_name="ix_stock_price_sync_state_bootstrap_status",
        new_index_name="ix_stock_price_sync_state_sync_status",
        target_column="sync_status",
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_name = "stock_price_sync_state"

    if table_name not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns(table_name)}

    if "sync_status" in existing_columns and "bootstrap_status" not in existing_columns:
        op.alter_column(table_name, "sync_status", new_column_name="bootstrap_status")
    if "sync_started_at" in existing_columns and "bootstrap_started_at" not in existing_columns:
        op.alter_column(table_name, "sync_started_at", new_column_name="bootstrap_started_at")
    if "sync_completed_at" in existing_columns and "bootstrap_completed_at" not in existing_columns:
        op.alter_column(table_name, "sync_completed_at", new_column_name="bootstrap_completed_at")

    _rename_index(
        bind=bind,
        table_name=table_name,
        old_index_name="ix_stock_price_sync_state_sync_status",
        new_index_name="ix_stock_price_sync_state_bootstrap_status",
        target_column="bootstrap_status",
    )
