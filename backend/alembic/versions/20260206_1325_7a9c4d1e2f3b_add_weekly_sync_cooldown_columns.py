"""add_weekly_sync_cooldown_columns

Revision ID: 7a9c4d1e2f3b
Revises: 6d5f7a8b9c0d
Create Date: 2026-02-06 13:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7a9c4d1e2f3b'
down_revision: Union[str, None] = '6d5f7a8b9c0d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_name = 'stock_history_backfill_states'

    if table_name not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns(table_name)}

    if 'weekly_sync_last_attempt_at' not in existing_columns:
        op.add_column(
            table_name,
            sa.Column('weekly_sync_last_attempt_at', sa.DateTime(), nullable=True),
        )

    if 'weekly_sync_last_attempt_start_date' not in existing_columns:
        op.add_column(
            table_name,
            sa.Column('weekly_sync_last_attempt_start_date', sa.Date(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_name = 'stock_history_backfill_states'

    if table_name not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns(table_name)}

    if 'weekly_sync_last_attempt_start_date' in existing_columns:
        op.drop_column(table_name, 'weekly_sync_last_attempt_start_date')
    if 'weekly_sync_last_attempt_at' in existing_columns:
        op.drop_column(table_name, 'weekly_sync_last_attempt_at')
