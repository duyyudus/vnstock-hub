"""add_weekly_sync_last_attempt_to_price_sync_state

Revision ID: f0a1b2c3d4e5
Revises: 91b2c3d4e5f6
Create Date: 2026-02-09 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f0a1b2c3d4e5'
down_revision: Union[str, None] = '91b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_name = 'stock_price_sync_state'

    if table_name not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
    if 'weekly_sync_last_attempt_at' not in existing_columns:
        op.add_column(
            table_name,
            sa.Column('weekly_sync_last_attempt_at', sa.DateTime(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_name = 'stock_price_sync_state'

    if table_name not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
    if 'weekly_sync_last_attempt_at' in existing_columns:
        op.drop_column(table_name, 'weekly_sync_last_attempt_at')
