"""add_stock_price_sync_state

Revision ID: 91b2c3d4e5f6
Revises: 7a9c4d1e2f3b
Create Date: 2026-02-09 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '91b2c3d4e5f6'
down_revision: Union[str, None] = '7a9c4d1e2f3b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    table_name = 'stock_price_sync_state'
    if table_name in inspector.get_table_names():
        return

    op.create_table(
        table_name,
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('listing_date', sa.Date(), nullable=True),
        sa.Column('bootstrap_status', sa.String(length=20), nullable=False, server_default='idle'),
        sa.Column('bootstrap_started_at', sa.DateTime(), nullable=True),
        sa.Column('bootstrap_completed_at', sa.DateTime(), nullable=True),
        sa.Column('earliest_synced_date', sa.Date(), nullable=True),
        sa.Column('latest_synced_date', sa.Date(), nullable=True),
        sa.Column('last_incremental_sync_at', sa.DateTime(), nullable=True),
        sa.Column('last_error', sa.String(length=500), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.UniqueConstraint('symbol', name='uq_stock_price_sync_state_symbol'),
    )

    op.create_index('ix_stock_price_sync_state_symbol', table_name, ['symbol'], unique=True)
    op.create_index('ix_stock_price_sync_state_bootstrap_status', table_name, ['bootstrap_status'])
    op.create_index('ix_stock_price_sync_state_latest_synced_date', table_name, ['latest_synced_date'])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    table_name = 'stock_price_sync_state'
    if table_name not in inspector.get_table_names():
        return

    existing_indexes = {idx['name'] for idx in inspector.get_indexes(table_name)}
    if 'ix_stock_price_sync_state_latest_synced_date' in existing_indexes:
        op.drop_index('ix_stock_price_sync_state_latest_synced_date', table_name=table_name)
    if 'ix_stock_price_sync_state_bootstrap_status' in existing_indexes:
        op.drop_index('ix_stock_price_sync_state_bootstrap_status', table_name=table_name)
    if 'ix_stock_price_sync_state_symbol' in existing_indexes:
        op.drop_index('ix_stock_price_sync_state_symbol', table_name=table_name)

    op.drop_table(table_name)
