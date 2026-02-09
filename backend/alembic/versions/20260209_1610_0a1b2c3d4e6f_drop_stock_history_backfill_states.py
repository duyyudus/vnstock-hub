"""drop_stock_history_backfill_states

Revision ID: 0a1b2c3d4e6f
Revises: f0a1b2c3d4e5
Create Date: 2026-02-09 16:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0a1b2c3d4e6f'
down_revision: Union[str, None] = 'f0a1b2c3d4e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_name = 'stock_history_backfill_states'

    if table_name in inspector.get_table_names():
        op.drop_table(table_name)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_name = 'stock_history_backfill_states'

    if table_name in inspector.get_table_names():
        return

    op.create_table(
        table_name,
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('target_start_date', sa.Date(), nullable=True),
        sa.Column('oldest_date', sa.Date(), nullable=True),
        sa.Column('latest_date', sa.Date(), nullable=True),
        sa.Column('last_seen_at', sa.DateTime(), nullable=True),
        sa.Column('last_attempt_at', sa.DateTime(), nullable=True),
        sa.Column('next_attempt_at', sa.DateTime(), nullable=True),
        sa.Column('weekly_sync_last_attempt_at', sa.DateTime(), nullable=True),
        sa.Column('weekly_sync_last_attempt_start_date', sa.Date(), nullable=True),
        sa.Column('no_progress_attempts', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('is_exhausted', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('exhausted_until', sa.DateTime(), nullable=True),
        sa.Column('last_error', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.UniqueConstraint('symbol', name='uq_stock_history_backfill_states_symbol'),
    )

    op.create_index('ix_stock_history_backfill_states_symbol', table_name, ['symbol'], unique=True)
    op.create_index('ix_stock_history_backfill_next_attempt', table_name, ['next_attempt_at'])
    op.create_index('ix_stock_history_backfill_last_seen', table_name, ['last_seen_at'])
    op.create_index(
        'ix_stock_history_backfill_exhausted',
        table_name,
        ['is_exhausted', 'exhausted_until']
    )
