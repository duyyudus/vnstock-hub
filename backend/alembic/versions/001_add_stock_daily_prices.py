"""Add stock_daily_prices table

Revision ID: 001_add_stock_daily_prices
Revises: 
Create Date: 2026-01-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_add_stock_daily_prices'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create stock_daily_prices table
    op.create_table(
        'stock_daily_prices',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('open', sa.Float(), nullable=True),
        sa.Column('high', sa.Float(), nullable=True),
        sa.Column('low', sa.Float(), nullable=True),
        sa.Column('close', sa.Float(), nullable=False),
        sa.Column('volume', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint('symbol', 'date', name='uq_symbol_date'),
    )
    
    # Create index for efficient lookups
    op.create_index(
        'ix_stock_daily_prices_symbol_date',
        'stock_daily_prices',
        ['symbol', 'date']
    )


def downgrade() -> None:
    op.drop_index('ix_stock_daily_prices_symbol_date', table_name='stock_daily_prices')
    op.drop_table('stock_daily_prices')
