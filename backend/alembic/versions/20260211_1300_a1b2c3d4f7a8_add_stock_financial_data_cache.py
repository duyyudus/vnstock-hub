"""add_stock_financial_data_cache

Revision ID: a1b2c3d4f7a8
Revises: 0a1b2c3d4e6f
Create Date: 2026-02-11 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4f7a8'
down_revision: Union[str, None] = '0a1b2c3d4e6f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    table_name = 'stock_financial_data_cache'
    if table_name in inspector.get_table_names():
        return

    op.create_table(
        table_name,
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('data_type', sa.String(length=30), nullable=False),
        sa.Column('period', sa.String(length=10), nullable=False),
        sa.Column('lang', sa.String(length=10), nullable=False, server_default='en'),
        sa.Column('data', sa.JSON(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.UniqueConstraint(
            'symbol',
            'data_type',
            'period',
            'lang',
            name='uq_stock_financial_data_cache_key',
        ),
    )

    op.create_index(
        'ix_stock_financial_data_cache_lookup',
        table_name,
        ['symbol', 'data_type', 'period', 'lang'],
        unique=False,
    )
    op.create_index(
        'ix_stock_financial_data_cache_symbol_updated_at',
        table_name,
        ['symbol', 'updated_at'],
        unique=False,
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    table_name = 'stock_financial_data_cache'
    if table_name not in inspector.get_table_names():
        return

    existing_indexes = {idx['name'] for idx in inspector.get_indexes(table_name)}
    if 'ix_stock_financial_data_cache_symbol_updated_at' in existing_indexes:
        op.drop_index('ix_stock_financial_data_cache_symbol_updated_at', table_name=table_name)
    if 'ix_stock_financial_data_cache_lookup' in existing_indexes:
        op.drop_index('ix_stock_financial_data_cache_lookup', table_name=table_name)

    op.drop_table(table_name)
