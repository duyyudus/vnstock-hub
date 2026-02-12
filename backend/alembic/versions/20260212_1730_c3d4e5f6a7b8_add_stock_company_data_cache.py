"""add_stock_company_data_cache

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-02-12 17:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    table_name = 'stock_company_data_cache'
    if table_name in inspector.get_table_names():
        return

    op.create_table(
        table_name,
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('data_type', sa.String(length=30), nullable=False),
        sa.Column('data', sa.JSON(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.UniqueConstraint(
            'symbol',
            'data_type',
            name='uq_stock_company_data_cache_key',
        ),
    )

    op.create_index(
        'ix_stock_company_data_cache_lookup',
        table_name,
        ['symbol', 'data_type'],
        unique=False,
    )
    op.create_index(
        'ix_stock_company_data_cache_symbol_updated_at',
        table_name,
        ['symbol', 'updated_at'],
        unique=False,
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    table_name = 'stock_company_data_cache'
    if table_name not in inspector.get_table_names():
        return

    existing_indexes = {idx['name'] for idx in inspector.get_indexes(table_name)}
    if 'ix_stock_company_data_cache_symbol_updated_at' in existing_indexes:
        op.drop_index('ix_stock_company_data_cache_symbol_updated_at', table_name=table_name)
    if 'ix_stock_company_data_cache_lookup' in existing_indexes:
        op.drop_index('ix_stock_company_data_cache_lookup', table_name=table_name)

    op.drop_table(table_name)
