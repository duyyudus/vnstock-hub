"""add fund_navs table

Revision ID: 9c7a3e4f5b2d
Revises: 20260120_1713_1a5fe5192058
Create Date: 2026-01-24 14:27:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9c7a3e4f5b2d'
down_revision: Union[str, None] = '1a5fe5192058'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'fund_navs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(length=30), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('nav', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'date', name='uq_fund_symbol_date')
    )
    op.create_index('ix_fund_navs_symbol_date', 'fund_navs', ['symbol', 'date'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_fund_navs_symbol_date', table_name='fund_navs')
    op.drop_table('fund_navs')
