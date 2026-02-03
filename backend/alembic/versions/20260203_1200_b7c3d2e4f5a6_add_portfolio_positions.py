"""add portfolio positions

Revision ID: b7c3d2e4f5a6
Revises: 8c7a1e2b3c4d
Create Date: 2026-02-03 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7c3d2e4f5a6'
down_revision: Union[str, None] = '8c7a1e2b3c4d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'portfolio_positions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('ticker', sa.String(length=10), nullable=False),
        sa.Column('quantity', sa.Float(), nullable=False),
        sa.Column('average_cost', sa.Float(), nullable=False),
        sa.Column('purchase_date', sa.Date(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'ticker', name='uq_portfolio_position_user_ticker')
    )
    op.create_index(op.f('ix_portfolio_positions_id'), 'portfolio_positions', ['id'], unique=False)
    op.create_index(op.f('ix_portfolio_positions_user_id'), 'portfolio_positions', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_portfolio_positions_user_id'), table_name='portfolio_positions')
    op.drop_index(op.f('ix_portfolio_positions_id'), table_name='portfolio_positions')
    op.drop_table('portfolio_positions')
