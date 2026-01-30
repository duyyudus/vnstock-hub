"""add bookmark groups

Revision ID: 8c7a1e2b3c4d
Revises: 5f8c2c1a9d7b
Create Date: 2026-01-30 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8c7a1e2b3c4d'
down_revision: Union[str, None] = '5f8c2c1a9d7b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'bookmark_groups',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=120), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'name', name='uq_bookmark_group_user_name')
    )
    op.create_index(op.f('ix_bookmark_groups_id'), 'bookmark_groups', ['id'], unique=False)
    op.create_index(op.f('ix_bookmark_groups_user_id'), 'bookmark_groups', ['user_id'], unique=False)

    op.create_table(
        'bookmark_stocks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('group_id', sa.Integer(), nullable=False),
        sa.Column('ticker', sa.String(length=10), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['group_id'], ['bookmark_groups.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('group_id', 'ticker', name='uq_bookmark_group_ticker')
    )
    op.create_index(op.f('ix_bookmark_stocks_group_id'), 'bookmark_stocks', ['group_id'], unique=False)
    op.create_index(op.f('ix_bookmark_stocks_id'), 'bookmark_stocks', ['id'], unique=False)
    op.create_index('ix_bookmark_group_ticker', 'bookmark_stocks', ['group_id', 'ticker'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_bookmark_group_ticker', table_name='bookmark_stocks')
    op.drop_index(op.f('ix_bookmark_stocks_id'), table_name='bookmark_stocks')
    op.drop_index(op.f('ix_bookmark_stocks_group_id'), table_name='bookmark_stocks')
    op.drop_table('bookmark_stocks')
    op.drop_index(op.f('ix_bookmark_groups_user_id'), table_name='bookmark_groups')
    op.drop_index(op.f('ix_bookmark_groups_id'), table_name='bookmark_groups')
    op.drop_table('bookmark_groups')
