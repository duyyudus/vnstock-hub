"""add user export categories

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-02-13 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column['name'] for column in inspector.get_columns('users')}

    if 'company_export_category' not in columns:
        op.add_column('users', sa.Column('company_export_category', sa.String(length=120), nullable=True))

    if 'finance_export_category' not in columns:
        op.add_column('users', sa.Column('finance_export_category', sa.String(length=120), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column['name'] for column in inspector.get_columns('users')}

    if 'finance_export_category' in columns:
        op.drop_column('users', 'finance_export_category')

    if 'company_export_category' in columns:
        op.drop_column('users', 'company_export_category')
