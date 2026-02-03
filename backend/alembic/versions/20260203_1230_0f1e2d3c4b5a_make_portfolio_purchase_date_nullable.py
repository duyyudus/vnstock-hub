"""make portfolio purchase date nullable

Revision ID: 0f1e2d3c4b5a
Revises: b7c3d2e4f5a6
Create Date: 2026-02-03 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0f1e2d3c4b5a'
down_revision: Union[str, None] = 'b7c3d2e4f5a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        'portfolio_positions',
        'purchase_date',
        existing_type=sa.Date(),
        nullable=True
    )


def downgrade() -> None:
    op.alter_column(
        'portfolio_positions',
        'purchase_date',
        existing_type=sa.Date(),
        nullable=False
    )
