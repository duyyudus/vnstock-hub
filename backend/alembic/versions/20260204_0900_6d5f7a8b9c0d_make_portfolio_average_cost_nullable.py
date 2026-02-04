"""make portfolio average cost nullable

Revision ID: 6d5f7a8b9c0d
Revises: 0f1e2d3c4b5a
Create Date: 2026-02-04 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6d5f7a8b9c0d'
down_revision: Union[str, None] = '0f1e2d3c4b5a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        'portfolio_positions',
        'average_cost',
        existing_type=sa.Float(),
        nullable=True
    )


def downgrade() -> None:
    op.alter_column(
        'portfolio_positions',
        'average_cost',
        existing_type=sa.Float(),
        nullable=False
    )
