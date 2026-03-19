"""drop_legacy_price_history_tables

Revision ID: d8e9f0a1b2c3
Revises: c7d8e9f0a1b2
Create Date: 2026-03-19 10:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d8e9f0a1b2c3"
down_revision: Union[str, None] = "c7d8e9f0a1b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _table_exists(inspector, "stock_daily_prices") and _table_exists(inspector, "stock_daily_history"):
        op.drop_table("stock_daily_prices")

    inspector = sa.inspect(bind)
    if _table_exists(inspector, "stock_price_sync_state") and _table_exists(inspector, "stock_history_sync_state"):
        op.drop_table("stock_price_sync_state")


def downgrade() -> None:
    return
