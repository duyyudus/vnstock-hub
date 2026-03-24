"""add trading positions

Revision ID: a1c2e3f4b5d6
Revises: e9f1a2b3c4d5
Create Date: 2026-03-24 11:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1c2e3f4b5d6"
down_revision: Union[str, None] = "e9f1a2b3c4d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "trading_positions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("account_label", sa.String(length=120), nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("average_entry_cost", sa.Float(), nullable=False),
        sa.Column("opened_date", sa.Date(), nullable=False),
        sa.Column("target_price", sa.Float(), nullable=True),
        sa.Column("stop_loss", sa.Float(), nullable=True),
        sa.Column("notes", sa.String(length=1000), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "account_label",
            "ticker",
            name="uq_trading_position_user_account_ticker",
        ),
    )
    op.create_index(op.f("ix_trading_positions_id"), "trading_positions", ["id"], unique=False)
    op.create_index(op.f("ix_trading_positions_user_id"), "trading_positions", ["user_id"], unique=False)
    op.create_index(
        "ix_trading_position_user_account",
        "trading_positions",
        ["user_id", "account_label"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_trading_position_user_account", table_name="trading_positions")
    op.drop_index(op.f("ix_trading_positions_user_id"), table_name="trading_positions")
    op.drop_index(op.f("ix_trading_positions_id"), table_name="trading_positions")
    op.drop_table("trading_positions")
