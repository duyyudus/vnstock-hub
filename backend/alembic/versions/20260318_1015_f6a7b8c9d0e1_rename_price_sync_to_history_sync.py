"""rename_price_sync_to_history_sync

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-03-18 10:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f6a7b8c9d0e1'
down_revision: Union[str, None] = 'e5f6a7b8c9d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _column_names(inspector: sa.Inspector, table_name: str) -> set[str]:
    return {column["name"] for column in inspector.get_columns(table_name)}


def _index_names(inspector: sa.Inspector, table_name: str) -> set[str]:
    return {index["name"] for index in inspector.get_indexes(table_name)}


def _unique_constraint_names(inspector: sa.Inspector, table_name: str) -> set[str]:
    return {constraint["name"] for constraint in inspector.get_unique_constraints(table_name)}


def _rename_index(bind, table_name: str, old_name: str, new_name: str, columns: list[str], unique: bool = False) -> None:
    inspector = sa.inspect(bind)
    existing_indexes = _index_names(inspector, table_name)
    existing_columns = _column_names(inspector, table_name)

    if old_name in existing_indexes and new_name not in existing_indexes:
        if bind.dialect.name == "postgresql":
            op.execute(sa.text(f'ALTER INDEX "{old_name}" RENAME TO "{new_name}"'))
        else:
            op.drop_index(old_name, table_name=table_name)
            op.create_index(new_name, table_name, columns, unique=unique)
        return

    if new_name not in existing_indexes and all(column in existing_columns for column in columns):
        op.create_index(new_name, table_name, columns, unique=unique)


def _rename_unique_constraint(
    bind,
    table_name: str,
    old_name: str,
    new_name: str,
    columns: list[str],
) -> None:
    inspector = sa.inspect(bind)
    existing_constraints = _unique_constraint_names(inspector, table_name)
    existing_columns = _column_names(inspector, table_name)

    if old_name in existing_constraints and new_name not in existing_constraints:
        if bind.dialect.name == "postgresql":
            op.execute(sa.text(
                f'ALTER TABLE "{table_name}" RENAME CONSTRAINT "{old_name}" TO "{new_name}"'
            ))
        else:
            op.drop_constraint(old_name, table_name, type_="unique")
            op.create_unique_constraint(new_name, table_name, columns)
        return

    if new_name not in existing_constraints and all(column in existing_columns for column in columns):
        op.create_unique_constraint(new_name, table_name, columns)


def _rename_table_if_needed(old_name: str, new_name: str) -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if _table_exists(inspector, old_name) and not _table_exists(inspector, new_name):
        op.rename_table(old_name, new_name)


def upgrade() -> None:
    bind = op.get_bind()

    _rename_table_if_needed("stock_daily_prices", "stock_daily_history")
    _rename_table_if_needed("stock_price_sync_state", "stock_history_sync_state")

    inspector = sa.inspect(bind)

    if _table_exists(inspector, "stock_daily_history"):
        table_name = "stock_daily_history"
        existing_columns = _column_names(inspector, table_name)

        extra_columns = [
            sa.Column("foreign_buy_volume", sa.BigInteger(), nullable=True),
            sa.Column("foreign_buy_value", sa.Float(), nullable=True),
            sa.Column("foreign_sell_volume", sa.BigInteger(), nullable=True),
            sa.Column("foreign_sell_value", sa.Float(), nullable=True),
            sa.Column("foreign_net_volume", sa.BigInteger(), nullable=True),
            sa.Column("foreign_net_value", sa.Float(), nullable=True),
            sa.Column("prop_buy_volume", sa.BigInteger(), nullable=True),
            sa.Column("prop_buy_value", sa.Float(), nullable=True),
            sa.Column("prop_sell_volume", sa.BigInteger(), nullable=True),
            sa.Column("prop_sell_value", sa.Float(), nullable=True),
        ]
        for column in extra_columns:
            if column.name not in existing_columns:
                op.add_column(table_name, column)

        _rename_unique_constraint(
            bind=bind,
            table_name=table_name,
            old_name="uq_symbol_date",
            new_name="uq_stock_daily_history_symbol_date",
            columns=["symbol", "date"],
        )
        _rename_index(
            bind=bind,
            table_name=table_name,
            old_name="ix_stock_daily_prices_symbol_date",
            new_name="ix_stock_daily_history_symbol_date",
            columns=["symbol", "date"],
        )

    inspector = sa.inspect(bind)
    if _table_exists(inspector, "stock_history_sync_state"):
        table_name = "stock_history_sync_state"

        _rename_unique_constraint(
            bind=bind,
            table_name=table_name,
            old_name="uq_stock_price_sync_state_symbol",
            new_name="uq_stock_history_sync_state_symbol",
            columns=["symbol"],
        )
        _rename_index(
            bind=bind,
            table_name=table_name,
            old_name="ix_stock_price_sync_state_symbol",
            new_name="ix_stock_history_sync_state_symbol",
            columns=["symbol"],
            unique=True,
        )
        _rename_index(
            bind=bind,
            table_name=table_name,
            old_name="ix_stock_price_sync_state_sync_status",
            new_name="ix_stock_history_sync_state_sync_status",
            columns=["sync_status"],
        )
        _rename_index(
            bind=bind,
            table_name=table_name,
            old_name="ix_stock_price_sync_state_latest_synced_date",
            new_name="ix_stock_history_sync_state_latest_synced_date",
            columns=["latest_synced_date"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _table_exists(inspector, "stock_history_sync_state"):
        table_name = "stock_history_sync_state"
        _rename_unique_constraint(
            bind=bind,
            table_name=table_name,
            old_name="uq_stock_history_sync_state_symbol",
            new_name="uq_stock_price_sync_state_symbol",
            columns=["symbol"],
        )
        _rename_index(
            bind=bind,
            table_name=table_name,
            old_name="ix_stock_history_sync_state_symbol",
            new_name="ix_stock_price_sync_state_symbol",
            columns=["symbol"],
            unique=True,
        )
        _rename_index(
            bind=bind,
            table_name=table_name,
            old_name="ix_stock_history_sync_state_sync_status",
            new_name="ix_stock_price_sync_state_sync_status",
            columns=["sync_status"],
        )
        _rename_index(
            bind=bind,
            table_name=table_name,
            old_name="ix_stock_history_sync_state_latest_synced_date",
            new_name="ix_stock_price_sync_state_latest_synced_date",
            columns=["latest_synced_date"],
        )

    inspector = sa.inspect(bind)
    if _table_exists(inspector, "stock_daily_history"):
        table_name = "stock_daily_history"
        existing_columns = _column_names(inspector, table_name)
        for column_name in [
            "prop_sell_value",
            "prop_sell_volume",
            "prop_buy_value",
            "prop_buy_volume",
            "foreign_net_value",
            "foreign_net_volume",
            "foreign_sell_value",
            "foreign_sell_volume",
            "foreign_buy_value",
            "foreign_buy_volume",
        ]:
            if column_name in existing_columns:
                op.drop_column(table_name, column_name)

        _rename_unique_constraint(
            bind=bind,
            table_name=table_name,
            old_name="uq_stock_daily_history_symbol_date",
            new_name="uq_symbol_date",
            columns=["symbol", "date"],
        )
        _rename_index(
            bind=bind,
            table_name=table_name,
            old_name="ix_stock_daily_history_symbol_date",
            new_name="ix_stock_daily_prices_symbol_date",
            columns=["symbol", "date"],
        )

    _rename_table_if_needed("stock_history_sync_state", "stock_price_sync_state")
    _rename_table_if_needed("stock_daily_history", "stock_daily_prices")
