"""backfill_history_tables_after_rename_race

Revision ID: c7d8e9f0a1b2
Revises: f6a7b8c9d0e1
Create Date: 2026-03-19 09:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c7d8e9f0a1b2"
down_revision: Union[str, None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _column_names(inspector: sa.Inspector, table_name: str) -> set[str]:
    return {column["name"] for column in inspector.get_columns(table_name)}


def _copy_missing_rows(
    source_table: str,
    target_table: str,
    columns: list[str],
    conflict_columns: list[str],
) -> None:
    if not columns:
        return

    source_alias = "src"
    target_alias = "dst"
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    select_columns = ", ".join(f'{source_alias}."{column}"' for column in columns)
    conflict_clause = " AND ".join(
        f'{target_alias}."{column}" = {source_alias}."{column}"'
        for column in conflict_columns
    )

    op.execute(
        sa.text(
            f"""
            INSERT INTO "{target_table}" ({quoted_columns})
            SELECT {select_columns}
            FROM "{source_table}" AS {source_alias}
            WHERE NOT EXISTS (
                SELECT 1
                FROM "{target_table}" AS {target_alias}
                WHERE {conflict_clause}
            )
            """
        )
    )


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _table_exists(inspector, "stock_daily_prices") and _table_exists(inspector, "stock_daily_history"):
        source_columns = _column_names(inspector, "stock_daily_prices")
        target_columns = _column_names(inspector, "stock_daily_history")
        daily_columns = [
            column
            for column in ["symbol", "date", "open", "high", "low", "close", "volume", "created_at"]
            if column in source_columns and column in target_columns
        ]
        _copy_missing_rows(
            source_table="stock_daily_prices",
            target_table="stock_daily_history",
            columns=daily_columns,
            conflict_columns=["symbol", "date"],
        )

    inspector = sa.inspect(bind)
    if _table_exists(inspector, "stock_price_sync_state") and _table_exists(inspector, "stock_history_sync_state"):
        source_columns = _column_names(inspector, "stock_price_sync_state")
        target_columns = _column_names(inspector, "stock_history_sync_state")
        sync_state_columns = [
            column
            for column in [
                "symbol",
                "listing_date",
                "sync_status",
                "sync_started_at",
                "sync_completed_at",
                "earliest_synced_date",
                "latest_synced_date",
                "last_incremental_sync_at",
                "weekly_sync_last_attempt_at",
                "last_error",
                "retry_count",
                "updated_at",
            ]
            if column in source_columns and column in target_columns
        ]
        _copy_missing_rows(
            source_table="stock_price_sync_state",
            target_table="stock_history_sync_state",
            columns=sync_state_columns,
            conflict_columns=["symbol"],
        )


def downgrade() -> None:
    # Data backfill is intentionally irreversible.
    return
