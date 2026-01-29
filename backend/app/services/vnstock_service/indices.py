from __future__ import annotations

from typing import List
import asyncio
from datetime import datetime, timedelta
import pandas as pd

from sqlalchemy import select
from app.db.database import async_session
from app.db.models import StockIndex

from .core import (
    frontend_executor,
    logger,
    api_circuit_breaker,
    CircuitOpenError,
    _is_rate_limit_error,
    _record_rate_limit,
)
from .models import IndexValue
from .symbols import VALID_GROUPS, get_group_code_for_index

# Index name mapping for display
INDEX_NAMES = {
    'VNINDEX': 'VN-Index',
    'HNXINDEX': 'HNX-Index',
    'UPCOMINDEX': 'UPCOM-Index',
    'VN30': 'VN30',
    'HNX30': 'HNX30',
}


class IndicesService:
    """Index-related operations."""

    async def sync_indices(self) -> None:
        """
        Fetch all indices from vnstock and update the database.
        """
        loop = asyncio.get_event_loop()
        indices_df = await loop.run_in_executor(frontend_executor, self._fetch_all_indices_from_lib)

        # Define manual indices (Exchanges/Groups that are not in all_indices but are supported)
        manual_indices = [
            {'symbol': 'HOSE', 'name': 'HOSE Exchange', 'group': 'Exchange'},
            {'symbol': 'HNX', 'name': 'HNX Exchange', 'group': 'Exchange'},
            {'symbol': 'UPCOM', 'name': 'UPCOM Exchange', 'group': 'Exchange'},
        ]

        async with async_session() as session:
            count = 0

            # Process manual indices
            for item in manual_indices:
                try:
                    symbol = item['symbol']
                    # Ensure supported (it should be, since we manually added it)
                    if symbol not in VALID_GROUPS:
                        continue

                    # Upsert
                    stmt = select(StockIndex).where(StockIndex.symbol == symbol)
                    result = await session.execute(stmt)
                    existing = result.scalar_one_or_none()

                    if existing:
                        existing.name = item['name']
                        existing.group = item['group']
                        existing.updated_at = datetime.utcnow()
                    else:
                        new_index = StockIndex(
                            symbol=symbol,
                            name=item['name'],
                            group=item['group']
                        )
                        session.add(new_index)
                    count += 1
                except Exception as e:
                    logger.warning(f"Error processing manual index {item['symbol']}: {e}")

            # Process fetched indices
            if indices_df is not None and not indices_df.empty:
                for _, row in indices_df.iterrows():
                    try:
                        symbol = row.get('symbol')
                        if not symbol:
                            continue

                        # Check if supported
                        group_code = get_group_code_for_index(symbol)
                        if group_code not in VALID_GROUPS:
                            continue

                        # Prepare data
                        name = row.get('name', '')
                        if 'full_name' in row:
                            name = row['full_name']
                        elif not name and 'index_name' in row:
                            name = row['index_name']

                        if not name:
                            name = symbol

                        group = row.get('group', None)
                        description = row.get('description', None)

                        # Upsert logic
                        stmt = select(StockIndex).where(StockIndex.symbol == symbol)
                        result = await session.execute(stmt)
                        existing = result.scalar_one_or_none()

                        if existing:
                            existing.name = name
                            existing.group = group
                            existing.description = description
                            existing.updated_at = datetime.utcnow()
                        else:
                            new_index = StockIndex(
                                symbol=symbol,
                                name=name,
                                group=group,
                                description=description
                            )
                            session.add(new_index)
                        count += 1
                    except Exception as e:
                        logger.warning(f"Error processing index {row.get('symbol')}: {e}")

            await session.commit()
            logger.info(f"Synced {count} supported indices to database")

    def _fetch_all_indices_from_lib(self) -> pd.DataFrame:
        """Fetch all indices from vnstock library synchronously."""
        from vnstock import Listing

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError("Circuit breaker open - cannot fetch indices list")

        try:
            result = Listing(source='VCI').all_indices()
            api_circuit_breaker.record_success()
            return result
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching indices: {e}")
            raise

    async def get_indices(self) -> List[StockIndex]:
        """
        Get all available indices from the database.
        """
        async with async_session() as session:
            stmt = select(StockIndex).order_by(StockIndex.symbol)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_index_values(self, symbols: List[str] | None = None) -> List[IndexValue]:
        """
        Fetch latest values for major market indices.
        """
        if symbols is None:
            symbols = ['VNINDEX', 'HNXINDEX', 'UPCOMINDEX', 'VN30', 'HNX30']

        loop = asyncio.get_event_loop()
        results = []

        for symbol in symbols:
            try:
                index_data = await loop.run_in_executor(frontend_executor, self._fetch_index_value_sync, symbol)
                if index_data:
                    results.append(index_data)
            except Exception as e:
                logger.warning(f"Error fetching index value for {symbol}: {e}")

        return results

    def _fetch_index_value_sync(self, symbol: str) -> IndexValue | None:
        """
        Fetch latest value for a single index synchronously.
        Checks circuit breaker before API call to fail fast when rate limited.
        """
        from vnstock import Vnstock

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch index value for {symbol}")

        try:
            today = datetime.now().strftime('%Y-%m-%d')
            week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

            vs = Vnstock(symbol=symbol, source='VCI')
            stock = vs.stock()
            df = stock.quote.history(start=week_ago, end=today)

            if df is not None and not df.empty:
                last_row = df.iloc[-1]
                open_price = float(last_row['open'])
                close_price = float(last_row['close'])
                change_value = close_price - open_price
                change_pct = (change_value / open_price) * 100 if open_price > 0 else 0

                api_circuit_breaker.record_success()
                return IndexValue(
                    symbol=symbol,
                    name=INDEX_NAMES.get(symbol, symbol),
                    value=round(close_price, 2),
                    change=round(change_pct, 2),
                    change_value=round(change_value, 2)
                )

            api_circuit_breaker.record_success()
            return None

        except (SystemExit, Exception) as e:
            # Check if this is a rate limit error
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching index value for {symbol}: {e}")
            logger.warning(f"Error fetching index value for {symbol}: {e}")
            return None
