from __future__ import annotations

from typing import List, Dict
import asyncio
from datetime import datetime, timedelta
import pandas as pd

from sqlalchemy import select
from app.db.database import async_session
from app.db.models import StockCompany
from app.services.sync_status import sync_status
from app.core.logging_config import log_background_start

from .core import (
    background_executor,
    logger,
    bg_logger,
    api_circuit_breaker,
    CircuitOpenError,
    retry_with_backoff,
    _is_rate_limit_error,
    _record_rate_limit,
)
from .models import StockInfo


class StockMetadataService:
    """Enrich stock data with company metadata and financial ratios."""

    def __init__(self) -> None:
        self._enriching_tickers = set()

    async def apply_cache_to_stocks(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """Apply currently cached data to stocks without fetching new data."""
        if not stocks:
            return []

        tickers = [s.ticker for s in stocks]
        async with async_session() as session:
            stmt = select(StockCompany).where(StockCompany.symbol.in_(tickers))
            result = await session.execute(stmt)
            cached_data = {c.symbol: c for c in result.scalars().all()}

            for stock in stocks:
                if stock.ticker in cached_data:
                    company = cached_data[stock.ticker]
                    if not stock.company_name and company.company_name:
                        stock.company_name = company.company_name
                    # Don't overwrite exchange if we already have it from the price board
                    if not stock.exchange and company.exchange:
                        stock.exchange = company.exchange
                    if stock.charter_capital == 0 and company.charter_capital:
                        stock.charter_capital = company.charter_capital
                    if stock.pe_ratio is None and company.pe_ratio:
                        stock.pe_ratio = company.pe_ratio
        return stocks

    async def enrich_stocks_with_metadata(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """Add company names and financial metadata to stock info objects, using DB cache."""
        if not stocks:
            return []

        tickers = [s.ticker for s in stocks if s.ticker not in self._enriching_tickers]
        if not tickers:
            return stocks

        now = datetime.utcnow()
        stale_threshold = now - timedelta(days=7)
        error_stale_threshold = now - timedelta(hours=1)  # Retry missing PE after 1 hour

        async with async_session() as session:
            # Try to get from DB
            stmt = select(StockCompany).where(StockCompany.symbol.in_(tickers))
            result = await session.execute(stmt)
            cached_data = {c.symbol: c for c in result.scalars().all()}

            # Identify what's missing or stale
            tickers_needing_name = [t for t in tickers if t not in cached_data]
            tickers_needing_finance = [
                t for t in tickers
                if (t not in cached_data or
                    cached_data[t].updated_at is None or
                    (cached_data[t].pe_ratio is None and cached_data[t].updated_at < error_stale_threshold) or
                    cached_data[t].updated_at < stale_threshold)
            ]

            # Fetch missing names if needed
            if tickers_needing_name:
                loop = asyncio.get_event_loop()
                try:
                    all_symbols_df = await loop.run_in_executor(background_executor, self._fetch_all_symbols)
                except CircuitOpenError as e:
                    bg_logger.warning(f"Skipping symbol name enrichment due to rate limit: {e}")
                    all_symbols_df = None

                if all_symbols_df is not None and not all_symbols_df.empty:
                    for _, row in all_symbols_df.iterrows():
                        symbol = row['symbol']
                        name = row['organ_name']
                        if symbol in tickers_needing_name:
                            if symbol not in cached_data:
                                new_company = StockCompany(symbol=symbol, company_name=name)
                                session.add(new_company)
                                cached_data[symbol] = new_company
                            else:
                                cached_data[symbol].company_name = name

            # Fetch missing/stale financial data if needed
            if tickers_needing_finance:
                # Early bail-out if rate limited - skip API calls entirely
                if sync_status.is_rate_limited:
                    bg_logger.debug("Skipping metadata enrichment API calls due to rate limit")
                    # Still apply existing cached data below
                else:
                    # Limit batch size to avoid long hangs in one request
                    batch_limit = 50
                    tickers_to_fetch = [t for t in tickers_needing_finance if t not in self._enriching_tickers][:batch_limit]

                    if tickers_to_fetch:
                        # Mark as enriching to avoid multiple tasks for same symbols
                        self._enriching_tickers.update(tickers_to_fetch)

                        try:
                            log_background_start(
                                "Metadata Enrichment",
                                f"{len(tickers_to_fetch)}/{len(tickers_needing_finance)} stocks"
                            )
                            loop = asyncio.get_event_loop()

                            # Fetch one by one and commit incrementally
                            for symbol in tickers_to_fetch:
                                # Check rate limit on each iteration for early exit
                                if sync_status.is_rate_limited or not api_circuit_breaker.can_proceed():
                                    bg_logger.warning("Rate limit detected during enrichment, stopping batch")
                                    break
                                try:
                                    # Add a small delay between symbols
                                    await asyncio.sleep(1.0)
                                    data = await loop.run_in_executor(background_executor, self._fetch_stock_finance_sync, symbol)

                                    if data and symbol in cached_data:
                                        cached_data[symbol].pe_ratio = data.get('pe_ratio')
                                        cached_data[symbol].updated_at = now
                                        await session.commit()
                                    elif symbol in cached_data:
                                        # Still update to avoid retrying immediately, but mark as updated now
                                        cached_data[symbol].updated_at = now
                                        await session.commit()
                                except Exception as e:
                                    bg_logger.error(f"Error enriching {symbol}: {e}")
                                    if _is_rate_limit_error(e):
                                        bg_logger.warning("Rate limit hit during enrichment, stopping batch")
                                        break
                        finally:
                            # Clean up
                            for t in tickers_to_fetch:
                                self._enriching_tickers.discard(t)

            await session.commit()

            for stock in stocks:
                if stock.ticker in cached_data:
                    company = cached_data[stock.ticker]
                    # Update cache if we have better data from price board
                    if not company.company_name and stock.company_name:
                        company.company_name = stock.company_name
                    # Update exchange if it's currently empty and we have it from price board
                    if not company.exchange and stock.exchange:
                        company.exchange = stock.exchange

                    if not stock.company_name and company.company_name:
                        stock.company_name = company.company_name
                    # Don't overwrite exchange if we already have it from the price board
                    if not stock.exchange and company.exchange:
                        stock.exchange = company.exchange
                    # Use cached value if real-time value is missing
                    if stock.charter_capital == 0 and company.charter_capital:
                        stock.charter_capital = company.charter_capital
                    if stock.pe_ratio is None and company.pe_ratio:
                        stock.pe_ratio = company.pe_ratio

            await session.commit()

        return stocks

    def _fetch_stock_finance_sync(self, symbol: str) -> Dict | None:
        """
        Fetch financial metadata for a single symbol synchronously.
        Uses retry mechanism with exponential backoff for rate limits.
        """
        from vnstock import Vnstock

        def fetch_finance():
            # We only need the latest ratio
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            ratio = s.finance.ratio(period='quarter', lang='vi')
            if ratio is not None and not ratio.empty:
                # Try multiple possible column names for P/E
                pe = None
                if ('Chỉ tiêu định giá', 'P/E') in ratio.columns:
                    pe = ratio.iloc[0].get(('Chỉ tiêu định giá', 'P/E'))
                else:
                    # Look for any column containing 'P/E'
                    for col in ratio.columns:
                        col_str = str(col)
                        if 'P/E' in col_str:
                            pe = ratio.iloc[0].get(col)
                            break

                return {
                    'pe_ratio': float(pe) if pd.notna(pe) else None
                }
            return None

        try:
            return retry_with_backoff(fetch_finance, max_retries=3)
        except Exception as e:
            bg_logger.error(f"Error fetching financial metadata for {symbol}: {e}")
            raise e

    def _fetch_all_symbols(self) -> pd.DataFrame:
        """Fetch all symbols from vnstock."""
        from vnstock import Listing

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError("Circuit breaker open - cannot fetch all symbols")

        try:
            listing = Listing(source='VCI')
            result = listing.all_symbols()
            api_circuit_breaker.record_success()
            return result
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching all symbols: {e}")
            raise
