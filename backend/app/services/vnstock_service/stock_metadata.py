from __future__ import annotations

from typing import List, Dict
import asyncio
from datetime import datetime, timedelta
import pandas as pd

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from app.db.database import async_session
from app.db.models import StockCompany, StockFinancialDataCache
from app.services.sync_status import sync_status
from app.core.logging_config import (
    log_background_start,
    log_background_complete,
    log_background_error,
)

from .core import (
    background_executor,
    logger,
    bg_logger,
    api_circuit_breaker,
    CircuitOpenError,
    _is_rate_limit_error,
    _record_rate_limit,
)
from .models import StockInfo


class StockMetadataService:
    """Enrich stock data with company metadata and financial ratios."""

    def __init__(self, finance_service=None) -> None:
        self._enriching_tickers = set()
        self._finance_service = finance_service

    def _extract_pe_ratio_from_records(self, ratio_records: List[Dict] | None) -> float | None:
        """Read P/E from normalized cached ratio rows."""
        if not ratio_records:
            return None

        if self._finance_service is not None:
            try:
                return self._finance_service.extract_latest_pe_ratio(ratio_records)
            except Exception:
                pass

        first_row = ratio_records[0]
        if not isinstance(first_row, dict):
            return None

        for key, value in first_row.items():
            key_upper = str(key).upper().replace(" ", "")
            if "P/E" not in key_upper and key_upper not in {"PE", "P_E"} and not key_upper.endswith("_PE"):
                continue
            try:
                if value is None:
                    continue
                parsed = float(value)
                if pd.isna(parsed):
                    continue
                return parsed
            except (TypeError, ValueError):
                continue

        return None

    @staticmethod
    def _extract_market_cap_vnd_from_records(ratio_records: List[Dict] | None) -> float | None:
        """Read snapshot market cap in VND from normalized cached ratio rows."""
        if not ratio_records:
            return None

        first_row = ratio_records[0]
        if not isinstance(first_row, dict):
            return None

        for key, value in first_row.items():
            key_upper = str(key).upper().replace(" ", "")
            if "MARKETCAP" not in key_upper:
                continue
            try:
                if value is None:
                    continue
                parsed = float(value)
                if pd.isna(parsed) or parsed <= 0:
                    continue
                return parsed
            except (TypeError, ValueError):
                continue

        return None

    def _compute_live_pe_ratio(self, stock: StockInfo, ratio_records: List[Dict] | None) -> float | None:
        """
        Compute live P/E using current market cap (or price) against the latest
        trailing earnings base implied by the freshest ratio snapshot.
        """
        snapshot_pe = self._extract_pe_ratio_from_records(ratio_records)
        snapshot_market_cap_vnd = self._extract_market_cap_vnd_from_records(ratio_records)
        if snapshot_pe is None or snapshot_pe <= 0 or snapshot_market_cap_vnd is None or snapshot_market_cap_vnd <= 0:
            return None

        trailing_earnings_vnd = snapshot_market_cap_vnd / snapshot_pe
        if not pd.notna(trailing_earnings_vnd) or trailing_earnings_vnd <= 0:
            return None

        live_market_cap_vnd = stock.market_cap * 1_000_000_000 if stock.market_cap and stock.market_cap > 0 else None
        if live_market_cap_vnd is not None:
            live_pe = live_market_cap_vnd / trailing_earnings_vnd
            return float(live_pe) if pd.notna(live_pe) and live_pe > 0 else None

        return None

    async def apply_cache_to_stocks(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """Apply currently cached data to stocks without fetching new data."""
        if not stocks:
            return []

        tickers = [s.ticker for s in stocks]
        async with async_session() as session:
            company_stmt = select(StockCompany).where(StockCompany.symbol.in_(tickers))
            company_result = await session.execute(company_stmt)
            cached_data = {c.symbol: c for c in company_result.scalars().all()}

            ratio_stmt = select(StockFinancialDataCache).where(
                StockFinancialDataCache.symbol.in_(tickers),
                StockFinancialDataCache.data_type == 'ratios',
                StockFinancialDataCache.period == 'quarter',
                StockFinancialDataCache.lang == 'en',
            )
            ratio_result = await session.execute(ratio_stmt)
            ratio_cache = {row.symbol: row for row in ratio_result.scalars().all()}

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

                ratio_cache_row = ratio_cache.get(stock.ticker)
                ratio_records = ratio_cache_row.data if ratio_cache_row is not None else None
                live_ratio_pe = self._compute_live_pe_ratio(stock, ratio_records)
                if live_ratio_pe is not None:
                    stock.pe_ratio = live_ratio_pe
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
        metadata_log_stats: Dict[str, int | bool | str] | None = None

        async with async_session() as session:
            try:
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
                        missing_set = set(tickers_needing_name)
                        rows_to_insert = []
                        for _, row in all_symbols_df.iterrows():
                            symbol = row['symbol']
                            if symbol in missing_set:
                                rows_to_insert.append({
                                    'symbol': symbol,
                                    'company_name': row['organ_name'],
                                })

                        if rows_to_insert:
                            # Idempotent insert to avoid duplicate-key races
                            stmt = insert(StockCompany).values(rows_to_insert)
                            stmt = stmt.on_conflict_do_nothing(index_elements=['symbol'])
                            await session.execute(stmt)

                            # Refresh cache for newly inserted (or concurrently inserted) rows
                            symbols = [r['symbol'] for r in rows_to_insert]
                            stmt = select(StockCompany).where(StockCompany.symbol.in_(symbols))
                            result = await session.execute(stmt)
                            for company in result.scalars().all():
                                cached_data[company.symbol] = company

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
                            requested_count = len(tickers_to_fetch)
                            metadata_log_stats = {
                                'requested_count': requested_count,
                                'processed_count': 0,
                                'updated_pe_count': 0,
                                'touched_count': 0,
                                'error_count': 0,
                                'stopped_early': False,
                                'stop_reason': "",
                            }
                            # Mark as enriching to avoid multiple tasks for same symbols
                            self._enriching_tickers.update(tickers_to_fetch)

                            try:
                                log_background_start(
                                    "Metadata Enrichment",
                                    f"{len(tickers_to_fetch)}/{len(tickers_needing_finance)} stocks"
                                )

                                # Fetch one by one and commit incrementally
                                for symbol in tickers_to_fetch:
                                    # Check rate limit on each iteration for early exit
                                    if sync_status.is_rate_limited:
                                        metadata_log_stats['stopped_early'] = True
                                        metadata_log_stats['stop_reason'] = "rate_limited"
                                        bg_logger.warning("Rate limit detected during enrichment, stopping batch")
                                        break
                                    if not api_circuit_breaker.can_proceed():
                                        metadata_log_stats['stopped_early'] = True
                                        metadata_log_stats['stop_reason'] = "circuit_open"
                                        bg_logger.warning("Rate limit detected during enrichment, stopping batch")
                                        break

                                    metadata_log_stats['processed_count'] += 1
                                    try:
                                        # Add a small delay between symbols
                                        await asyncio.sleep(1.0)
                                        data = await self._fetch_stock_finance(symbol)

                                        if data and symbol in cached_data:
                                            cached_data[symbol].pe_ratio = data.get('pe_ratio')
                                            cached_data[symbol].updated_at = now
                                            metadata_log_stats['updated_pe_count'] += 1
                                            await session.commit()
                                        elif symbol in cached_data:
                                            # Still update to avoid retrying immediately, but mark as updated now
                                            cached_data[symbol].updated_at = now
                                            metadata_log_stats['touched_count'] += 1
                                            await session.commit()
                                    except Exception as e:
                                        metadata_log_stats['error_count'] += 1
                                        bg_logger.error(f"Error enriching {symbol}: {e}")
                                        if _is_rate_limit_error(e):
                                            metadata_log_stats['stopped_early'] = True
                                            metadata_log_stats['stop_reason'] = "rate_limited"
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

                await session.commit()
            except Exception as e:
                if metadata_log_stats is not None:
                    log_background_error("Metadata Enrichment", str(e))
                raise
            else:
                if metadata_log_stats is not None:
                    summary = (
                        f"processed={metadata_log_stats['processed_count']}/{metadata_log_stats['requested_count']}, "
                        f"pe_updated={metadata_log_stats['updated_pe_count']}, "
                        f"touched={metadata_log_stats['touched_count']}, "
                        f"errors={metadata_log_stats['error_count']}, "
                        f"stopped_early={str(metadata_log_stats['stopped_early']).lower()}"
                    )
                    stop_reason = metadata_log_stats.get('stop_reason')
                    if stop_reason:
                        summary += f", reason={stop_reason}"
                    log_background_complete("Metadata Enrichment", summary)

        return stocks

    async def _fetch_stock_finance(self, symbol: str) -> Dict | None:
        """
        Fetch financial metadata for a single symbol via DB-first finance service.
        """
        if self._finance_service is None:
            return None

        try:
            ratio_records = await self._finance_service.get_financial_ratios(
                symbol=symbol[:3],
                lang='en',
            )
            pe_ratio = self._finance_service.extract_latest_pe_ratio(ratio_records)
            if pe_ratio is None:
                return None
            return {'pe_ratio': pe_ratio}
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
            listing = Listing(source='KBS')
            result = listing.all_symbols()
            api_circuit_breaker.record_success()
            return result
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching all symbols: {e}")
            raise
