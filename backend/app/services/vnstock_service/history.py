from __future__ import annotations

from typing import List, Dict, Any, Callable, Awaitable, Set
import asyncio
import threading
from datetime import datetime, date, timedelta
import pandas as pd

from sqlalchemy import select, and_
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.database import async_session
from app.db.models import StockDailyPrice, StockCompany, StockPriceSyncState
from app.services.sync_status import sync_status

from .core import (
    frontend_executor,
    logger,
    bg_logger,
    api_circuit_breaker,
    CircuitOpenError,
    retry_with_backoff,
    _record_rate_limit,
    _is_rate_limit_error,
    get_sync_engine,
)
from .models import StockInfo


# Weekly price history defaults
REQUEST_HISTORY_SYNC_TIMEOUT_SECONDS = 12.0


class HistoryService:
    """Historical price and volume data operations."""

    def __init__(self) -> None:
        self._weekly_prices_retry_cooldown = timedelta(minutes=10)
        self._symbol_sync_locks: Dict[str, threading.Lock] = {}
        self._symbol_sync_locks_guard = threading.Lock()
        self._request_sync_handler: Callable[[str, date, date, float], Awaitable[Dict[str, Any]]] | None = None
        self._weekly_sync_trigger_handler: Callable[[List[str]], Awaitable[Dict[str, Any]]] | None = None

    def set_on_demand_history_sync_handler(
        self,
        handler: Callable[[str, date, date, float], Awaitable[Dict[str, Any]]]
    ) -> None:
        self._request_sync_handler = handler

    def set_weekly_history_sync_trigger_handler(
        self,
        handler: Callable[[List[str]], Awaitable[Dict[str, Any]]]
    ) -> None:
        self._weekly_sync_trigger_handler = handler

    def _get_symbol_sync_lock(self, symbol: str) -> threading.Lock:
        symbol_key = symbol[:3].upper()
        with self._symbol_sync_locks_guard:
            lock = self._symbol_sync_locks.get(symbol_key)
            if lock is None:
                lock = threading.Lock()
                self._symbol_sync_locks[symbol_key] = lock
            return lock

    def _normalize_price_history_payload(
        self,
        symbol: str,
        hist: pd.DataFrame,
        created_at: datetime | None = None
    ) -> tuple[List[Dict[str, Any]], date | None, date | None]:
        symbol_key = symbol[:3].upper()
        row_created_at = created_at or datetime.utcnow()
        deduped_by_date: Dict[date, Dict[str, Any]] = {}

        for _, row in hist.iterrows():
            raw_time = row.get('time')
            if pd.isna(raw_time):
                continue
            raw_close = row.get('close')
            if pd.isna(raw_close):
                continue
            try:
                price_date = pd.to_datetime(raw_time).date()
            except Exception:
                continue

            deduped_by_date[price_date] = {
                'symbol': symbol_key,
                'date': price_date,
                'open': float(row.get('open', 0)) if pd.notna(row.get('open')) else None,
                'high': float(row.get('high', 0)) if pd.notna(row.get('high')) else None,
                'low': float(row.get('low', 0)) if pd.notna(row.get('low')) else None,
                'close': float(raw_close),
                'volume': int(row.get('volume', 0)) if pd.notna(row.get('volume')) else None,
                'created_at': row_created_at,
            }

        if not deduped_by_date:
            return [], None, None

        ordered_dates = sorted(deduped_by_date.keys())
        payload = [deduped_by_date[d] for d in ordered_dates]
        return payload, ordered_dates[0], ordered_dates[-1]

    async def start_background_workers(self) -> None:
        """No background workers are started for history service."""
        return

    async def stop_background_workers(self) -> None:
        """No-op hook kept for lifecycle symmetry."""
        return

    def enrich_with_price_changes(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """
        Enrich stock data with historical price changes (1w, 1m, 6m, 1y, 2y, 3y).
        """
        return self.enrich_with_price_changes_sync(stocks)

    def enrich_with_price_changes_sync(
        self,
        stocks: List[StockInfo],
        fetch_missing_history: bool = False
    ) -> List[StockInfo]:
        """
        Synchronous fallback for price change enrichment.
        Queries DB cache directly, fetches missing from API.
        """
        # Calculate target dates
        today = datetime.now().date()
        target_dates = {
            '1w': today - timedelta(days=7),
            '1m': today - timedelta(days=30),
            '6m': today - timedelta(days=182),
            '1y': today - timedelta(days=365),
            '2y': today - timedelta(days=730),
            '3y': today - timedelta(days=1095),
        }

        # Use sync connection for DB lookup
        engine = get_sync_engine()

        symbols = [s.ticker[:3] for s in stocks]

        with Session(engine) as session:
            # Get cached prices for all symbols at target dates (with some tolerance)
            cached_prices = self._get_cached_prices_sync(session, symbols, target_dates)

            # Find symbols missing cache data
            symbols_needing_fetch = set()
            for symbol in symbols:
                for period in target_dates.keys():
                    if (symbol, period) not in cached_prices:
                        symbols_needing_fetch.add(symbol)

            # Fetch missing data from API and save to DB
            if symbols_needing_fetch:
                # Limit how many symbols we fetch history for in one request
                # To avoid hitting API limits and long timeouts
                if fetch_missing_history and not sync_status.is_rate_limited and api_circuit_breaker.can_proceed():
                    max_history_fetch = 100
                    symbols_to_fetch = list(symbols_needing_fetch)[:max_history_fetch]
                    logger.info(f"Fetching historical data for {len(symbols_to_fetch)}/{len(symbols_needing_fetch)} symbols")
                    self._fetch_and_cache_history_sync(session, symbols_to_fetch)
                    # Re-query cached prices after fetch
                    cached_prices = self._get_cached_prices_sync(session, symbols, target_dates)
                else:
                    bg_logger.debug("Skipping historical price fetch in request path")

        engine.dispose()

        # Calculate price changes from cached data
        for stock in stocks:
            symbol = stock.ticker[:3]
            # Convert current price to same units as history (1,000 VND)
            current_price_unit = stock.price / 1000

            # 1 week change
            if (symbol, '1w') in cached_prices and cached_prices[(symbol, '1w')] > 0:
                week_price = cached_prices[(symbol, '1w')]
                stock.price_change_1w = round(((current_price_unit - week_price) / week_price) * 100, 2)

            # 1 month change
            if (symbol, '1m') in cached_prices and cached_prices[(symbol, '1m')] > 0:
                month_price = cached_prices[(symbol, '1m')]
                stock.price_change_1m = round(((current_price_unit - month_price) / month_price) * 100, 2)

            # 6 month change
            if (symbol, '6m') in cached_prices and cached_prices[(symbol, '6m')] > 0:
                six_month_price = cached_prices[(symbol, '6m')]
                stock.price_change_6m = round(((current_price_unit - six_month_price) / six_month_price) * 100, 2)

            # 1 year change
            if (symbol, '1y') in cached_prices and cached_prices[(symbol, '1y')] > 0:
                year_price = cached_prices[(symbol, '1y')]
                stock.price_change_1y = round(((current_price_unit - year_price) / year_price) * 100, 2)

            # 2 year change
            if (symbol, '2y') in cached_prices and cached_prices[(symbol, '2y')] > 0:
                two_year_price = cached_prices[(symbol, '2y')]
                stock.price_change_2y = round(((current_price_unit - two_year_price) / two_year_price) * 100, 2)

            # 3 year change
            if (symbol, '3y') in cached_prices and cached_prices[(symbol, '3y')] > 0:
                three_year_price = cached_prices[(symbol, '3y')]
                stock.price_change_3y = round(((current_price_unit - three_year_price) / three_year_price) * 100, 2)

        return stocks

    async def trigger_missing_price_history_sync(self, stocks: List[StockInfo]) -> bool:
        """
        Deprecated in deterministic sync mode.
        Request-path historical sync is disabled to keep write flow deterministic.
        """
        return False

    def _get_cached_prices_sync(self, session, symbols: List[str], target_dates: Dict[str, date]) -> Dict[tuple, float]:
        """
        Get cached prices for given symbols at target dates.
        Returns dict of (symbol, period) -> close_price
        """
        result: Dict[tuple, float] = {}

        for period, target_date in target_dates.items():
            # Look for prices within 7 days of target (to handle weekends/holidays)
            min_date = target_date - timedelta(days=7)
            max_date = target_date + timedelta(days=1)

            stmt = select(StockDailyPrice).where(
                and_(
                    StockDailyPrice.symbol.in_(symbols),
                    StockDailyPrice.date >= min_date,
                    StockDailyPrice.date <= max_date
                )
            ).order_by(StockDailyPrice.date.desc())

            rows = session.execute(stmt).scalars().all()

            # Group by symbol and take the closest to target date
            symbol_prices: Dict[str, float] = {}
            for row in rows:
                if row.symbol not in symbol_prices:
                    symbol_prices[row.symbol] = row.close

            for symbol, close in symbol_prices.items():
                result[(symbol, period)] = close

        return result

    def _upsert_stock_price_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        session=None,
        raise_on_error: bool = False,
        log=None,
    ) -> int:
        """
        Fetch stock history from API and store in database via atomic upsert.
        Returns number of rows synced (inserted or updated).
        """
        from vnstock import Vnstock

        # Use provided session or create a temporary one
        own_session = False
        if session is None:
            engine = get_sync_engine()
            session = Session(engine)
            own_session = True

        symbol_key = symbol[:3].upper()
        symbol_lock = self._get_symbol_sync_lock(symbol_key)
        active_logger = log or bg_logger

        try:
            with symbol_lock:
                # Fetch from API with retry logic
                def fetch_history():
                    s = Vnstock().stock(symbol=symbol_key, source='VCI')
                    return s.quote.history(
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        interval='1D'
                    )

                hist = retry_with_backoff(fetch_history, max_retries=2)

                if hist is None or hist.empty:
                    return 0

                payload, min_payload_date, max_payload_date = self._normalize_price_history_payload(
                    symbol=symbol_key,
                    hist=hist
                )
                if not payload:
                    return 0

                insert_stmt = pg_insert(StockDailyPrice.__table__).values(payload)
                upsert_stmt = insert_stmt.on_conflict_do_update(
                    constraint='uq_symbol_date',
                    set_={
                        'open': insert_stmt.excluded.open,
                        'high': insert_stmt.excluded.high,
                        'low': insert_stmt.excluded.low,
                        'close': insert_stmt.excluded.close,
                        'volume': insert_stmt.excluded.volume,
                    }
                )

                session.execute(upsert_stmt)
                session.commit()
                count = len(payload)
                active_logger.debug(
                    f"Upserted {count} price records for {symbol_key} "
                    f"({min_payload_date} -> {max_payload_date})"
                )
                return count
        except Exception as e:
            try:
                session.rollback()
            except Exception:
                pass
            active_logger.error(
                f"Error in _upsert_stock_price_history for {symbol_key} "
                f"({start_date} -> {end_date}): {e}"
            )
            if raise_on_error:
                raise
            return 0
        finally:
            if own_session:
                session.close()

    def _fetch_and_cache_history_sync(self, session, symbols: List[str]) -> None:
        """
        Fetch historical data for given symbols from vnstock API and cache to DB.
        Optimized version using unified upsert helper.
        """
        today = date.today()
        three_years_ago = today - timedelta(days=1130)

        for symbol in symbols:
            # Check circuit breaker before each symbol to fail fast if rate limited
            if not api_circuit_breaker.can_proceed():
                bg_logger.warning("Circuit breaker open, skipping history fetch for remaining symbols")
                return

            try:
                count = self._upsert_stock_price_history(
                    symbol=symbol,
                    start_date=three_years_ago,
                    end_date=today,
                    session=session
                )
                if count > 0:
                    bg_logger.debug(f"Cached {count} synced price records for {symbol}")

            except Exception as e:
                try:
                    session.rollback()
                except Exception:
                    pass
                bg_logger.error(f"Error syncing history for {symbol}: {e}")
                continue

    @staticmethod
    def _default_request_sync_metadata() -> Dict[str, Any]:
        return {
            "sync_performed": False,
            "sync_timed_out": False,
            "sync_error": None,
            "updated_through": None,
            "repaired_missing_dates": 0,
        }

    async def _sync_history_for_request(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        if self._request_sync_handler is None:
            return self._default_request_sync_metadata()

        try:
            result = await self._request_sync_handler(
                symbol,
                start_date,
                end_date,
                REQUEST_HISTORY_SYNC_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.warning(
                f"Error running request-path history sync for {symbol}: {e}"
            )
            fallback = self._default_request_sync_metadata()
            fallback["sync_error"] = str(e)[:500]
            return fallback

        merged = self._default_request_sync_metadata()
        if isinstance(result, dict):
            for key in merged.keys():
                if key in result:
                    merged[key] = result[key]
        return merged

    async def get_volume_history(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Fetch volume history for a given stock symbol.
        """
        symbol_clean = symbol[:3].upper()
        try:
            safe_days = max(1, int(days))
        except (TypeError, ValueError):
            safe_days = 30

        end_date = date.today()
        start_date = end_date - timedelta(days=safe_days - 1)
        sync_meta = await self._sync_history_for_request(
            symbol=symbol_clean,
            start_date=start_date,
            end_date=end_date,
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            frontend_executor,
            self._fetch_volume_history_sync,
            symbol_clean,
            safe_days
        )
        result.update(sync_meta)
        return result

    async def get_price_history(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Fetch price history for a given stock symbol.
        """
        symbol_clean = symbol[:3].upper()
        try:
            safe_days = max(1, int(days))
        except (TypeError, ValueError):
            safe_days = 30

        end_date = date.today()
        start_date = end_date - timedelta(days=safe_days - 1)
        sync_meta = await self._sync_history_for_request(
            symbol=symbol_clean,
            start_date=start_date,
            end_date=end_date,
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            frontend_executor,
            self._fetch_price_history_sync,
            symbol_clean,
            safe_days
        )
        result.update(sync_meta)
        return result

    async def get_price_history_ohlcv(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch full OHLCV history for a given stock symbol from database cache.
        """
        symbol_clean = symbol[:3].upper()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            frontend_executor,
            self._fetch_price_history_ohlcv_sync,
            symbol_clean,
        )

    async def get_stocks_volume_series(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> Dict[str, Any]:
        """
        Fetch volume-value time series for multiple symbols within a date range.
        """
        clean_symbols = list(dict.fromkeys(s[:3].upper() for s in symbols if s))

        if end_date < start_date:
            start_date, end_date = end_date, start_date

        today = date.today()
        bounded_end = min(end_date, today)
        bounded_start = min(start_date, bounded_end)

        if not clean_symbols:
            return {
                "stocks": [],
                "start_date": bounded_start.strftime("%Y-%m-%d"),
                "end_date": bounded_end.strftime("%Y-%m-%d"),
                "is_stale": False,
                "is_syncing": False,
            }

        async with async_session() as session:
            stmt = select(StockDailyPrice).where(
                and_(
                    StockDailyPrice.symbol.in_(clean_symbols),
                    StockDailyPrice.date >= bounded_start,
                    StockDailyPrice.date <= bounded_end,
                )
            ).order_by(StockDailyPrice.symbol.asc(), StockDailyPrice.date.asc())
            result = await session.execute(stmt)
            records = result.scalars().all()

        company_names = await self._get_company_names(clean_symbols)
        series_by_symbol: Dict[str, List[Dict[str, Any]]] = {symbol: [] for symbol in clean_symbols}

        for record in records:
            value = None
            if record.volume is not None and record.close is not None:
                value = round((record.volume * record.close) / 1e6, 2)

            series_by_symbol.setdefault(record.symbol, []).append({
                "date": record.date.strftime("%Y-%m-%d"),
                "value": value,
            })

        stale_check_payload = {
            symbol: [{"date": point["date"]} for point in points]
            for symbol, points in series_by_symbol.items()
        }
        stale_symbols = self._get_symbols_with_stale_latest(
            stocks_data=stale_check_payload,
            requested_symbols=clean_symbols,
            end_date=bounded_end,
        )
        is_stale = len(stale_symbols) > 0
        is_syncing_symbols = await self._get_symbols_currently_syncing(clean_symbols)
        is_syncing = len(is_syncing_symbols) > 0

        if stale_symbols:
            triggered = await self._trigger_price_history_sync(
                symbols=stale_symbols,
                start_date=bounded_start,
                end_date=bounded_end,
                force=False,
            )
            if triggered:
                is_syncing = True

        stocks_response = [
            {
                "symbol": symbol,
                "ticker": symbol,
                "company_name": company_names.get(symbol, symbol),
                "data": series_by_symbol.get(symbol, []),
            }
            for symbol in clean_symbols
        ]

        return {
            "stocks": stocks_response,
            "start_date": bounded_start.strftime("%Y-%m-%d"),
            "end_date": bounded_end.strftime("%Y-%m-%d"),
            "is_stale": is_stale,
            "is_syncing": is_syncing,
        }

    def _fetch_volume_history_sync(self, symbol: str, days: int) -> Dict[str, Any]:
        """Fetch volume history synchronously."""
        symbol_clean = symbol[:3].upper()
        try:
            safe_days = max(1, int(days))
        except (TypeError, ValueError):
            safe_days = 30
        company_name = symbol_clean

        # Use sync connection for DB lookup
        engine = get_sync_engine()

        try:
            with Session(engine) as session:
                # Get company name
                stmt = select(StockCompany).where(StockCompany.symbol == symbol_clean)
                company = session.execute(stmt).scalar_one_or_none()
                if company:
                    company_name = company.company_name

                # Calculate date range
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=safe_days - 1)

                # Query cached data
                stmt = select(StockDailyPrice).where(
                    and_(
                        StockDailyPrice.symbol == symbol_clean,
                        StockDailyPrice.date >= start_date,
                        StockDailyPrice.date <= end_date
                    )
                ).order_by(StockDailyPrice.date.asc())

                cached_records = session.execute(stmt).scalars().all()

                data = []
                for record in cached_records:
                    value = None
                    if record.volume and record.close:
                        # Calculate value in billion VND: (volume * close_price_in_1000_VND) / 1e6
                        value = (record.volume * record.close) / 1e6

                    data.append({
                        'date': record.date.strftime('%Y-%m-%d'),
                        'volume': record.volume if record.volume else 0,
                        'value': round(value, 2) if value else None
                    })

                return {
                    'symbol': symbol_clean,
                    'company_name': company_name,
                    'data': data
                }
        except Exception as e:
            logger.warning(f"Error in volume history fetch: {e}")
            return {
                'symbol': symbol_clean,
                'company_name': company_name,
                'data': []
            }
        finally:
            engine.dispose()

    def _fetch_price_history_sync(self, symbol: str, days: int) -> Dict[str, Any]:
        """Fetch price history synchronously."""
        symbol_clean = symbol[:3].upper()
        try:
            safe_days = max(1, int(days))
        except (TypeError, ValueError):
            safe_days = 30
        company_name = symbol_clean

        # Use sync connection for DB lookup
        engine = get_sync_engine()

        try:
            with Session(engine) as session:
                # Get company name
                stmt = select(StockCompany).where(StockCompany.symbol == symbol_clean)
                company = session.execute(stmt).scalar_one_or_none()
                if company:
                    company_name = company.company_name

                # Calculate date range
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=safe_days - 1)

                # Query cached data
                stmt = select(StockDailyPrice).where(
                    and_(
                        StockDailyPrice.symbol == symbol_clean,
                        StockDailyPrice.date >= start_date,
                        StockDailyPrice.date <= end_date
                    )
                ).order_by(StockDailyPrice.date.asc())

                cached_records = session.execute(stmt).scalars().all()

                data = [
                    {
                        'date': record.date.strftime('%Y-%m-%d'),
                        # StockDailyPrice.close is stored in 1,000 VND; convert to VND for UI parity.
                        'close': round(record.close * 1000, 2),
                    }
                    for record in cached_records
                ]

                return {
                    'symbol': symbol_clean,
                    'company_name': company_name,
                    'data': data
                }
        except Exception as e:
            logger.warning(f"Error in price history fetch: {e}")
            return {
                'symbol': symbol_clean,
                'company_name': company_name,
                'data': []
            }
        finally:
            engine.dispose()

    def _fetch_price_history_ohlcv_sync(self, symbol: str) -> Dict[str, Any]:
        """Fetch full OHLCV history synchronously from DB."""
        symbol_clean = symbol[:3].upper()
        company_name = symbol_clean

        engine = get_sync_engine()

        try:
            with Session(engine) as session:
                stmt = select(StockCompany).where(StockCompany.symbol == symbol_clean)
                company = session.execute(stmt).scalar_one_or_none()
                if company:
                    company_name = company.company_name

                stmt = select(StockDailyPrice).where(
                    StockDailyPrice.symbol == symbol_clean
                ).order_by(StockDailyPrice.date.desc())
                cached_records = session.execute(stmt).scalars().all()

                data = [
                    {
                        'date': record.date.strftime('%Y-%m-%d'),
                        'open': record.open,
                        'high': record.high,
                        'low': record.low,
                        'close': record.close,
                        'volume': record.volume,
                    }
                    for record in cached_records
                ]

                return {
                    'symbol': symbol_clean,
                    'company_name': company_name,
                    'data': data,
                }
        except Exception as e:
            logger.warning(f"Error in OHLCV history fetch: {e}")
            return {
                'symbol': symbol_clean,
                'company_name': company_name,
                'data': [],
            }
        finally:
            engine.dispose()

    async def get_stocks_weekly_prices(
        self,
        symbols: List[str],
        start_year: int,
        include_benchmarks: bool = True
    ) -> Dict[str, Any]:
        """
        Get weekly price data for multiple stocks.
        Returns cached data immediately and triggers background sync if stale.
        """
        # Clean symbols (use first 3 chars)
        clean_symbols = list(dict.fromkeys(s[:3].upper() for s in symbols if s))

        # Calculate date range
        start_date = date(start_year, 1, 1)
        end_date = date.today()
        # Load from database
        stocks_data = await self._load_weekly_prices_from_db(clean_symbols, start_date, end_date)

        # Check freshness from latest locally available date only.
        stale_latest_symbols = self._get_symbols_with_stale_latest(stocks_data, clean_symbols, end_date)
        is_stale = len(stale_latest_symbols) > 0

        # Load benchmarks if requested
        benchmarks = {}
        if include_benchmarks:
            benchmarks = await self._load_benchmark_prices(start_date, end_date)

        # Get company names
        company_names = await self._get_company_names(clean_symbols)

        # Format response
        stocks_response = []
        for symbol in clean_symbols:
            prices = stocks_data.get(symbol, [])
            stocks_response.append({
                'symbol': symbol,
                'ticker': symbol,
                'company_name': company_names.get(symbol, symbol),
                'prices': prices
            })

        is_syncing_symbols = await self._get_symbols_currently_syncing(clean_symbols)
        is_syncing = len(is_syncing_symbols) > 0

        if stale_latest_symbols:
            triggered = await self._trigger_price_history_sync(
                symbols=stale_latest_symbols,
                start_date=start_date,
                end_date=end_date,
                force=False
            )
            if triggered:
                is_syncing = True

        return {
            'stocks': stocks_response,
            'benchmarks': benchmarks,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'is_stale': is_stale,
            'is_syncing': is_syncing
        }

    async def _get_company_names(self, symbols: List[str]) -> Dict[str, str]:
        """Get company names for given symbols from database."""
        async with async_session() as session:
            stmt = select(StockCompany).where(StockCompany.symbol.in_(symbols))
            result = await session.execute(stmt)
            companies = result.scalars().all()
            return {c.symbol: c.company_name for c in companies}

    async def _load_weekly_prices_from_db(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load daily prices from database and aggregate to weekly.
        Uses Friday as the weekly reference point.
        """
        async with async_session() as session:
            stmt = select(StockDailyPrice).where(
                and_(
                    StockDailyPrice.symbol.in_(symbols),
                    StockDailyPrice.date >= start_date,
                    StockDailyPrice.date <= end_date
                )
            ).order_by(StockDailyPrice.symbol, StockDailyPrice.date)

            result = await session.execute(stmt)
            records = result.scalars().all()

            # Group by symbol
            symbol_data: Dict[str, List[Dict[str, Any]]] = {}
            for record in records:
                if record.symbol not in symbol_data:
                    symbol_data[record.symbol] = []
                symbol_data[record.symbol].append({
                    'date': record.date,
                    'close': record.close
                })

            # Aggregate to weekly using pandas
            weekly_data: Dict[str, List[Dict[str, Any]]] = {}
            for symbol, daily_prices in symbol_data.items():
                if not daily_prices:
                    weekly_data[symbol] = []
                    continue

                df = pd.DataFrame(daily_prices)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()

                # Resample to weekly (Friday close) - 'W-FRI' means week ending on Friday
                weekly_df = df.resample('W-FRI').last().dropna()

                # Ensure we strictly respect the start_date after resampling
                weekly_df = weekly_df[weekly_df.index >= pd.Timestamp(start_date)]

                weekly_data[symbol] = [
                    {
                        'date': idx.strftime('%Y-%m-%d'),
                        'close': float(row['close'])
                    }
                    for idx, row in weekly_df.iterrows()
                ]

            return weekly_data

    def _check_prices_staleness(
        self,
        stocks_data: Dict[str, List[Dict[str, Any]]],
        requested_symbols: List[str],
        end_date: date
    ) -> bool:
        """
        Check if price data is stale for requested symbols.
        Returns True if any requested stock:
        - Has no data
        - Latest date is >7 days old
        """
        return len(self._get_symbols_with_stale_latest(stocks_data, requested_symbols, end_date)) > 0

    def _get_symbols_with_stale_latest(
        self,
        stocks_data: Dict[str, List[Dict[str, Any]]],
        requested_symbols: List[str],
        end_date: date
    ) -> List[str]:
        if not requested_symbols:
            return []

        stale_symbols: List[str] = []
        stale_threshold = end_date - timedelta(days=7)

        for symbol in requested_symbols:
            prices = stocks_data.get(symbol, [])
            if not prices:
                stale_symbols.append(symbol)
                continue

            latest_date_str = prices[-1]['date']
            try:
                latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d').date()
            except (TypeError, ValueError):
                stale_symbols.append(symbol)
                continue

            if latest_date < stale_threshold:
                stale_symbols.append(symbol)

        return stale_symbols

    async def _load_benchmark_prices(
        self,
        start_date: date,
        end_date: date
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load weekly prices for VNINDEX and VN30 benchmarks.
        Fetches from vnstock API since these are index values, not stock prices.
        """
        benchmarks = {}
        loop = asyncio.get_event_loop()

        for index_symbol in ['VNINDEX', 'VN30']:
            try:
                prices = await loop.run_in_executor(
                    None,
                    self._fetch_index_history_sync,
                    index_symbol,
                    start_date,
                    end_date
                )
                if prices:
                    benchmarks[index_symbol] = prices
            except Exception as e:
                logger.warning(f"Error fetching benchmark {index_symbol}: {e}")

        return benchmarks

    def _fetch_index_history_sync(
        self,
        index_symbol: str,
        start_date: date,
        end_date: date
    ) -> List[Dict[str, Any]]:
        """Fetch historical index values and aggregate to weekly."""
        from vnstock import Vnstock

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch index history for {index_symbol}")

        try:
            vs = Vnstock(symbol=index_symbol, source='VCI')
            stock = vs.stock()
            df = stock.quote.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1D'
            )
            api_circuit_breaker.record_success()

            if df is not None and not df.empty:
                # Convert to proper format
                df['date'] = pd.to_datetime(df['time'])
                df = df.set_index('date').sort_index()

                # Resample to weekly (Friday close)
                weekly_df = df[['close']].resample('W-FRI').last().dropna()

                # Ensure we strictly respect the start_date after resampling
                weekly_df = weekly_df[weekly_df.index >= pd.Timestamp(start_date)]

                return [
                    {
                        'date': idx.strftime('%Y-%m-%d'),
                        'close': float(row['close'])
                    }
                    for idx, row in weekly_df.iterrows()
                ]
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching index history for {index_symbol}: {e}")
            logger.warning(f"Error fetching index history for {index_symbol}: {e}")

        return []

    async def _trigger_price_history_sync(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        force: bool = False
    ) -> bool:
        """
        Trigger delegated price sync for stale symbols.
        Returns True if sync was triggered or currently running.
        """
        if not symbols:
            return False

        normalized_symbols = list(dict.fromkeys(s[:3].upper() for s in symbols if s))
        if not normalized_symbols:
            return False

        now = datetime.utcnow()
        cooldown_state: Dict[str, datetime] = {}
        syncing_symbols = await self._get_symbols_currently_syncing(normalized_symbols)
        if not force:
            cooldown_state = await self._get_weekly_sync_cooldown_state(normalized_symbols)

        symbols_to_sync: List[str] = []
        for symbol in normalized_symbols:
            if symbol in syncing_symbols:
                continue

            if not force:
                last_attempt = cooldown_state.get(symbol)
                if last_attempt and (now - last_attempt) < self._weekly_prices_retry_cooldown:
                    continue

            symbols_to_sync.append(symbol)

        if not symbols_to_sync:
            return len(syncing_symbols) > 0

        if self._weekly_sync_trigger_handler is None:
            logger.warning("Weekly sync trigger handler is not configured; skipping delegated sync")
            return len(syncing_symbols) > 0

        await self._set_weekly_sync_cooldown_state(
            symbols=symbols_to_sync,
            attempted_at=now,
        )

        try:
            result = await self._weekly_sync_trigger_handler(symbols_to_sync)
        except Exception as e:
            logger.warning(f"Error triggering delegated weekly price sync: {e}")
            return len(syncing_symbols) > 0

        if not isinstance(result, dict):
            return True

        started = bool(result.get("started"))
        state = str(result.get("state", "")).strip().lower()
        return started or state == "running"

    async def _get_weekly_sync_cooldown_state(
        self,
        symbols: List[str]
    ) -> Dict[str, datetime]:
        if not symbols:
            return {}

        try:
            async with async_session() as session:
                stmt = select(
                    StockPriceSyncState.symbol,
                    StockPriceSyncState.weekly_sync_last_attempt_at,
                ).where(StockPriceSyncState.symbol.in_(symbols))
                result = await session.execute(stmt)
                rows = result.all()

            state: Dict[str, datetime] = {}
            for symbol, last_attempt_at in rows:
                if last_attempt_at is not None:
                    state[symbol] = last_attempt_at
            return state
        except Exception as e:
            logger.warning(f"Error loading weekly sync cooldown state: {e}")
            return {}

    async def _set_weekly_sync_cooldown_state(
        self,
        symbols: List[str],
        attempted_at: datetime,
    ) -> None:
        if not symbols:
            return

        try:
            async with async_session() as session:
                stmt = select(StockPriceSyncState).where(
                    StockPriceSyncState.symbol.in_(symbols)
                )
                existing_rows = await session.execute(stmt)
                existing = {row.symbol: row for row in existing_rows.scalars().all()}

                for symbol in symbols:
                    state = existing.get(symbol)
                    if state is None:
                        state = StockPriceSyncState(symbol=symbol)
                        session.add(state)
                        existing[symbol] = state

                    state.weekly_sync_last_attempt_at = attempted_at
                    state.updated_at = attempted_at

                await session.commit()
        except Exception as e:
            logger.warning(f"Error updating weekly sync cooldown state: {e}")

    async def _get_symbols_currently_syncing(self, symbols: List[str]) -> Set[str]:
        if not symbols:
            return set()

        # Ignore stale DB "running" flags when price sync runtime is not active.
        if not sync_status.price_sync.is_running:
            return set()

        try:
            async with async_session() as session:
                stmt = select(StockPriceSyncState.symbol).where(
                    and_(
                        StockPriceSyncState.symbol.in_(symbols),
                        StockPriceSyncState.sync_status == "running",
                    )
                )
                result = await session.execute(stmt)
                return {row[0] for row in result.all() if row[0]}
        except Exception as e:
            logger.warning(f"Error loading currently syncing symbols: {e}")
            return set()
