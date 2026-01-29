from __future__ import annotations

from typing import List, Dict, Any
import asyncio
from datetime import datetime, date, timedelta
import pandas as pd

from sqlalchemy import select, and_
from sqlalchemy.orm import Session

from app.db.database import async_session
from app.db.models import StockDailyPrice, StockCompany
from app.services.sync_status import sync_status
from app.core.logging_config import log_background_start, log_background_complete

from .core import (
    background_executor,
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


class HistoryService:
    """Historical price and volume data operations."""

    def __init__(self) -> None:
        # Track background sync task for weekly prices
        self._weekly_prices_sync_task: asyncio.Task | None = None
        self._weekly_prices_syncing_symbols = set()

    def enrich_with_price_changes(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """
        Enrich stock data with historical price changes (1w, 1m, 1y).
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
            '1y': today - timedelta(days=365),
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
                for period in ['1w', '1m', '1y']:
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

            # 1 year change
            if (symbol, '1y') in cached_prices and cached_prices[(symbol, '1y')] > 0:
                year_price = cached_prices[(symbol, '1y')]
                stock.price_change_1y = round(((current_price_unit - year_price) / year_price) * 100, 2)

        return stocks

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
        session=None
    ) -> int:
        """
        Fetch stock history from API and store in database using upsert-like logic.
        Returns number of new records inserted.
        """
        from vnstock import Vnstock

        # Use provided session or create a temporary one
        own_session = False
        if session is None:
            engine = get_sync_engine()
            session = Session(engine)
            own_session = True

        try:
            # Fetch from API with retry logic
            def fetch_history():
                s = Vnstock().stock(symbol=symbol, source='VCI')
                return s.quote.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1D'
                )

            hist = retry_with_backoff(fetch_history, max_retries=2)

            if hist is None or hist.empty:
                return 0

            count = 0
            for _, row in hist.iterrows():
                try:
                    price_date = pd.to_datetime(row['time']).date()

                    # Check if already exists in DB
                    stmt = select(StockDailyPrice).where(
                        and_(
                            StockDailyPrice.symbol == symbol,
                            StockDailyPrice.date == price_date
                        )
                    )
                    existing = session.execute(stmt).scalar_one_or_none()

                    if not existing:
                        price_record = StockDailyPrice(
                            symbol=symbol,
                            date=price_date,
                            open=float(row.get('open', 0)) if pd.notna(row.get('open')) else None,
                            high=float(row.get('high', 0)) if pd.notna(row.get('high')) else None,
                            low=float(row.get('low', 0)) if pd.notna(row.get('low')) else None,
                            close=float(row['close']),
                            volume=int(row.get('volume', 0)) if pd.notna(row.get('volume')) else None
                        )
                        session.add(price_record)
                        count += 1
                except Exception as e:
                    bg_logger.error(f"Error processing price for {symbol} on {row.get('time')}: {e}")
                    continue

            if count > 0:
                session.commit()

            return count
        except Exception as e:
            bg_logger.error(f"Error in _upsert_stock_price_history for {symbol}: {e}")
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
        one_year_ago = today - timedelta(days=400)

        for symbol in symbols:
            # Check circuit breaker before each symbol to fail fast if rate limited
            if not api_circuit_breaker.can_proceed():
                bg_logger.warning("Circuit breaker open, skipping history fetch for remaining symbols")
                return

            try:
                count = self._upsert_stock_price_history(
                    symbol=symbol,
                    start_date=one_year_ago,
                    end_date=today,
                    session=session
                )
                if count > 0:
                    bg_logger.debug(f"Cached {count} new price records for {symbol}")

            except Exception as e:
                bg_logger.error(f"Error syncing history for {symbol}: {e}")
                continue

    async def get_volume_history(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Fetch volume history for a given stock symbol.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(frontend_executor, self._fetch_volume_history_sync, symbol, days)

    def _fetch_volume_history_sync(self, symbol: str, days: int) -> Dict[str, Any]:
        """Fetch volume history synchronously."""
        from vnstock import Vnstock

        symbol_clean = symbol[:3]
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
                start_date = end_date - timedelta(days=days + 10)  # Extra buffer for weekends/holidays

                # Query cached data
                stmt = select(StockDailyPrice).where(
                    and_(
                        StockDailyPrice.symbol == symbol_clean,
                        StockDailyPrice.date >= start_date,
                        StockDailyPrice.date <= end_date
                    )
                ).order_by(StockDailyPrice.date.desc())

                cached_records = session.execute(stmt).scalars().all()

                # If we have enough cached data, use it
                if len(cached_records) >= days:
                    data = []
                    for record in sorted(cached_records[:days], key=lambda x: x.date):
                        value = None
                        if record.volume and record.close:
                            # Calculate value in billion VND: (volume * close_price_in_1000_VND) / 1e6
                            value = (record.volume * record.close) / 1e6

                        data.append({
                            'date': record.date.strftime('%Y-%m-%d'),
                            'volume': record.volume if record.volume else 0,
                            'value': round(value, 2) if value else None
                        })

                    engine.dispose()
                    return {
                        'symbol': symbol_clean,
                        'company_name': company_name,
                        'data': data
                    }

                # Otherwise, fetch from API and cache
                # Check circuit breaker before making API call
                if not api_circuit_breaker.can_proceed():
                    raise CircuitOpenError(f"Circuit breaker open - cannot fetch volume history for {symbol_clean}")

                try:
                    s = Vnstock().stock(symbol=symbol_clean, source='VCI')
                    hist = s.quote.history(
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        interval='1D'
                    )
                    api_circuit_breaker.record_success()

                    if hist is not None and not hist.empty:
                        # Cache the data
                        for _, row in hist.iterrows():
                            try:
                                price_date = pd.to_datetime(row['time']).date()

                                # Check if already exists
                                existing = session.execute(
                                    select(StockDailyPrice).where(
                                        and_(
                                            StockDailyPrice.symbol == symbol_clean,
                                            StockDailyPrice.date == price_date
                                        )
                                    )
                                ).scalar_one_or_none()

                                if not existing:
                                    price_record = StockDailyPrice(
                                        symbol=symbol_clean,
                                        date=price_date,
                                        open=float(row.get('open', 0)) if pd.notna(row.get('open')) else None,
                                        high=float(row.get('high', 0)) if pd.notna(row.get('high')) else None,
                                        low=float(row.get('low', 0)) if pd.notna(row.get('low')) else None,
                                        close=float(row['close']),
                                        volume=int(row.get('volume', 0)) if pd.notna(row.get('volume')) else None
                                    )
                                    session.add(price_record)
                            except Exception as e:
                                bg_logger.error(f"Error caching price for {symbol_clean} on {row.get('time')}: {e}")
                                continue

                        session.commit()

                        # Convert to response format
                        data = []
                        hist_sorted = hist.sort_values('time', ascending=True).tail(days)

                        for _, row in hist_sorted.iterrows():
                            volume = int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0
                            close = float(row['close']) if pd.notna(row['close']) else 0
                            value = None
                            if volume and close:
                                # Calculate value in billion VND
                                value = (volume * close) / 1e6

                            data.append({
                                'date': pd.to_datetime(row['time']).strftime('%Y-%m-%d'),
                                'volume': volume,
                                'value': round(value, 2) if value else None
                            })

                        engine.dispose()
                        return {
                            'symbol': symbol_clean,
                            'company_name': company_name,
                            'data': data
                        }

                except (SystemExit, Exception) as e:
                    if _is_rate_limit_error(e):
                        _record_rate_limit(reset_seconds=30.0)
                        raise CircuitOpenError(f"Rate limited fetching volume history for {symbol_clean}: {e}")
                    logger.warning(f"Error fetching volume history for {symbol_clean}: {e}")

                engine.dispose()
                return {
                    'symbol': symbol_clean,
                    'company_name': company_name,
                    'data': []
                }

        except CircuitOpenError:
            raise  # Re-raise circuit breaker errors
        except Exception as e:
            logger.warning(f"Error in volume history fetch: {e}")
            engine.dispose()
            return {
                'symbol': symbol_clean,
                'company_name': company_name,
                'data': []
            }

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
        clean_symbols = [s[:3] for s in symbols]

        # Calculate date range
        start_date = date(start_year, 1, 1)
        end_date = date.today()

        # Load from database
        stocks_data = await self._load_weekly_prices_from_db(clean_symbols, start_date, end_date)

        # Check staleness and historical data coverage
        is_stale = self._check_prices_staleness(stocks_data, start_date, end_date)

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

        # Trigger background sync if stale
        is_syncing = False
        if is_stale:
            is_syncing = await self._trigger_price_history_sync(clean_symbols, start_date, end_date)

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
        start_date: date,
        end_date: date
    ) -> bool:
        """
        Check if price data is stale or incomplete.
        Returns True if any stock:
        - Has no data
        - Latest date is >7 days old (stale)
        - Earliest date is >30 days after requested start_date (incomplete historical data)
        """
        if not stocks_data:
            return True

        stale_threshold = end_date - timedelta(days=7)
        # Allow 30 days tolerance for start date (some stocks may not have been listed that early)
        start_threshold = start_date + timedelta(days=30)

        for symbol, prices in stocks_data.items():
            if not prices:
                return True

            # Check latest date (data freshness)
            latest_date_str = prices[-1]['date']
            latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d').date()

            if latest_date < stale_threshold:
                return True

            # Check earliest date (historical data coverage)
            earliest_date_str = prices[0]['date']
            earliest_date = datetime.strptime(earliest_date_str, '%Y-%m-%d').date()

            if earliest_date > start_threshold:
                # Data doesn't cover the requested start date range
                return True

        return False

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
        end_date: date
    ) -> bool:
        """
        Trigger background sync for price history.
        Returns True if sync was triggered, False if already syncing.
        """
        # Check if already syncing these symbols
        symbols_to_sync = [s for s in symbols if s not in self._weekly_prices_syncing_symbols]

        if not symbols_to_sync:
            return True  # Already syncing

        # Mark as syncing
        self._weekly_prices_syncing_symbols.update(symbols_to_sync)

        # Create background task
        asyncio.create_task(
            self._sync_price_history_background(symbols_to_sync, start_date, end_date)
        )

        return True

    async def _sync_price_history_background(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> None:
        """
        Background task to sync price history for given symbols.
        Fetches from vnstock API and stores in database.
        """
        # Early bail-out if rate limited
        if sync_status.is_rate_limited:
            bg_logger.warning("Skipping price history sync due to rate limit")
            for symbol in symbols:
                self._weekly_prices_syncing_symbols.discard(symbol)
            return

        log_background_start("Price History Sync", f"{len(symbols)} symbols")

        loop = asyncio.get_event_loop()

        try:
            # Sync in batches to avoid overwhelming the API
            batch_size = 10
            for i in range(0, len(symbols), batch_size):
                # Check rate limit on each batch
                if sync_status.is_rate_limited or not api_circuit_breaker.can_proceed():
                    bg_logger.warning("Rate limit detected during price sync, stopping early")
                    break

                batch = symbols[i:i + batch_size]

                for symbol in batch:
                    # Check rate limit on each symbol for faster exit
                    if sync_status.is_rate_limited or not api_circuit_breaker.can_proceed():
                        bg_logger.warning("Rate limit detected, stopping price sync")
                        break
                    try:
                        await loop.run_in_executor(
                            background_executor,
                            self._fetch_and_store_stock_history,
                            symbol,
                            start_date,
                            end_date
                        )
                        # Small delay between symbols
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        bg_logger.error(f"Error syncing {symbol}: {e}")
                        # Check if it's a rate limit error
                        if _is_rate_limit_error(e):
                            _record_rate_limit(reset_seconds=30.0)
                            bg_logger.warning("Rate limit hit, stopping price sync")
                            break

                # Longer delay between batches
                if i + batch_size < len(symbols):
                    await asyncio.sleep(2.0)

            log_background_complete("Price History Sync", f"{len(symbols)} symbols processed")
        finally:
            # Clear syncing status
            for symbol in symbols:
                self._weekly_prices_syncing_symbols.discard(symbol)

    def _fetch_and_store_stock_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> None:
        """Fetch stock history from API and store in database using unified helper."""
        try:
            count = self._upsert_stock_price_history(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            if count > 0:
                bg_logger.debug(f"Synced {count} price records for {symbol}")
        except Exception as e:
            bg_logger.error(f"Error in background sync for {symbol}: {e}")
        finally:
            # Clean up connections
            engine = get_sync_engine()
            engine.dispose()
