"""
Service for interacting with vnstock library to fetch Vietnam stock market data.

Architecture:
- Frontend executor: Handles user-facing API calls (larger pool for responsiveness)
- Background executor: Handles sync operations (smaller pool to limit API pressure)
- Circuit breaker: Stops all API calls when rate limited
- Non-blocking retry: Uses asyncio.sleep instead of time.sleep
"""
from typing import List, Dict, Callable, TypeVar, Any
from dataclasses import dataclass
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta
import os
import time
import pandas as pd
from sqlalchemy import select, and_
from app.db.database import async_session
from app.db.models import StockCompany, StockDailyPrice, StockIndex, FundNav, FundDetailCache, FundListing
from app.core.config import settings
from app.services.sync_status import sync_status
from app.core.circuit_breaker import api_circuit_breaker, CircuitOpenError
from app.core.logging_config import (
    get_main_logger,
    get_background_logger,
    log_background_start,
    log_background_complete,
    log_background_error
)

# Initialize loggers
logger = get_main_logger()
bg_logger = get_background_logger()

# Determine worker counts based on CPU cores
_cpu_count = os.cpu_count() or 4

# Frontend executor: handles user-facing API calls
# Sized larger to handle concurrent user requests with good responsiveness
_frontend_executor = ThreadPoolExecutor(
    max_workers=max(8, _cpu_count * 2),
    thread_name_prefix="frontend"
)

# Background executor: handles sync operations
# Sized smaller to avoid overwhelming the vnstock API with concurrent requests
_background_executor = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="bg_sync"
)

# Thread-local storage for vnstock Fund API instances
_fund_api_local = threading.local()


def _ensure_pandas_applymap():
    """Provide DataFrame.applymap for pandas versions where it was removed."""
    if hasattr(pd.DataFrame, "applymap"):
        return
    def _applymap(self, func):
        return self.apply(lambda col: col.map(func))
    pd.DataFrame.applymap = _applymap  # type: ignore[attr-defined]


class RateLimitError(Exception):
    """Custom exception raised when API rate limit is hit to prevent SystemExit."""
    pass


# Rate limit detection keywords
RATE_LIMIT_KEYWORDS = [
    "Rate limit", "rate limit", "429",
    "quá nhiều request", "GIỚI HẠN API"
]

T = TypeVar('T')


def _is_rate_limit_error(error: BaseException) -> bool:
    """Check if an exception indicates a rate limit error."""
    if isinstance(error, (RateLimitError, CircuitOpenError)):
        return True
    if isinstance(error, SystemExit):
        return True
    error_msg = str(error)
    return any(keyword in error_msg for keyword in RATE_LIMIT_KEYWORDS)


def _record_rate_limit(reset_seconds: float = 30.0) -> None:
    """Record a rate limit event across circuit breaker and sync status."""
    api_circuit_breaker.record_failure(reset_timeout=reset_seconds)
    sync_status.set_rate_limited(reset_in_seconds=reset_seconds)


def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 2,
    rate_limit_keywords: List[str] = None,
) -> T:
    """
    Execute a function with retry logic. ALWAYS fails fast on rate limit.

    This is the synchronous version for use in executor threads.
    It NEVER uses time.sleep() to avoid blocking executor threads.

    Args:
        func: Callable to execute
        max_retries: Maximum retry attempts for non-rate-limit errors
        rate_limit_keywords: Keywords in error message that indicate rate limit

    Returns:
        Result of the function call

    Raises:
        CircuitOpenError: If circuit breaker is open
        RateLimitError: If rate limit is hit
        Exception: For other errors after retries exhausted
    """
    if rate_limit_keywords is None:
        rate_limit_keywords = RATE_LIMIT_KEYWORDS

    # Circuit breaker check BEFORE attempting - fail fast
    if not api_circuit_breaker.can_proceed():
        raise CircuitOpenError("Circuit breaker is open - API rate limited")

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            result = func()
            api_circuit_breaker.record_success()
            return result
        except (SystemExit, Exception) as e:
            last_exception = e
            is_rate_limit = _is_rate_limit_error(e)

            if is_rate_limit:
                # Record failure in circuit breaker
                _record_rate_limit(reset_seconds=30.0)

                # Convert SystemExit to RateLimitError to prevent process termination
                if isinstance(e, SystemExit):
                    logger.warning("Caught SystemExit (Rate Limit) - converting to RateLimitError")
                    last_exception = RateLimitError(str(e))

                # ALWAYS fail fast on rate limit - no blocking sleep
                logger.warning(f"Rate limit hit - failing fast (attempt {attempt + 1})")
                raise last_exception
            else:
                # Non-rate-limit error - may retry
                if attempt < max_retries:
                    logger.warning(f"Error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    continue
                raise

    raise last_exception


async def async_retry_with_backoff(
    func: Callable[..., T],
    executor: ThreadPoolExecutor = None,
    max_retries: int = 3,
    initial_delay: float = 30.0,
    backoff_multiplier: float = 2.0,
    rate_limit_keywords: List[str] = None
) -> T:
    """
    Async retry with non-blocking delays using asyncio.sleep.

    This should be used for background sync tasks. The func is run in
    an executor, and delays use asyncio.sleep (non-blocking).

    Args:
        func: Synchronous callable to execute
        executor: ThreadPoolExecutor to use (defaults to _background_executor)
        max_retries: Maximum retry attempts
        initial_delay: Initial delay in seconds after rate limit
        backoff_multiplier: Multiplier for delay on each retry
        rate_limit_keywords: Keywords that indicate rate limit error

    Returns:
        Result of the function call

    Raises:
        CircuitOpenError: If circuit breaker is open
        RateLimitError: If rate limit persists after retries
    """
    if rate_limit_keywords is None:
        rate_limit_keywords = RATE_LIMIT_KEYWORDS
    if executor is None:
        executor = _background_executor

    loop = asyncio.get_event_loop()
    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        # Check circuit breaker before each attempt
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError("Circuit breaker is open - API rate limited")

        try:
            result = await loop.run_in_executor(executor, func)
            api_circuit_breaker.record_success()
            return result
        except (SystemExit, RateLimitError, CircuitOpenError, Exception) as e:
            last_exception = e
            is_rate_limit = _is_rate_limit_error(e)

            if is_rate_limit:
                _record_rate_limit(reset_seconds=delay)

                # Convert SystemExit
                if isinstance(e, SystemExit):
                    last_exception = RateLimitError(str(e))

                if attempt < max_retries:
                    bg_logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Waiting {delay:.1f}s (non-blocking)..."
                    )
                    # Non-blocking sleep - event loop can handle other tasks
                    await asyncio.sleep(delay)
                    delay *= backoff_multiplier
                else:
                    raise last_exception
            else:
                # Non-rate-limit error
                raise

    raise last_exception


@dataclass
class IndexValue:
    """Index value data class."""
    symbol: str
    name: str
    value: float  # Current/latest close price
    change: float  # Price change from open (percentage)
    change_value: float  # Absolute change from open


@dataclass
class StockInfo:
    """Stock information data class."""
    ticker: str
    price: float
    market_cap: float  # In billion VND (tỷ đồng)
    company_name: str = ""
    exchange: str = ""
    charter_capital: float = 0.0  # In billion VND
    pe_ratio: float | None = None
    accumulated_value: float | None = None  # In billion VND
    price_change_24h: float | None = None  # Percentage
    price_change_1w: float | None = None  # Percentage
    price_change_1m: float | None = None  # Percentage
    price_change_1y: float | None = None  # Percentage


class VnstockService:
    """Service class for vnstock operations."""
    
    # Valid groups supported by symbols_by_group
    VALID_GROUPS = {
        'HOSE', 'VN30', 'VNMidCap', 'VNSmallCap', 'VNAllShare', 'VN100', 
        'ETF', 'HNX', 'HNX30', 'HNXCon', 'HNXFin', 'HNXLCap', 'HNXMSCap', 
        'HNXMan', 'UPCOM', 'FU_INDEX', 'FU_BOND', 'BOND', 'CW'
    }

    def __init__(self):
        # Initialize vnstock API key if provided
        if settings.vnstock_api_key:
            try:
                import vnstock
                vnstock.change_api_key(settings.vnstock_api_key)
                logger.info("vnstock API key configured")
            except Exception as e:
                logger.error(f"Error configuring vnstock API key: {e}")

        self._enriching_tickers = set()
        # Fund holding caches (no DB backing): {key: (data, timestamp)}
        self._fund_listing_cache = {}
        self._fund_top_holding_cache = {}
        self._fund_industry_holding_cache = {}
        self._fund_asset_holding_cache = {}
        self._fund_listing_df_cache = None
        self._fund_listing_df_timestamp = 0
        self._fund_listing_refresh_lock = threading.Lock()
        self._sync_engine = None
        # Fund benchmark cache to avoid API hits on every request
        self._fund_benchmark_cache = {}
        
        # Cache TTLs in seconds (for holdings without DB backing)
        self._FUND_LISTING_TTL = 3600  # 1 hour
        self._FUND_DETAILS_TTL = 1800  # 30 minutes
        self._FUND_LISTING_DB_TTL = 7 * 24 * 3600  # 7 days
        self._FUND_BENCHMARK_TTL = 6 * 3600  # 6 hours

    def _fund_listing_is_fresh(self, last_updated: datetime | None) -> bool:
        if not last_updated:
            return False
        return (datetime.utcnow() - last_updated).total_seconds() < self._FUND_LISTING_DB_TTL

    def _get_fund_listing_records_from_db_sync(
        self,
        fund_type: str | None = None
    ) -> tuple[list[dict], datetime | None]:
        """Load fund listing records from DB."""
        from sqlalchemy.orm import Session

        engine = self._get_sync_engine()
        with Session(engine) as session:
            stmt = select(FundListing)
            if fund_type:
                stmt = stmt.where(FundListing.fund_type == fund_type)
            rows = list(session.execute(stmt).scalars().all())

            if not rows:
                return [], None

            last_updated = max((r.updated_at for r in rows if r.updated_at), default=None)
            records: list[dict] = []
            for row in rows:
                record = dict(row.data) if isinstance(row.data, dict) else {}
                if 'symbol' not in record and row.symbol:
                    record['symbol'] = row.symbol
                if 'name' not in record and row.name:
                    record['name'] = row.name
                if 'fund_type' not in record and row.fund_type:
                    record['fund_type'] = row.fund_type
                if 'fund_owner' not in record and row.fund_owner:
                    record['fund_owner'] = row.fund_owner
                records.append(record)

            return records, last_updated

    def _get_fund_listing_df_from_db_sync(self) -> tuple[pd.DataFrame | None, datetime | None]:
        """Load fund listing from DB as DataFrame."""
        records, last_updated = self._get_fund_listing_records_from_db_sync()
        if not records:
            return None, last_updated
        return pd.DataFrame(records), last_updated

    def _upsert_fund_listing_db_sync(self, records: list[dict]) -> None:
        """Upsert fund listing records to DB."""
        from sqlalchemy.orm import Session
        from sqlalchemy.dialects.postgresql import insert

        if not records:
            return

        engine = self._get_sync_engine()
        with Session(engine) as session:
            for record in records:
                symbol = record.get('symbol') or record.get('fund_code') or record.get('short_name')
                if not symbol:
                    continue

                name = record.get('name') or record.get('fund_name') or symbol
                fund_type = record.get('fund_type') or record.get('type')
                fund_owner = record.get('fund_owner') or record.get('owner') or record.get('management_company')
                now = datetime.utcnow()

                stmt = insert(FundListing).values(
                    symbol=symbol,
                    name=name,
                    fund_type=fund_type,
                    fund_owner=fund_owner,
                    data=record,
                    updated_at=now
                ).on_conflict_do_update(
                    index_elements=[FundListing.symbol],
                    set_={
                        "name": name,
                        "fund_type": fund_type,
                        "fund_owner": fund_owner,
                        "data": record,
                        "updated_at": now
                    }
                )
                session.execute(stmt)
            session.commit()

    def _get_fund_detail_cache_sync(self, symbol: str, detail_type: str) -> tuple[list, bool]:
        """Get cached fund detail data and freshness flag."""
        from sqlalchemy.orm import Session

        engine = self._get_sync_engine()
        with Session(engine) as session:
            stmt = select(FundDetailCache).where(
                and_(
                    FundDetailCache.symbol == symbol,
                    FundDetailCache.detail_type == detail_type
                )
            )
            record = session.execute(stmt).scalar_one_or_none()

            if not record:
                return [], False

            age_seconds = (datetime.utcnow() - record.updated_at).total_seconds() if record.updated_at else None
            is_fresh = age_seconds is not None and age_seconds < self._FUND_DETAILS_TTL
            data = record.data if isinstance(record.data, list) else []
            return data, is_fresh

    def _upsert_fund_detail_cache_sync(self, symbol: str, detail_type: str, data: list) -> None:
        """Upsert cached fund detail data."""
        from sqlalchemy.orm import Session

        engine = self._get_sync_engine()
        with Session(engine) as session:
            stmt = select(FundDetailCache).where(
                and_(
                    FundDetailCache.symbol == symbol,
                    FundDetailCache.detail_type == detail_type
                )
            )
            record = session.execute(stmt).scalar_one_or_none()

            if record:
                record.data = data
                record.updated_at = datetime.utcnow()
            else:
                session.add(FundDetailCache(
                    symbol=symbol,
                    detail_type=detail_type,
                    data=data,
                    updated_at=datetime.utcnow()
                ))
            session.commit()

    def _is_cache_valid(self, cache: Dict, key: str, ttl: int) -> bool:
        """Check if cache entry exists and is not expired."""
        if key in cache:
            data, timestamp = cache[key]
            if time.time() - timestamp < ttl:
                return True
        return False

    def _get_cached_fund_benchmarks(self) -> Dict[str, Any] | None:
        """Return cached fund benchmarks if fresh."""
        cache_key = "fund_benchmarks"
        if self._is_cache_valid(self._fund_benchmark_cache, cache_key, self._FUND_BENCHMARK_TTL):
            return self._fund_benchmark_cache[cache_key][0]
        return None

    def _set_cached_fund_benchmarks(self, benchmarks: Dict[str, Any]) -> None:
        """Store fund benchmarks in memory cache."""
        if not benchmarks:
            return
        self._fund_benchmark_cache["fund_benchmarks"] = (benchmarks, time.time())
    
    def _get_sync_engine(self):
        """Get or create cached synchronous SQLAlchemy engine."""
        if self._sync_engine is None:
            from sqlalchemy import create_engine
            sync_url = settings.database_url.replace('+asyncpg', '')
            self._sync_engine = create_engine(sync_url)
        return self._sync_engine

    def _get_thread_local_fund_api(self):
        """Get or create a thread-local vnstock Fund API instance."""
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError("Circuit breaker open - cannot initialize fund API")

        fund_api = getattr(_fund_api_local, "fund_api", None)
        if fund_api is not None:
            return fund_api

        try:
            from vnstock import Fund
            fund_api = Fund()
            _fund_api_local.fund_api = fund_api
            return fund_api
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=60.0)
                raise CircuitOpenError("Rate limited while initializing fund API")
            raise

    async def sync_indices(self) -> None:
        """
        Fetch all indices from vnstock and update the database.
        """
        loop = asyncio.get_event_loop()
        indices_df = await loop.run_in_executor(_frontend_executor, self._fetch_all_indices_from_lib)
        
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
                    if symbol not in self.VALID_GROUPS:
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
                        group_code = self._get_group_code_for_index(symbol)
                        if group_code not in self.VALID_GROUPS:
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

    # Index name mapping for display
    INDEX_NAMES = {
        'VNINDEX': 'VN-Index',
        'HNXINDEX': 'HNX-Index',
        'UPCOMINDEX': 'UPCOM-Index',
        'VN30': 'VN30',
        'HNX30': 'HNX30',
    }

    async def get_index_values(self, symbols: List[str] = None) -> List[IndexValue]:
        """
        Fetch latest values for major market indices.
        
        Args:
            symbols: List of index symbols. Defaults to main indices if not specified.
            
        Returns:
            List of IndexValue objects with current price and change data.
        """
        if symbols is None:
            symbols = ['VNINDEX', 'HNXINDEX', 'UPCOMINDEX', 'VN30', 'HNX30']
        
        loop = asyncio.get_event_loop()
        results = []
        
        for symbol in symbols:
            try:
                index_data = await loop.run_in_executor(_frontend_executor, self._fetch_index_value_sync, symbol)
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
                    name=self.INDEX_NAMES.get(symbol, symbol),
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


    async def get_index_stocks(self, index_symbol: str, limit: int = 100) -> List[StockInfo]:
        """
        Fetch stocks for a specific index with price and market cap data.
        
        Args:
            index_symbol: The symbol of the index (e.g., "VN100", "VN30", "VNDIAMOND")
            limit: Maximum number of stocks to return
            
        Returns:
            List of StockInfo objects
        """
        loop = asyncio.get_event_loop()
        stocks = await loop.run_in_executor(_frontend_executor, self._fetch_index_data, index_symbol, limit)
        
        # Launch background task for enrichment
        asyncio.create_task(self._enrich_stocks_with_metadata(stocks))
        
        # Apply current cache to the response immediately
        return await self._apply_cache_to_stocks(stocks)

    async def get_industry_list(self) -> List[Dict[str, str]]:
        """
        Fetch all ICB level 2 industries.
        """
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(_frontend_executor, self._fetch_industries_sync)
        if df is not None and not df.empty:
            # Filter for level 2 industries
            l2_df = df[df['level'] == 2]
            return l2_df[['icb_name', 'en_icb_name', 'icb_code']].to_dict('records')
        return []

    def _fetch_industries_sync(self) -> pd.DataFrame:
        """Fetch industries synchronously."""
        from vnstock import Listing

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError("Circuit breaker open - cannot fetch industries")

        try:
            result = Listing(source='VCI').industries_icb()
            api_circuit_breaker.record_success()
            return result
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching industries: {e}")
            raise

    async def get_industry_stocks(self, industry_name: str, limit: int = 100) -> List[StockInfo]:
        """
        Fetch stocks for a specific ICB industry.
        """
        loop = asyncio.get_event_loop()
        stocks = await loop.run_in_executor(_frontend_executor, self._fetch_industry_data, industry_name, limit)
        
        # Launch background task for enrichment
        asyncio.create_task(self._enrich_stocks_with_metadata(stocks))
        
        # Apply current cache to the response immediately
        return await self._apply_cache_to_stocks(stocks)

    async def get_vn100_stocks(self) -> List[StockInfo]:
        """
        Fetch VN-100 stocks (top 100 by market cap) with price and market cap data.
        
        Returns:
            List of StockInfo objects include company_name
        """
        return await self.get_index_stocks("VN100", settings.vn100_limit)

    async def get_vn30_stocks(self) -> List[StockInfo]:
        """
        Fetch VN-30 stocks (top 30 by market cap and liquidity) with price and market cap data.
        
        Returns:
            List of StockInfo objects include company_name
        """
        return await self.get_index_stocks("VN30", settings.vn30_limit)

    async def _apply_cache_to_stocks(self, stocks: List[StockInfo]) -> List[StockInfo]:
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

    async def _enrich_stocks_with_metadata(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """Add company names and financial metadata to stock info objects, using DB cache."""
        if not stocks:
            return []

        tickers = [s.ticker for s in stocks 
                   if s.ticker not in self._enriching_tickers]
        if not tickers:
            return stocks

        now = datetime.utcnow()
        stale_threshold = now - timedelta(days=7) 
        error_stale_threshold = now - timedelta(hours=1) # Retry missing PE after 1 hour

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
                    all_symbols_df = await loop.run_in_executor(_background_executor, self._fetch_all_symbols)
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
                            log_background_start("Metadata Enrichment", f"{len(tickers_to_fetch)}/{len(tickers_needing_finance)} stocks")
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
                                    data = await loop.run_in_executor(_background_executor, self._fetch_stock_finance_sync, symbol)
                                    
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
    
    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten multi-level column names."""
        if df.columns.nlevels > 1:
            new_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    # Join non-NaN parts with underscore
                    new_cols.append('_'.join([str(c) for c in col if pd.notna(c)]))
                else:
                    new_cols.append(str(col))
            df.columns = new_cols
        return df
    
    def _get_group_code_for_index(self, index_symbol: str) -> str:
        """
        Map index symbol from all_indices() to group code expected by symbols_by_group().
        """
        mapping = {
            'VN30': 'VN30',
            'VN100': 'VN100',
            'VNMID': 'VNMidCap',
            'VNSML': 'VNSmallCap',
            'VNALL': 'VNAllShare',
            'HNX30': 'HNX30',
            # Add more mappings as needed based on valid groups:
            # ['HOSE', 'VN30', 'VNMidCap', 'VNSmallCap', 'VNAllShare', 'VN100', 
            #  'ETF', 'HNX', 'HNX30', 'HNXCon', 'HNXFin', 'HNXLCap', 'HNXMSCap', 
            #  'HNXMan', 'UPCOM', 'FU_INDEX', 'FU_BOND', 'BOND', 'CW']
        }
        return mapping.get(index_symbol, index_symbol)

    def _fetch_index_data(self, index_name: str, limit: int) -> List[StockInfo]:
        """
        Synchronous method to fetch index data (VN100, VN30, etc.) using vnstock.
        Called in thread pool executor to avoid blocking.
        """
        from vnstock import Listing

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch index {index_name}")

        try:
            # Map index name to group code
            group_code = self._get_group_code_for_index(index_name)

            # Get stock symbols for the specified group
            listing = Listing(source='VCI')
            try:
                symbols_df = listing.symbols_by_group(group_code)
                api_circuit_breaker.record_success()
            except ValueError as e:
                logger.warning(f"Group '{group_code}' (mapped from '{index_name}') not supported by symbols_by_group: {e}")
                return []
            except (SystemExit, Exception) as e:
                if _is_rate_limit_error(e):
                    _record_rate_limit(reset_seconds=30.0)
                    raise CircuitOpenError(f"Rate limited fetching symbols for {group_code}: {e}")
                logger.warning(f"Error fetching symbols for group '{group_code}': {e}")
                return []

            if symbols_df is None or symbols_df.empty:
                logger.warning(f"Could not fetch symbols for {group_code} group")
                return []
            else:
                # The series returned by symbols_by_group contains the symbols
                symbols = symbols_df.tolist()

            return self._fetch_symbols_data(symbols, limit)

        except CircuitOpenError:
            raise  # Re-raise circuit breaker errors
        except Exception as e:
            logger.warning(f"Error fetching {index_name} data: {e}")
            return []

    def _fetch_industry_data(self, industry_name: str, limit: int) -> List[StockInfo]:
        """
        Synchronous method to fetch industry data using vnstock.
        """
        from vnstock import Listing

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch industry {industry_name}")

        try:
            listing = Listing(source='VCI')
            # Get all symbols with industry info
            df = listing.symbols_by_industries()
            api_circuit_breaker.record_success()

            if df is not None and not df.empty:
                # Filter by icb_name2 (Level 2) or icb_name3/4 if needed
                # We'll match against any of them for flexibility
                cols_to_check = ['icb_name2', 'icb_name3', 'icb_name4']
                mask = pd.Series([False] * len(df))
                for col in cols_to_check:
                    if col in df.columns:
                        mask |= (df[col] == industry_name)

                filtered_df = df[mask]
                symbols = filtered_df['symbol'].tolist()

                return self._fetch_symbols_data(symbols, limit)
            return []
        except CircuitOpenError:
            raise  # Re-raise circuit breaker errors
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching industry {industry_name}: {e}")
            logger.warning(f"Error fetching industry {industry_name} data: {e}")
            return []

    def _fetch_symbols_data(self, symbols: List[str], limit: int) -> List[StockInfo]:
        """
        Generic method to fetch price and market cap data for a list of symbols.
        """
        from vnstock import Trading
        from app.core.circuit_breaker import api_circuit_breaker, CircuitOpenError

        # Check circuit breaker before making API calls
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError("Circuit breaker open - API rate limited")

        try:
            # Get price board for stocks in batches
            trading = Trading(source='VCI')
            stocks_data = []
            batch_size = 50 # Smaller batch for more reliable fetching
            
            # Process symbols
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                try:
                    price_board = trading.price_board(batch)
                    api_circuit_breaker.record_success()

                    if price_board is not None and not price_board.empty:
                        # Flatten multi-level column names
                        price_board = self._flatten_columns(price_board)
                        
                        for _, row in price_board.iterrows():
                            ticker = row.get('listing_symbol', '')
                            if not ticker:
                                continue
                            
                            # Get price (in VND)
                            price = 0
                            if 'match_match_price' in row.index:
                                try:
                                    price = float(row['match_match_price'])
                                except (ValueError, TypeError):
                                    pass
                            
                            # Calculate market cap: price * listed_share
                            # Returns value in VND
                            market_cap = 0
                            listed_shares = 0
                            if 'listing_listed_share' in row.index:
                                try:
                                    listed_shares = float(row['listing_listed_share'])
                                    # Market cap in billion VND = (price * shares) / 1e9
                                    market_cap = (price * listed_shares) / 1e9
                                except (ValueError, TypeError):
                                    pass
                            
                            # Get charter capital (in billion VND)
                            # Use listing_charter_capital if available, else estimate from listed shares
                            charter_capital = 0.0
                            if 'listing_charter_capital' in row.index:
                                try:
                                    charter_capital = float(row['listing_charter_capital']) / 1e9
                                except (ValueError, TypeError):
                                    pass
                            
                            # Get P/E ratio
                            pe_ratio = None
                            if 'financial_pe' in row.index:
                                try:
                                    pe_val = row['financial_pe']
                                    if pd.notna(pe_val) and pe_val != 0:
                                        pe_ratio = float(pe_val)
                                except (ValueError, TypeError):
                                    pass

                            # Get accumulated trading value (in billion VND)
                            # API returns value in Million VND, divide by 1000 to get Billion VND
                            accumulated_value = None
                            if 'match_accumulated_value' in row.index:
                                try:
                                    acc_val = row['match_accumulated_value']
                                    if pd.notna(acc_val):
                                        accumulated_value = float(acc_val) / 1e3
                                except (ValueError, TypeError):
                                    pass

                            # Get 24h price change percentage
                            price_change_24h = None
                            if 'match_price_change_ratio' in row.index:
                                try:
                                    change_val = row['match_price_change_ratio']
                                    if pd.notna(change_val):
                                        # Convert to percentage (value is already ratio)
                                        price_change_24h = float(change_val) * 100
                                except (ValueError, TypeError):
                                    pass
                            
                            # Fallback: calculate from reference price if match_price_change_ratio is missing or zero
                            if (price_change_24h is None or price_change_24h == 0) and 'listing_ref_price' in row.index:
                                try:
                                    ref_price = float(row['listing_ref_price'])
                                    if ref_price > 0 and price > 0:
                                        price_change_24h = ((price - ref_price) / ref_price) * 100
                                except (ValueError, TypeError):
                                    pass
                            
                            # Fallback for charter capital: shares * 10,000 (par value) / 1e9
                            if charter_capital == 0 and listed_shares > 0:
                                charter_capital = (listed_shares * 10000) / 1e9

                            if ticker and price > 0:
                                # Map exchange codes to full names if needed
                                exchange = row.get('listing_exchange', '')
                                if exchange == 'HSX':
                                    exchange = 'HOSE'
                                
                                # Get company name if available
                                company_name = row.get('listing_organ_name', '')
                                
                                stocks_data.append(StockInfo(
                                    ticker=str(ticker),
                                    price=price,
                                    company_name=company_name,
                                    exchange=exchange,
                                    market_cap=round(market_cap, 2),
                                    charter_capital=round(charter_capital, 2),
                                    pe_ratio=round(pe_ratio, 2) if pe_ratio is not None else None,
                                    accumulated_value=round(accumulated_value, 2) if accumulated_value is not None else None,
                                    price_change_24h=round(price_change_24h, 2) if price_change_24h is not None else None
                                ))
                                
                except (SystemExit, Exception) as e:
                    if _is_rate_limit_error(e):
                        _record_rate_limit(reset_seconds=60.0)
                        raise CircuitOpenError(f"Rate limited while fetching price board batch {i}")
                    logger.warning(f"Error fetching batch {i}: {e}")
                    continue
            
            # Sort by market cap descending and take the requested limit
            stocks_data.sort(key=lambda x: x.market_cap, reverse=True)
            top_stocks = stocks_data[:limit]
            
            # Fetch historical price changes for top stocks
            top_stocks = self._enrich_with_price_changes(top_stocks)
            
            return top_stocks
            
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            # Check if this is a rate limit error and record circuit breaker failure
            error_name = type(e).__name__
            if error_name in {"RateLimitExceeded", "RateLimitError"} or "rate limit" in str(e).lower():
                _record_rate_limit(reset_seconds=60.0)
                raise CircuitOpenError(f"Rate limited while fetching symbols data: {e}")
            logger.warning(f"Error fetching symbols data: {e}")
            import traceback
            bg_logger.error(f"Stack trace for symbols data error:\n{traceback.format_exc()}")
            return []

    def _enrich_with_price_changes(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """
        Enrich stock data with historical price changes (1w, 1m, 1y).
        """
        return self._enrich_with_price_changes_sync(stocks)
    
    def _enrich_with_price_changes_sync(
        self,
        stocks: List[StockInfo],
        fetch_missing_history: bool = False
    ) -> List[StockInfo]:
        """
        Synchronous fallback for price change enrichment.
        Queries DB cache directly, fetches missing from API.
        """
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        
        # Calculate target dates
        today = datetime.now().date()
        target_dates = {
            '1w': today - timedelta(days=7),
            '1m': today - timedelta(days=30),
            '1y': today - timedelta(days=365),
        }
        
        # Use sync connection for DB lookup  
        engine = self._get_sync_engine()
        
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
        result = {}
        
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
            symbol_prices = {}
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
        from sqlalchemy.orm import Session
        
        # Use provided session or create a temporary one
        own_session = False
        if session is None:
            engine = self._get_sync_engine()
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
        from app.core.circuit_breaker import api_circuit_breaker

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
    


    async def get_income_statement(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        """
        Fetch income statement data for a given stock symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'VIC', 'VNM')
            period: 'quarter' or 'year'
            lang: 'en' or 'vi'
            
        Returns:
            List of income statement records with period data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_frontend_executor, self._fetch_income_statement_sync, symbol, period, lang)
    
    def _fetch_income_statement_sync(self, symbol: str, period: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch income statement synchronously."""
        from vnstock import Vnstock

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch income statement for {symbol}")

        try:
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            df = s.finance.income_statement(period=period, lang=lang)
            api_circuit_breaker.record_success()

            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                # Convert to list of dicts, handling NaN values
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching income statement for {symbol}: {e}")
            logger.warning(f"Error fetching income statement for {symbol}: {e}")
            return []
    
    async def get_balance_sheet(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        """
        Fetch balance sheet data for a given stock symbol.
        
        Args:
            symbol: Stock ticker symbol
            period: 'quarter' or 'year'
            lang: 'en' or 'vi'
            
        Returns:
            List of balance sheet records with period data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_frontend_executor, self._fetch_balance_sheet_sync, symbol, period, lang)
    
    def _fetch_balance_sheet_sync(self, symbol: str, period: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch balance sheet synchronously."""
        from vnstock import Vnstock

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch balance sheet for {symbol}")

        try:
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            df = s.finance.balance_sheet(period=period, lang=lang)
            api_circuit_breaker.record_success()

            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching balance sheet for {symbol}: {e}")
            logger.warning(f"Error fetching balance sheet for {symbol}: {e}")
            return []
    
    async def get_cash_flow(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        """
        Fetch cash flow statement data for a given stock symbol.
        
        Args:
            symbol: Stock ticker symbol
            period: 'quarter' or 'year'
            lang: 'en' or 'vi'
            
        Returns:
            List of cash flow records with period data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_frontend_executor, self._fetch_cash_flow_sync, symbol, period, lang)
    
    def _fetch_cash_flow_sync(self, symbol: str, period: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch cash flow synchronously."""
        from vnstock import Vnstock

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch cash flow for {symbol}")

        try:
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            df = s.finance.cash_flow(period=period, lang=lang)
            api_circuit_breaker.record_success()

            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching cash flow for {symbol}: {e}")
            logger.warning(f"Error fetching cash flow for {symbol}: {e}")
            return []
    
    async def get_financial_ratios(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        """
        Fetch financial ratios for a given stock symbol.
        Includes P/E, P/B, P/S, ROE, ROA, etc.
        
        Args:
            symbol: Stock ticker symbol
            period: 'quarter' or 'year'
            lang: 'en' or 'vi'
            
        Returns:
            List of financial ratio records with period data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_frontend_executor, self._fetch_financial_ratios_sync, symbol, period, lang)
    
    def _fetch_financial_ratios_sync(self, symbol: str, period: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch financial ratios synchronously."""
        from vnstock import Vnstock

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch financial ratios for {symbol}")

        try:
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            df = s.finance.ratio(period=period, lang=lang)
            api_circuit_breaker.record_success()

            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching financial ratios for {symbol}: {e}")
            logger.warning(f"Error fetching financial ratios for {symbol}: {e}")
            return []



    async def get_company_overview(self, symbol: str, source: str = "auto") -> List[Dict[str, Any]]:
        """Fetch company overview for a given stock symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_frontend_executor, self._fetch_company_overview_sync, symbol, source)

    def _fetch_company_overview_sync(self, symbol: str, source: str = "auto") -> List[Dict[str, Any]]:
        """Fetch company overview synchronously."""
        from vnstock import Company

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch company overview for {symbol}")

        source_normalized = (source or "auto").strip().lower()
        if source_normalized not in {"auto", "vci", "kbs"}:
            raise ValueError(f"Unsupported company overview source: {source}")

        def _normalize_records(df: Any) -> List[Dict[str, Any]]:
            if df is None:
                return []
            if isinstance(df, dict):
                df = pd.DataFrame([df])
            elif isinstance(df, list):
                df = pd.DataFrame(df)
            if not isinstance(df, pd.DataFrame) or df.empty:
                return []
            df = self._flatten_columns(df)
            records = df.to_dict('records')
            for record in records:
                for key, value in record.items():
                    if value is None:
                        continue
                    # Avoid ambiguous truth values for list/dict-like objects
                    if pd.api.types.is_scalar(value) and pd.isna(value):
                        record[key] = None
            return records

        def _fetch_from_source(source: str) -> List[Dict[str, Any]]:
            _ensure_pandas_applymap()
            c = Company(symbol=symbol[:3], source=source)
            df = c.overview()
            api_circuit_breaker.record_success()
            return _normalize_records(df)

        def _handle_error(source_label: str, err: BaseException) -> None:
            if _is_rate_limit_error(err):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching company overview for {symbol}: {err}")
            # Unwrap tenacity RetryError to reveal the underlying exception
            try:
                from tenacity import RetryError
                if isinstance(err, RetryError) and err.last_attempt:
                    root = err.last_attempt.exception()
                    logger.warning(
                        f"Error fetching company overview for {symbol} from {source_label}: {err}; root={type(root).__name__}: {root}"
                    )
                    return
            except Exception:
                pass
            logger.warning(f"Error fetching company overview for {symbol} from {source_label}: {err}")

        if source_normalized == "vci":
            try:
                return _fetch_from_source('VCI')
            except CircuitOpenError:
                raise
            except (SystemExit, Exception) as e:
                _handle_error("VCI", e)
                return []

        if source_normalized == "kbs":
            try:
                return _fetch_from_source('KBS')
            except CircuitOpenError:
                raise
            except (SystemExit, Exception) as e:
                _handle_error("KBS", e)
                return []

        # auto: try VCI first, fallback to KBS
        try:
            records = _fetch_from_source('VCI')
            if records:
                return records
            logger.warning(f"Empty company overview for {symbol} from VCI, trying KBS.")
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            _handle_error("VCI", e)

        try:
            return _fetch_from_source('KBS')
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            _handle_error("KBS", e)
            return []

    async def get_shareholders(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch shareholders for a given stock symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_frontend_executor, self._fetch_shareholders_sync, symbol)

    def _fetch_shareholders_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch shareholders synchronously."""
        from vnstock import Company

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch shareholders for {symbol}")

        try:
            c = Company(symbol=symbol[:3], source='VCI')
            df = c.shareholders()
            api_circuit_breaker.record_success()

            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching shareholders for {symbol}: {e}")
            logger.warning(f"Error fetching shareholders for {symbol}: {e}")
            return []

    async def get_officers(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch officers for a given stock symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_frontend_executor, self._fetch_officers_sync, symbol)

    def _fetch_officers_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch officers synchronously."""
        from vnstock import Company

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch officers for {symbol}")

        try:
            c = Company(symbol=symbol[:3], source='VCI')
            df = c.officers()
            api_circuit_breaker.record_success()

            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching officers for {symbol}: {e}")
            logger.warning(f"Error fetching officers for {symbol}: {e}")
            return []

    async def get_subsidiaries(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch subsidiaries for a given stock symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_frontend_executor, self._fetch_subsidiaries_sync, symbol)

    def _fetch_subsidiaries_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch subsidiaries synchronously."""
        from vnstock import Company

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch subsidiaries for {symbol}")

        try:
            c = Company(symbol=symbol[:3], source='VCI')
            df = c.subsidiaries()
            api_circuit_breaker.record_success()

            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching subsidiaries for {symbol}: {e}")
            logger.warning(f"Error fetching subsidiaries for {symbol}: {e}")
            return []

    async def get_volume_history(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Fetch volume history for a given stock symbol.

        Args:
            symbol: Stock ticker symbol
            days: Number of days to fetch (default: 30)

        Returns:
            Dict with symbol, company_name, and data (list of VolumeDataPoint dicts)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_frontend_executor, self._fetch_volume_history_sync, symbol, days)

    def _fetch_volume_history_sync(self, symbol: str, days: int) -> Dict[str, Any]:
        """Fetch volume history synchronously."""
        from vnstock import Vnstock
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        symbol_clean = symbol[:3]
        company_name = symbol_clean

        # Use sync connection for DB lookup
        engine = self._get_sync_engine()

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

    # Track background sync task for weekly prices
    _weekly_prices_sync_task = None
    _weekly_prices_syncing_symbols = set()

    async def get_stocks_weekly_prices(
        self,
        symbols: List[str],
        start_year: int,
        include_benchmarks: bool = True
    ) -> Dict[str, Any]:
        """
        Get weekly price data for multiple stocks.
        Returns cached data immediately and triggers background sync if stale.

        Args:
            symbols: List of stock symbols
            start_year: Starting year for the data
            include_benchmarks: Whether to include VNINDEX and VN30 benchmarks

        Returns:
            Dict with stocks data, benchmarks, date range, and sync status
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
            symbol_data = {}
            for record in records:
                if record.symbol not in symbol_data:
                    symbol_data[record.symbol] = []
                symbol_data[record.symbol].append({
                    'date': record.date,
                    'close': record.close
                })

            # Aggregate to weekly using pandas
            weekly_data = {}
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
                # (Resampling can sometimes include the previous week's end)
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
                            _background_executor,
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
            engine = self._get_sync_engine()
            engine.dispose()

    async def get_fund_listing(self, fund_type: str = "") -> List[Dict[str, Any]]:
        """
        Fetch all available funds.
        """
        cache_key = fund_type or "all"
        if self._is_cache_valid(self._fund_listing_cache, cache_key, self._FUND_LISTING_TTL):
            return self._fund_listing_cache[cache_key][0]

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(_frontend_executor, self._fetch_fund_listing_sync, fund_type)

        # Update cache
        if data:
            self._fund_listing_cache[cache_key] = (data, time.time())
        return data

    def _fetch_fund_listing_sync(self, fund_type: str, fail_fast: bool = True) -> List[Dict[str, Any]]:
        """Fetch fund listing synchronously."""
        records, last_updated = self._get_fund_listing_records_from_db_sync(fund_type or None)

        if records and self._fund_listing_is_fresh(last_updated):
            return records

        # Fallback to in-memory DF cache if DB cache missing
        df = None
        if self._fund_listing_df_cache is not None and (time.time() - self._fund_listing_df_timestamp < self._FUND_LISTING_TTL):
            df = self._fund_listing_df_cache.copy()
            bg_logger.debug(f"Using in-memory fund listing cache for endpoint (type={fund_type or 'all'})")

        if df is None:
            # Prevent refresh stampede: only one thread refreshes at a time.
            acquired = self._fund_listing_refresh_lock.acquire(blocking=False)
            if not acquired:
                return records or []

            def fetch_listing():
                fund = self._get_thread_local_fund_api()
                return fund.listing()

            try:
                df = retry_with_backoff(fetch_listing, max_retries=3)
                if df is not None and not df.empty:
                    self._fund_listing_df_cache = df
                    self._fund_listing_df_timestamp = time.time()
                    bg_logger.debug("Fetched and cached fresh fund listing from API for endpoint")
                    df = df.copy()
            except Exception as e:
                logger.warning(f"Error fetching fund listing from API: {e}")
                if records:
                    return records
                return []
            finally:
                if acquired:
                    self._fund_listing_refresh_lock.release()

        if df is not None and not df.empty:
            df = self._flatten_columns(df)
            if fund_type:
                df = df[df['fund_type'] == fund_type]

            records = df.to_dict('records')
            for record in records:
                # Normalize field names for frontend
                self._normalize_fund_field(record, 'symbol', ['short_name', 'fund_code'])
                self._normalize_fund_field(record, 'name', ['name', 'short_name'])
                self._normalize_fund_field(record, 'fund_owner', ['fund_owner_name', 'management_company'])

                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None

            # Persist refreshed listing
            if records and not fund_type:
                self._upsert_fund_listing_db_sync(records)

            return records

        return records or []

    def _normalize_fund_field(self, record: dict, target_key: str, source_keys: List[str]) -> None:
        """
        Map vnstock field name to expected frontend field name.

        Args:
            record: Dictionary record to modify
            target_key: Desired field name for frontend
            source_keys: List of possible source field names from vnstock (in priority order)
        """
        if target_key in record:
            return
        for key in source_keys:
            if key in record:
                record[target_key] = record[key]
                break

    async def get_fund_nav_report(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch NAV (Net Asset Value) history for a specific fund.
        Uses database as primary source, syncs from API only for missing/newer data.
        """
        # Load directly from database (with API sync if data is stale)
        return await self._get_fund_nav_with_sync(symbol)

    async def _get_fund_nav_with_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get NAV data from database, syncing missing data from API.
        This is the core DB-backed NAV storage method used by multiple features.
        """
        async with async_session() as session:
            # Step 1: Load existing NAV records from database
            stmt = select(FundNav).where(FundNav.symbol == symbol).order_by(FundNav.date)
            result = await session.execute(stmt)
            db_records = list(result.scalars().all())
            
            # Step 2: Determine if we need to sync from API
            need_api_sync = False
            latest_db_date = None
            
            if not db_records:
                # No data in DB, need full sync
                need_api_sync = True
            else:
                latest_db_date = db_records[-1].date
                today = date.today()
                # Sync if latest DB date is more than 3 days old
                # Using 3 days to handle weekends (fund NAV only updates on trading days)
                if (today - latest_db_date).days > 3:
                    need_api_sync = True
            
            # Step 3: Sync from API if needed
            if need_api_sync:
                loop = asyncio.get_event_loop()
                api_records = await loop.run_in_executor(
                    _background_executor, 
                    self._fetch_fund_nav_from_api_sync, 
                    symbol
                )
                
                if api_records:
                    # Filter to only new records (dates not in DB)
                    existing_dates = {r.date for r in db_records}
                    new_records = [
                        r for r in api_records 
                        if r['date'] not in existing_dates
                    ]
                    
                    # Insert new records into database
                    if new_records:
                        for record in new_records:
                            fund_nav = FundNav(
                                symbol=symbol,
                                date=record['date'],
                                nav=record['nav']
                            )
                            session.add(fund_nav)
                        await session.commit()
                        bg_logger.debug(f"Stored {len(new_records)} new NAV records for fund {symbol}")
                        
                        # Reload from DB to get complete sorted list
                        stmt = select(FundNav).where(FundNav.symbol == symbol).order_by(FundNav.date)
                        result = await session.execute(stmt)
                        db_records = list(result.scalars().all())
            
            # Step 4: Convert to dataframe and sample to weekly frequency
            df = pd.DataFrame([
                {'date': record.date, 'nav': record.nav}
                for record in db_records
            ])
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                # resample to weekly, taking the last available point in each week
                # Consistent with performance API (using Sunday labels)
                df_weekly = df.resample('W', on='date').last().dropna().reset_index()
                
                return [
                    {
                        'date': row['date'].strftime('%Y-%m-%d'),
                        'nav': row['nav']
                    }
                    for _, row in df_weekly.iterrows()
                ]
            
            return []

    def _fetch_fund_nav_from_api_sync(self, symbol: str, fail_fast: bool = True) -> List[Dict[str, Any]]:
        """Fetch fund NAV data from vnstock API synchronously."""
        def fetch_nav():
            fund = self._get_thread_local_fund_api()
            df = fund.details.nav_report(symbol=symbol)
            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = []
                for _, row in df.iterrows():
                    date_val = row.get('date') or row.get('nav_date')
                    nav_val = row.get('nav') or row.get('nav_per_unit') or row.get('value')
                    
                    if pd.isna(date_val) or pd.isna(nav_val):
                        continue
                    
                    try:
                        parsed_date = pd.to_datetime(date_val).date()
                        records.append({
                            'date': parsed_date,
                            'nav': float(nav_val)
                        })
                    except Exception:
                        continue
                return records
            return []

        try:
            return retry_with_backoff(fetch_nav, max_retries=2)
        except Exception as e:
            bg_logger.error(f"Error fetching NAV from API for {symbol}: {e}")
            return []

    def _get_fund_nav_with_sync_db(
        self,
        db_session,
        symbol: str,
        fund_api,
        skip_api_sync: bool = False,
        fail_fast: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Sync version of NAV retrieval with database storage.
        Used by _compute_fund_performance_sync for efficiency.
        
        Args:
            db_session: SQLAlchemy sync Session
            symbol: Fund symbol
            fund_api: vnstock Fund instance for API calls
            skip_api_sync: If True, skip API sync and only return DB data (for fast loading)
        """
        from sqlalchemy import select as sync_select
        
        # Step 1: Load existing NAV records from database
        stmt = sync_select(FundNav).where(FundNav.symbol == symbol).order_by(FundNav.date)
        db_records = list(db_session.execute(stmt).scalars().all())
        
        # Step 2: Determine if we need to sync from API
        need_api_sync = False
        
        if not skip_api_sync:
            if not db_records:
                # No data in DB, need full sync
                need_api_sync = True
            else:
                latest_db_date = db_records[-1].date
                today = date.today()
                # Sync if latest DB date is more than 3 days old
                # Using 3 days to handle weekends (fund NAV only updates on trading days)
                if (today - latest_db_date).days > 3:
                    need_api_sync = True
        
        # Step 3: Sync from API if needed
        if need_api_sync:
            # Check if system is already rate limited before trying
            if sync_status.is_rate_limited:
                bg_logger.warning(f"Skipping NAV sync for {symbol} - already rate limited")
                return [
                    {'date': record.date.isoformat(), 'nav': record.nav}
                    for record in db_records
                ]

            try:
                if fund_api is None:
                    bg_logger.warning(f"Skipping NAV sync for {symbol} - fund API not available")
                    return [
                        {'date': record.date.isoformat(), 'nav': record.nav}
                        for record in db_records
                    ]

                def fetch_nav():
                    return fund_api.details.nav_report(symbol=symbol)
                
                nav_df = retry_with_backoff(fetch_nav, max_retries=3)
                
                if nav_df is not None and not nav_df.empty:
                    nav_df = self._flatten_columns(nav_df)
                    
                    # Get existing dates for deduplication
                    existing_dates = {r.date for r in db_records}
                    new_count = 0
                    
                    for _, row in nav_df.iterrows():
                        date_val = row.get('date') or row.get('nav_date')
                        nav_val = row.get('nav') or row.get('nav_per_unit') or row.get('value')
                        
                        if pd.isna(date_val) or pd.isna(nav_val):
                            continue
                        
                        try:
                            parsed_date = pd.to_datetime(date_val).date()
                            
                            if parsed_date not in existing_dates:
                                fund_nav = FundNav(
                                    symbol=symbol,
                                    date=parsed_date,
                                    nav=float(nav_val)
                                )
                                db_session.add(fund_nav)
                                existing_dates.add(parsed_date)
                                new_count += 1
                        except Exception:
                            continue
                    
                    if new_count > 0:
                        db_session.commit()
                        bg_logger.debug(f"Stored {new_count} new NAV records for fund {symbol}")

                        # Reload from DB
                        db_records = list(db_session.execute(stmt).scalars().all())

            except Exception as e:
                if _is_rate_limit_error(e):
                    bg_logger.warning(f"Rate limit hit while syncing NAV for {symbol}")
                    if fail_fast:
                        raise CircuitOpenError(f"Rate limited while syncing NAV for {symbol}")
                else:
                    bg_logger.error(f"Failed to fetch NAV for fund {symbol} from API: {e}")
        
        # Step 4: Return in frontend format
        return [
            {
                'date': record.date.isoformat(),
                'nav': record.nav
            }
            for record in db_records
        ]

    async def get_fund_top_holding(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch top stock holdings for a specific fund.
        """
        if self._is_cache_valid(self._fund_top_holding_cache, symbol, self._FUND_DETAILS_TTL):
            return self._fund_top_holding_cache[symbol][0]

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(_frontend_executor, self._fetch_fund_top_holding_sync, symbol)
        
        if data:
            self._fund_top_holding_cache[symbol] = (data, time.time())
        return data

    def _fetch_fund_top_holding_sync(self, symbol: str, fail_fast: bool = True) -> List[Dict[str, Any]]:
        """Fetch fund top holdings synchronously."""
        cached_data, is_fresh = self._get_fund_detail_cache_sync(symbol, "top_holding")
        if is_fresh:
            return cached_data

        stale_data = cached_data if cached_data else None

        def fetch_top_holding():
            fund = self._get_thread_local_fund_api()
            df = fund.details.top_holding(symbol=symbol)
            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    # Normalize field names for frontend
                    self._normalize_fund_field(record, 'ticker', ['stock_code', 'ticker', 'symbol'])
                    self._normalize_fund_field(record, 'allocation', ['net_asset_percent', 'allocation', 'weight', 'percentage'])
                    # Clean NaN values
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []

        try:
            data = retry_with_backoff(fetch_top_holding, max_retries=2)
            if data:
                self._upsert_fund_detail_cache_sync(symbol, "top_holding", data)
                return data
            return stale_data or []
        except Exception as e:
            logger.warning(f"Error fetching top holdings for {symbol}: {e}")
            return stale_data or []

    async def get_fund_industry_holding(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch industry allocation for a specific fund.
        """
        if self._is_cache_valid(self._fund_industry_holding_cache, symbol, self._FUND_DETAILS_TTL):
            return self._fund_industry_holding_cache[symbol][0]

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(_frontend_executor, self._fetch_fund_industry_holding_sync, symbol)
        
        if data:
            self._fund_industry_holding_cache[symbol] = (data, time.time())
        return data

    def _fetch_fund_industry_holding_sync(self, symbol: str, fail_fast: bool = True) -> List[Dict[str, Any]]:
        """Fetch fund industry holdings synchronously."""
        cached_data, is_fresh = self._get_fund_detail_cache_sync(symbol, "industry_holding")
        if is_fresh:
            return cached_data

        stale_data = cached_data if cached_data else None

        def fetch_industry():
            fund = self._get_thread_local_fund_api()
            df = fund.details.industry_holding(symbol=symbol)
            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    # Normalize field names for frontend
                    self._normalize_fund_field(record, 'allocation', ['net_asset_percent', 'allocation', 'weight', 'percentage'])
                    # Clean NaN values
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []

        try:
            data = retry_with_backoff(fetch_industry, max_retries=2)
            if data:
                self._upsert_fund_detail_cache_sync(symbol, "industry_holding", data)
                return data
            return stale_data or []
        except Exception as e:
            logger.warning(f"Error fetching industry holdings for {symbol}: {e}")
            return stale_data or []

    async def get_fund_asset_holding(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch asset type allocation for a specific fund.
        """
        if self._is_cache_valid(self._fund_asset_holding_cache, symbol, self._FUND_DETAILS_TTL):
            return self._fund_asset_holding_cache[symbol][0]

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(_frontend_executor, self._fetch_fund_asset_holding_sync, symbol)
        
        if data:
            self._fund_asset_holding_cache[symbol] = (data, time.time())
        return data

    def _fetch_fund_asset_holding_sync(self, symbol: str, fail_fast: bool = True) -> List[Dict[str, Any]]:
        """Fetch fund asset holdings synchronously."""
        cached_data, is_fresh = self._get_fund_detail_cache_sync(symbol, "asset_holding")
        if is_fresh:
            return cached_data

        stale_data = cached_data if cached_data else None

        def fetch_asset():
            fund = self._get_thread_local_fund_api()
            df = fund.details.asset_holding(symbol=symbol)
            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    # Normalize field names for frontend
                    self._normalize_fund_field(record, 'allocation', ['asset_percent', 'allocation', 'weight', 'percentage'])
                    # Clean NaN values
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []

        try:
            data = retry_with_backoff(fetch_asset, max_retries=2)
            if data:
                self._upsert_fund_detail_cache_sync(symbol, "asset_holding", data)
                return data
            return stale_data or []
        except Exception as e:
            logger.warning(f"Error fetching asset holdings for {symbol}: {e}")
            return stale_data or []

    # Background sync task (no memory cache - DB is the cache)
    _background_sync_task: asyncio.Task | None = None

    async def get_fund_performance_data(self) -> Dict[str, Any]:
        """
        Get aggregated fund performance data for comparison charts.
        Includes normalized NAV, periodic returns, and risk metrics.
        
        Always loads from database (which is fast since NAV data is persisted).
        Background sync runs if any fund data is stale (>3 days old).
        """
        from app.services.sync_status import sync_status

        loop = asyncio.get_event_loop()
        # Load from DB only (avoid API calls on request path)
        data = await loop.run_in_executor(
            None,
            lambda: self._compute_fund_performance_sync(skip_api_sync=True)
        )

        if data and data.get('funds'):
            data = data.copy()
            data['is_stale'] = False
            data['is_syncing'] = sync_status.fund_performance.is_syncing
            # Fill benchmarks from cache or fetch if missing
            if not data.get('benchmarks'):
                cached_benchmarks = self._get_cached_fund_benchmarks()
                if cached_benchmarks:
                    data['benchmarks'] = cached_benchmarks
                elif not sync_status.is_rate_limited and api_circuit_breaker.can_proceed():
                    try:
                        benchmarks = await loop.run_in_executor(
                            _frontend_executor,
                            self._fetch_fund_benchmarks_sync,
                            data.get('common_start_date')
                        )
                        if benchmarks:
                            self._set_cached_fund_benchmarks(benchmarks)
                            data['benchmarks'] = benchmarks
                    except (CircuitOpenError, RateLimitError) as e:
                        logger.warning(f"Skipping fund benchmarks fetch due to rate limit: {e}")
                    except Exception as e:
                        logger.warning(f"Error fetching fund benchmarks: {e}")
            # Trigger background sync if needed (non-blocking)
            self._trigger_background_sync_if_needed()
            return data

        # No DB data available - kick off background sync and return empty response
        should_sync = not sync_status.is_rate_limited
        if should_sync:
            self._trigger_background_sync_if_needed(force=True)

        return {
            "funds": [],
            "benchmarks": {},
            "common_start_date": None,
            "last_updated": None,
            "is_stale": True,
            "is_syncing": should_sync
        }
    
    def _trigger_background_sync_if_needed(self, force: bool = False):
        """Trigger background sync if not already running and data might be stale."""
        from app.services.sync_status import sync_status
        
        # Check if already syncing
        if sync_status.fund_performance.is_syncing:
            return
        
        # Check if there's an existing task that's still running
        if self._background_sync_task and not self._background_sync_task.done():
            return

        # Check if last sync was successful recently (within 6 hours)
        # This prevents redundant API hits on every page refresh
        if not force and sync_status.fund_performance.last_sync:
            try:
                last_sync_dt = datetime.fromisoformat(sync_status.fund_performance.last_sync)
                if (datetime.now() - last_sync_dt).total_seconds() < 6 * 3600:
                    return
            except (ValueError, TypeError):
                pass
        
        # Start background sync
        self._background_sync_task = asyncio.create_task(self._background_sync_coroutine())
    
    async def _background_sync_coroutine(self):
        """
        Background coroutine to sync fund NAV data incrementally.

        Processes funds in batches of 5, checking circuit breaker between batches
        and using asyncio.sleep for non-blocking delays. This prevents thread pool
        saturation and keeps the server responsive to frontend requests.
        """
        from app.services.sync_status import sync_status
        from app.core.circuit_breaker import api_circuit_breaker, CircuitOpenError

        BATCH_SIZE = 5
        BATCH_DELAY_SECONDS = 2.0  # Delay between batches to avoid rate limiting

        sync_status.start_fund_performance_sync()

        try:
            # Step 1: Get fund listing (fast, usually cached)
            loop = asyncio.get_event_loop()
            listing_df = await loop.run_in_executor(
                _background_executor,
                self._get_fund_listing_for_sync
            )

            if listing_df is None or listing_df.empty:
                bg_logger.warning("No funds to sync - empty listing")
                sync_status.complete_fund_performance_sync(success=True)
                return

            # Extract fund symbols
            fund_symbols = []
            for _, row in listing_df.iterrows():
                symbol = row.get('short_name') or row.get('fund_code') or row.get('symbol')
                if symbol:
                    fund_symbols.append(symbol)

            total_funds = len(fund_symbols)
            if total_funds == 0:
                sync_status.complete_fund_performance_sync(success=True)
                return

            bg_logger.info(f"Starting incremental fund sync: {total_funds} funds in batches of {BATCH_SIZE}")

            # Step 2: Process funds in batches
            processed = 0
            errors = 0

            for batch_start in range(0, total_funds, BATCH_SIZE):
                # Check circuit breaker at batch boundary
                if not api_circuit_breaker.can_proceed():
                    wait_time = api_circuit_breaker.time_until_half_open or 30.0
                    bg_logger.warning(f"Circuit breaker open, pausing sync for {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    # Check again after waiting
                    if not api_circuit_breaker.can_proceed():
                        bg_logger.error("Circuit breaker still open after wait, aborting sync")
                        sync_status.complete_fund_performance_sync(
                            success=False,
                            error="Rate limit - circuit breaker open"
                        )
                        return

                batch_end = min(batch_start + BATCH_SIZE, total_funds)
                batch_symbols = fund_symbols[batch_start:batch_end]

                bg_logger.debug(f"Processing batch {batch_start//BATCH_SIZE + 1}: funds {batch_start+1}-{batch_end}")

                # Process batch in a single executor call to reuse Fund API + DB session
                try:
                    batch_processed, batch_errors = await loop.run_in_executor(
                        _background_executor,
                        self._sync_fund_nav_batch_sync,
                        batch_symbols
                    )
                    processed += batch_processed
                    errors += batch_errors
                except CircuitOpenError as e:
                    bg_logger.warning(f"Circuit breaker tripped during fund batch: {e}")
                    errors += len(batch_symbols)
                    break  # Exit batch, will check circuit breaker at next batch boundary
                except Exception as e:
                    bg_logger.error(f"Error syncing fund batch {batch_start//BATCH_SIZE + 1}: {e}")
                    errors += len(batch_symbols)

                # Update progress
                progress = processed / total_funds
                sync_status.update_fund_performance_progress(progress)

                # Delay between batches (non-blocking)
                if batch_end < total_funds:
                    await asyncio.sleep(BATCH_DELAY_SECONDS)

            # Step 3: Mark complete
            if errors > total_funds * 0.5:  # More than 50% failed
                sync_status.complete_fund_performance_sync(
                    success=False,
                    error=f"Too many errors: {errors}/{total_funds} funds failed"
                )
            else:
                log_background_complete("Fund NAV Sync", f"Synced {processed}/{total_funds} funds")
                sync_status.complete_fund_performance_sync(success=True)

        except Exception as e:
            log_background_error("Fund NAV Sync", str(e))
            sync_status.complete_fund_performance_sync(success=False, error=str(e))

    def _get_fund_listing_for_sync(self):
        """Get fund listing for background sync, using cache if available."""
        # Check DB cache first (weekly)
        db_df, last_updated = self._get_fund_listing_df_from_db_sync()
        if db_df is not None and self._fund_listing_is_fresh(last_updated):
            self._fund_listing_df_cache = db_df
            self._fund_listing_df_timestamp = time.time()
            return db_df

        # Check in-memory cache
        if self._fund_listing_df_cache is not None and (time.time() - self._fund_listing_df_timestamp < self._FUND_LISTING_TTL):
            return self._fund_listing_df_cache

        try:
            def fetch_listing():
                fund = self._get_thread_local_fund_api()
                return fund.listing()

            listing_df = retry_with_backoff(fetch_listing, max_retries=2)
            if listing_df is not None and not listing_df.empty:
                listing_df = self._flatten_columns(listing_df)
                self._fund_listing_df_cache = listing_df
                self._fund_listing_df_timestamp = time.time()
                records = listing_df.to_dict('records')
                for record in records:
                    self._normalize_fund_field(record, 'symbol', ['short_name', 'fund_code'])
                    self._normalize_fund_field(record, 'name', ['name', 'short_name'])
                    self._normalize_fund_field(record, 'fund_owner', ['fund_owner_name', 'management_company'])
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                self._upsert_fund_listing_db_sync(records)
            return listing_df
        except Exception as e:
            bg_logger.warning(f"Failed to fetch fund listing: {e}")
            # Return stale cache as fallback
            if self._fund_listing_df_cache is not None:
                return self._fund_listing_df_cache
            return db_df

    def _sync_single_fund_nav(self, symbol: str, db_session=None, fund_api=None) -> bool:
        """
        Sync NAV data for a single fund from API to database.

        Returns True if sync was successful, False otherwise.
        Always fails fast on rate limit (raises CircuitOpenError).
        """
        from sqlalchemy.orm import Session
        from app.core.circuit_breaker import api_circuit_breaker, CircuitOpenError

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot sync {symbol}")

        own_session = False
        if db_session is None:
            engine = self._get_sync_engine()
            db_session = Session(engine)
            own_session = True

        if fund_api is None:
            try:
                fund_api = self._get_thread_local_fund_api()
            except CircuitOpenError:
                raise
            except Exception as e:
                bg_logger.error(f"Error initializing fund API for {symbol}: {e}")
                if own_session:
                    db_session.close()
                return False

        try:
            # Check existing data
            from sqlalchemy import select as sync_select
            stmt = sync_select(FundNav).where(FundNav.symbol == symbol).order_by(FundNav.date.desc()).limit(1)
            latest_record = db_session.execute(stmt).scalar_one_or_none()

            # Skip if data is fresh (within 3 days)
            if latest_record:
                days_old = (date.today() - latest_record.date).days
                if days_old <= 3:
                    bg_logger.debug(f"Fund {symbol} NAV data is fresh ({days_old} days old), skipping")
                    return True

            # Fetch from API
            try:
                def fetch_nav():
                    return fund_api.details.nav_report(symbol=symbol)

                nav_df = retry_with_backoff(
                    fetch_nav,
                    max_retries=2
                )

                if nav_df is None or nav_df.empty:
                    return False

                nav_df = self._flatten_columns(nav_df)

                # Get existing dates for deduplication
                existing_stmt = sync_select(FundNav.date).where(FundNav.symbol == symbol)
                existing_dates = {r for r in db_session.execute(existing_stmt).scalars().all()}

                new_count = 0
                for _, row in nav_df.iterrows():
                    date_val = row.get('date') or row.get('nav_date')
                    nav_val = row.get('nav') or row.get('nav_per_unit') or row.get('value')

                    if pd.isna(date_val) or pd.isna(nav_val):
                        continue

                    try:
                        parsed_date = pd.to_datetime(date_val).date()

                        if parsed_date not in existing_dates:
                            fund_nav = FundNav(
                                symbol=symbol,
                                date=parsed_date,
                                nav=float(nav_val)
                            )
                            db_session.add(fund_nav)
                            existing_dates.add(parsed_date)
                            new_count += 1
                    except Exception:
                        continue

                if new_count > 0:
                    db_session.commit()
                    bg_logger.debug(f"Stored {new_count} new NAV records for fund {symbol}")

                api_circuit_breaker.record_success()
                return True

            except Exception as e:
                error_name = type(e).__name__
                if error_name in {"RateLimitExceeded", "RateLimitError"} or "rate limit" in str(e).lower():
                    _record_rate_limit(reset_seconds=60.0)
                    raise CircuitOpenError(f"Rate limited while syncing {symbol}")
                bg_logger.error(f"Error syncing fund {symbol}: {e}")
                return False
        finally:
            if own_session:
                db_session.close()

    def _sync_fund_nav_batch_sync(self, symbols: List[str]) -> tuple[int, int]:
        """
        Sync NAV data for a batch of funds in a single thread.
        Reuses one Fund API instance and DB session to reduce rate-limit pressure.
        """
        # Keep under free-tier rate limit (60 req/min) with a small per-symbol delay
        per_symbol_delay = 1.1
        from sqlalchemy.orm import Session
        from app.core.circuit_breaker import api_circuit_breaker, CircuitOpenError

        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError("Circuit breaker open - cannot sync fund batch")

        try:
            fund_api = self._get_thread_local_fund_api()
        except CircuitOpenError:
            raise
        except Exception as e:
            raise

        engine = self._get_sync_engine()
        processed = 0
        errors = 0

        with Session(engine) as db_session:
            for symbol in symbols:
                if not api_circuit_breaker.can_proceed():
                    raise CircuitOpenError("Circuit breaker open - aborting fund batch")
                try:
                    ok = self._sync_single_fund_nav(symbol, db_session=db_session, fund_api=fund_api)
                    processed += 1
                    if not ok:
                        errors += 1
                    if per_symbol_delay:
                        time.sleep(per_symbol_delay)
                except CircuitOpenError:
                    raise
                except Exception as e:
                    errors += 1
                    bg_logger.error(f"Error syncing fund {symbol}: {e}")

        return processed, errors

    def _compute_fund_performance_sync(self, skip_api_sync: bool = False, fail_fast: bool = True) -> Dict[str, Any]:
        """
        Compute fund performance metrics synchronously.
        Normalizes NAV to base 100, calculates returns and risk metrics.
        
        Args:
            skip_api_sync: If True, only load from DB without API sync (for fast initial load).
            fail_fast: If True, raise immediately on rate limit. False for background tasks.
        """
        from sqlalchemy.orm import Session
        from sqlalchemy import select as sync_select
        import numpy as np

        # Risk-free rate for Vietnam (government bond yield ~4%)
        RISK_FREE_RATE = 0.04

        try:
            # Step 1: Get all funds (with retry and cache)
            listing_df = None

            # Check cache first
            if self._fund_listing_df_cache is not None and (time.time() - self._fund_listing_df_timestamp < self._FUND_LISTING_TTL):
                listing_df = self._fund_listing_df_cache
                bg_logger.debug("Using cached fund listing for performance calculation")

            fund_api = None
            if skip_api_sync:
                # Avoid API calls; fall back to DB-derived symbols if cache missing
                if listing_df is None:
                    db_df, _ = self._get_fund_listing_df_from_db_sync()
                    listing_df = db_df
                if listing_df is None:
                    engine = self._get_sync_engine()
                    with Session(engine) as db_session:
                        symbols = list(
                            db_session.execute(
                                sync_select(FundNav.symbol).distinct()
                            ).scalars().all()
                        )
                    if symbols:
                        listing_df = pd.DataFrame({
                            "short_name": symbols,
                            "name": symbols
                        })
                if listing_df is None or listing_df.empty:
                    return {"funds": [], "benchmarks": {}, "common_start_date": None, "last_updated": None}
            else:
                # Bail out early if already rate limited
                if sync_status.is_rate_limited or api_circuit_breaker.is_open:
                    return {"funds": [], "benchmarks": {}, "common_start_date": None, "last_updated": None}

                try:
                    fund_api = self._get_thread_local_fund_api()
                except CircuitOpenError:
                    if fail_fast:
                        raise
                    return {"funds": [], "benchmarks": {}, "common_start_date": None, "last_updated": None}
                except Exception:
                    raise

                if listing_df is None:
                    def fetch_listing():
                        return fund_api.listing()

                    try:
                        listing_df = retry_with_backoff(fetch_listing, max_retries=3)
                        if listing_df is not None and not listing_df.empty:
                            self._fund_listing_df_cache = listing_df
                            self._fund_listing_df_timestamp = time.time()
                            bg_logger.debug("Fetched and cached fresh fund listing from API")
                    except Exception as e:
                        logger.warning(f"Failed to fetch fund listing after retries: {e}")
                        # If we have a stale cache, use it as fallback
                        if self._fund_listing_df_cache is not None:
                            bg_logger.debug("Using stale fund listing cache as fallback")
                            listing_df = self._fund_listing_df_cache
                        else:
                            return {"funds": [], "benchmarks": {}, "common_start_date": None, "last_updated": None}
            
            if listing_df is None or listing_df.empty:
                return {"funds": [], "benchmarks": {}, "common_start_date": None, "last_updated": None}

            # Prepare for fetching NAVs
            if not skip_api_sync and fund_api is None:
                try:
                    fund_api = self._get_thread_local_fund_api()
                except CircuitOpenError:
                    if fail_fast:
                        raise
                    return {"funds": [], "benchmarks": {}, "common_start_date": None, "last_updated": None}
                except Exception:
                    raise

            funds_data = []
            all_nav_dates = set()

            # Set up sync database connection for NAV storage
            engine = self._get_sync_engine()

            # Step 2: Fetch NAV data for each fund (using DB-backed storage)
            with Session(engine) as db_session:
                for _, row in listing_df.iterrows():
                    # Stop early if rate limit is active
                    if sync_status.is_rate_limited or api_circuit_breaker.is_open:
                        bg_logger.warning("Rate limit detected - stopping fund performance computation early")
                        break

                    symbol = row.get('short_name') or row.get('fund_code') or row.get('symbol')
                    name = row.get('name') or row.get('fund_name') or symbol

                    if not symbol:
                        continue

                    try:
                        # Get NAV data from DB, sync from API if needed
                        nav_records = self._get_fund_nav_with_sync_db(
                            db_session, symbol, fund_api, skip_api_sync=skip_api_sync, fail_fast=fail_fast
                        )
                        
                        if len(nav_records) < 10:  # Need minimum data points
                            continue

                        for record in nav_records:
                            all_nav_dates.add(record['date'])

                        funds_data.append({
                            'symbol': symbol,
                            'name': name,
                            'nav_records': nav_records,
                            'data_start_date': nav_records[0]['date']
                        })

                    except CircuitOpenError as e:
                        bg_logger.warning(f"Rate limit during fund NAV fetch for {symbol}: {e}")
                        break
                    except Exception as e:
                        bg_logger.error(f"Error fetching NAV for fund {symbol}: {e}")
                        continue

            if not funds_data:
                return {"funds": [], "benchmarks": {}, "common_start_date": None, "last_updated": None}

            # Step 3: Find common start date (oldest date where most funds have data)
            sorted_dates = sorted(all_nav_dates)
            common_start_date = sorted_dates[0] if sorted_dates else None

            # Step 4: Process each fund - normalize and calculate metrics
            processed_funds = []
            for fund_data in funds_data:
                nav_records = fund_data['nav_records']

                # Convert to dataframe for easier calculations
                df = pd.DataFrame(nav_records)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)

                if len(df) < 2:
                    continue

                # Normalize NAV (base = 100 at start)
                base_nav = df.iloc[0]['nav']
                df['normalized_nav'] = (df['nav'] / base_nav) * 100

                # Calculate daily returns for volatility
                df['daily_return'] = df['nav'].pct_change()

                # Calculate periodic returns
                today = datetime.now()
                returns = {}
                yearly_returns = {}

                # YTD
                ytd_start = datetime(today.year, 1, 1)
                ytd_df = df[df['date'] >= ytd_start]
                if len(ytd_df) >= 2:
                    returns['ytd'] = round(((ytd_df.iloc[-1]['nav'] / ytd_df.iloc[0]['nav']) - 1) * 100, 2)

                # 1Y, 3Y, 5Y returns
                for years, key in [(1, '1y'), (3, '3y'), (5, '5y')]:
                    start_date = today - timedelta(days=years * 365)
                    period_df = df[df['date'] >= start_date]
                    if len(period_df) >= 2:
                        returns[key] = round(((period_df.iloc[-1]['nav'] / period_df.iloc[0]['nav']) - 1) * 100, 2)
                    else:
                        returns[key] = None

                # All-times return (from inception)
                if len(df) >= 2:
                    returns['all'] = round(((df.iloc[-1]['nav'] / df.iloc[0]['nav']) - 1) * 100, 2)

                # Yearly returns for heatmap
                first_year = df['date'].min().year
                for year in range(first_year, today.year + 1):
                    year_start = datetime(year, 1, 1)
                    year_end = datetime(year, 12, 31)
                    year_df = df[(df['date'] >= year_start) & (df['date'] <= year_end)]
                    if len(year_df) >= 2:
                        yearly_returns[str(year)] = round(
                            ((year_df.iloc[-1]['nav'] / year_df.iloc[0]['nav']) - 1) * 100, 2
                        )

                # Risk metrics (annualized)
                annualized_return = None
                annualized_volatility = None
                sharpe_ratio = None

                if returns.get('1y') is not None:
                    annualized_return = returns['1y']

                    # Calculate volatility from daily returns (past year)
                    one_year_ago = today - timedelta(days=365)
                    year_df = df[df['date'] >= one_year_ago]
                    if len(year_df) > 20:
                        daily_vol = year_df['daily_return'].std()
                        annualized_volatility = round(daily_vol * np.sqrt(252) * 100, 2)  # 252 trading days

                        if annualized_volatility and annualized_volatility > 0:
                            sharpe_ratio = round((annualized_return / 100 - RISK_FREE_RATE) / (annualized_volatility / 100), 2)

                # Prepare NAV history for charts (Sample to weekly to reduce payload)
                nav_history = []
                # resample to weekly, taking the last available point in each week
                # This naturally uses Sunday as the label, providing a consistent axis
                df_weekly = df.resample('W', on='date').last().dropna().reset_index()

                for _, row in df_weekly.iterrows():
                    nav_history.append({
                        'date': row['date'].strftime('%Y-%m-%d'),
                        'normalized_nav': round(row['normalized_nav'], 2),
                        'raw_nav': round(row['nav'], 2)
                    })

                processed_funds.append({
                    'symbol': fund_data['symbol'],
                    'name': fund_data['name'],
                    'data_start_date': fund_data['data_start_date'],
                    'nav_history': nav_history,
                    'returns': returns,
                    'risk_metrics': {
                        'annualized_return': annualized_return,
                        'annualized_volatility': annualized_volatility,
                        'sharpe_ratio': sharpe_ratio
                    },
                    'yearly_returns': yearly_returns
                })

            # Step 5: Fetch benchmark data (VN-Index and VN30)
            benchmarks = {}
            if not skip_api_sync and not sync_status.is_rate_limited and api_circuit_breaker.can_proceed():
                for benchmark_symbol in ['VNINDEX', 'VN30']:
                    try:
                        benchmark_data = self._fetch_benchmark_data_sync(benchmark_symbol, common_start_date)
                        if benchmark_data:
                            benchmarks[benchmark_symbol] = benchmark_data
                    except Exception as e:
                        logger.warning(f"Error fetching benchmark {benchmark_symbol}: {e}")

            return {
                'funds': processed_funds,
                'benchmarks': benchmarks,
                'common_start_date': common_start_date,
                'last_updated': datetime.now().isoformat()
            }

        except SystemExit as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=60.0)
                return {"funds": [], "benchmarks": {}, "common_start_date": None, "last_updated": None}
            raise
        except Exception as e:
            logger.error(f"Error computing fund performance: {e}")
            import traceback
            bg_logger.error(f"Stack trace for fund performance error:\n{traceback.format_exc()}")
            return {"funds": [], "benchmarks": {}, "common_start_date": None, "last_updated": None}

    def _fetch_benchmark_data_sync(self, symbol: str, common_start_date: str | None) -> Dict[str, Any] | None:
        """
        Fetch and process benchmark (VN-Index or VN30) data.
        """
        from vnstock import Vnstock
        import numpy as np

        RISK_FREE_RATE = 0.04

        try:
            today = datetime.now()
            start_date = common_start_date or (today - timedelta(days=365 * 5)).strftime('%Y-%m-%d')

            vs = Vnstock(symbol=symbol, source='VCI')
            stock = vs.stock()
            
            # Wrap benchmark fetch in retry logic
            def fetch_benchmark_history():
                return stock.quote.history(start=start_date, end=today.strftime('%Y-%m-%d'))
            
            try:
                df = retry_with_backoff(fetch_benchmark_history, max_retries=3)
            except Exception as e:
                logger.warning(f"Failed to fetch benchmark {symbol} after retries: {e}")
                return None

            if df is None or df.empty:
                return None

            df['date'] = pd.to_datetime(df['time'])
            df = df.sort_values('date').reset_index(drop=True)

            # Use close price as "NAV"
            df['nav'] = df['close']
            base_nav = df.iloc[0]['nav']
            df['normalized_nav'] = (df['nav'] / base_nav) * 100
            df['daily_return'] = df['nav'].pct_change()

            # Calculate returns
            returns = {}
            yearly_returns = {}

            # YTD
            ytd_start = datetime(today.year, 1, 1)
            ytd_df = df[df['date'] >= ytd_start]
            if len(ytd_df) >= 2:
                returns['ytd'] = round(((ytd_df.iloc[-1]['nav'] / ytd_df.iloc[0]['nav']) - 1) * 100, 2)

            # 1Y, 3Y, 5Y returns
            for years, key in [(1, '1y'), (3, '3y'), (5, '5y')]:
                period_start = today - timedelta(days=years * 365)
                period_df = df[df['date'] >= period_start]
                if len(period_df) >= 2:
                    returns[key] = round(((period_df.iloc[-1]['nav'] / period_df.iloc[0]['nav']) - 1) * 100, 2)
                else:
                    returns[key] = None

            # All-times return
            if len(df) >= 2:
                returns['all'] = round(((df.iloc[-1]['nav'] / df.iloc[0]['nav']) - 1) * 100, 2)

            # Yearly returns
            first_year = df['date'].min().year
            for year in range(first_year, today.year + 1):
                year_start = datetime(year, 1, 1)
                year_end = datetime(year, 12, 31)
                year_df = df[(df['date'] >= year_start) & (df['date'] <= year_end)]
                if len(year_df) >= 2:
                    yearly_returns[str(year)] = round(
                        ((year_df.iloc[-1]['nav'] / year_df.iloc[0]['nav']) - 1) * 100, 2
                    )

            # Risk metrics
            annualized_return = returns.get('1y')
            annualized_volatility = None
            sharpe_ratio = None

            one_year_ago = today - timedelta(days=365)
            year_df = df[df['date'] >= one_year_ago]
            if len(year_df) > 20:
                daily_vol = year_df['daily_return'].std()
                annualized_volatility = round(daily_vol * np.sqrt(252) * 100, 2)

                if annualized_volatility and annualized_volatility > 0 and annualized_return is not None:
                    sharpe_ratio = round((annualized_return / 100 - RISK_FREE_RATE) / (annualized_volatility / 100), 2)

            # NAV history (Sample to weekly to reduce payload)
            nav_history = []
            # resample to weekly
            df_weekly = df.resample('W', on='date').last().dropna().reset_index()

            for _, row in df_weekly.iterrows():
                nav_history.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'normalized_nav': round(row['normalized_nav'], 2),
                    'raw_nav': round(row['nav'], 2)
                })

            name_map = {'VNINDEX': 'VN-Index', 'VN30': 'VN30'}

            return {
                'symbol': symbol,
                'name': name_map.get(symbol, symbol),
                'nav_history': nav_history,
                'returns': returns,
                'risk_metrics': {
                    'annualized_return': annualized_return,
                    'annualized_volatility': annualized_volatility,
                    'sharpe_ratio': sharpe_ratio
                },
                'yearly_returns': yearly_returns
            }

        except Exception as e:
            logger.warning(f"Error fetching benchmark data for {symbol}: {e}")
            return None

    def _fetch_fund_benchmarks_sync(self, common_start_date: str | None) -> Dict[str, Any]:
        """Fetch benchmark metrics for fund performance charts."""
        benchmarks: Dict[str, Any] = {}
        for benchmark_symbol in ['VNINDEX', 'VN30']:
            try:
                benchmark_data = self._fetch_benchmark_data_sync(
                    benchmark_symbol,
                    common_start_date
                )
                if benchmark_data:
                    benchmarks[benchmark_symbol] = benchmark_data
            except Exception as e:
                logger.warning(f"Error fetching fund benchmark {benchmark_symbol}: {e}")
        return benchmarks


# Singleton instance
vnstock_service = VnstockService()
