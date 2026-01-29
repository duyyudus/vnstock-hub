from __future__ import annotations

from typing import List, Dict, Any
import asyncio
import threading
from datetime import datetime, date, timedelta
import time
import pandas as pd

from sqlalchemy import select, and_

from app.db.database import async_session
from app.db.models import FundNav, FundDetailCache, FundListing
from app.services.sync_status import sync_status
from app.core.logging_config import (
    log_background_start,
    log_background_complete,
    log_background_error
)

from .core import (
    frontend_executor,
    background_executor,
    logger,
    bg_logger,
    api_circuit_breaker,
    CircuitOpenError,
    RateLimitError,
    retry_with_backoff,
    _record_rate_limit,
    _is_rate_limit_error,
    get_sync_engine,
    get_thread_local_fund_api,
    _flatten_columns,
)


class FundsService:
    """Fund-related operations, caching, and background sync."""

    def __init__(self) -> None:
        # Fund holding caches (no DB backing): {key: (data, timestamp)}
        self._fund_listing_cache: Dict[str, Any] = {}
        self._fund_top_holding_cache: Dict[str, Any] = {}
        self._fund_industry_holding_cache: Dict[str, Any] = {}
        self._fund_asset_holding_cache: Dict[str, Any] = {}
        self._fund_listing_df_cache: pd.DataFrame | None = None
        self._fund_listing_df_timestamp = 0.0
        self._fund_listing_refresh_lock = threading.Lock()
        # Fund benchmark cache to avoid API hits on every request
        self._fund_benchmark_cache: Dict[str, Any] = {}

        # Cache TTLs in seconds (for holdings without DB backing)
        self._FUND_LISTING_TTL = 3600  # 1 hour
        self._FUND_DETAILS_TTL = 1800  # 30 minutes
        self._FUND_LISTING_DB_TTL = 7 * 24 * 3600  # 7 days
        self._FUND_BENCHMARK_TTL = 6 * 3600  # 6 hours

        # Background sync task (no memory cache - DB is the cache)
        self._background_sync_task: asyncio.Task | None = None

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

        engine = get_sync_engine()
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

        engine = get_sync_engine()
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

        engine = get_sync_engine()
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

        engine = get_sync_engine()
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

    async def get_fund_listing(self, fund_type: str = "") -> List[Dict[str, Any]]:
        """
        Fetch all available funds.
        """
        cache_key = fund_type or "all"
        if self._is_cache_valid(self._fund_listing_cache, cache_key, self._FUND_LISTING_TTL):
            return self._fund_listing_cache[cache_key][0]

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(frontend_executor, self._fetch_fund_listing_sync, fund_type)

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
                fund = get_thread_local_fund_api()
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
            df = _flatten_columns(df)
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
        return await self._get_fund_nav_with_sync(symbol)

    async def _get_fund_nav_with_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get NAV data from database, syncing missing data from API.
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
                if (today - latest_db_date).days > 3:
                    need_api_sync = True

            # Step 3: Sync from API if needed
            if need_api_sync:
                loop = asyncio.get_event_loop()
                api_records = await loop.run_in_executor(
                    background_executor,
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
            fund = get_thread_local_fund_api()
            df = fund.details.nav_report(symbol=symbol)
            if df is not None and not df.empty:
                df = _flatten_columns(df)
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
                    nav_df = _flatten_columns(nav_df)

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
        """Fetch top stock holdings for a specific fund."""
        if self._is_cache_valid(self._fund_top_holding_cache, symbol, self._FUND_DETAILS_TTL):
            return self._fund_top_holding_cache[symbol][0]

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(frontend_executor, self._fetch_fund_top_holding_sync, symbol)

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
            fund = get_thread_local_fund_api()
            df = fund.details.top_holding(symbol=symbol)
            if df is not None and not df.empty:
                df = _flatten_columns(df)
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
        """Fetch industry allocation for a specific fund."""
        if self._is_cache_valid(self._fund_industry_holding_cache, symbol, self._FUND_DETAILS_TTL):
            return self._fund_industry_holding_cache[symbol][0]

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(frontend_executor, self._fetch_fund_industry_holding_sync, symbol)

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
            fund = get_thread_local_fund_api()
            df = fund.details.industry_holding(symbol=symbol)
            if df is not None and not df.empty:
                df = _flatten_columns(df)
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
        """Fetch asset type allocation for a specific fund."""
        if self._is_cache_valid(self._fund_asset_holding_cache, symbol, self._FUND_DETAILS_TTL):
            return self._fund_asset_holding_cache[symbol][0]

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(frontend_executor, self._fetch_fund_asset_holding_sync, symbol)

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
            fund = get_thread_local_fund_api()
            df = fund.details.asset_holding(symbol=symbol)
            if df is not None and not df.empty:
                df = _flatten_columns(df)
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

    async def get_fund_performance_data(self) -> Dict[str, Any]:
        """
        Get aggregated fund performance data for comparison charts.
        """
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
                            frontend_executor,
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
        # Check if already syncing
        if sync_status.fund_performance.is_syncing:
            return

        # Check if there's an existing task that's still running
        if self._background_sync_task and not self._background_sync_task.done():
            return

        # Check if last sync was successful recently (within 6 hours)
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
        """
        BATCH_SIZE = 5
        BATCH_DELAY_SECONDS = 2.0  # Delay between batches to avoid rate limiting

        sync_status.start_fund_performance_sync()

        try:
            # Step 1: Get fund listing (fast, usually cached)
            loop = asyncio.get_event_loop()
            listing_df = await loop.run_in_executor(
                background_executor,
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

                bg_logger.debug(f"Processing batch {batch_start // BATCH_SIZE + 1}: funds {batch_start + 1}-{batch_end}")

                # Process batch in a single executor call to reuse Fund API + DB session
                try:
                    batch_processed, batch_errors = await loop.run_in_executor(
                        background_executor,
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
                    bg_logger.error(f"Error syncing fund batch {batch_start // BATCH_SIZE + 1}: {e}")
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
                fund = get_thread_local_fund_api()
                return fund.listing()

            listing_df = retry_with_backoff(fetch_listing, max_retries=2)
            if listing_df is not None and not listing_df.empty:
                listing_df = _flatten_columns(listing_df)
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
            engine = get_sync_engine()
            db_session = Session(engine)
            own_session = True

        if fund_api is None:
            try:
                fund_api = get_thread_local_fund_api()
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

                nav_df = retry_with_backoff(fetch_nav, max_retries=2)

                if nav_df is None or nav_df.empty:
                    return False

                nav_df = _flatten_columns(nav_df)

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
            fund_api = get_thread_local_fund_api()
        except CircuitOpenError:
            raise
        except Exception as e:
            raise

        engine = get_sync_engine()
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
                    engine = get_sync_engine()
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
                    fund_api = get_thread_local_fund_api()
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
                    fund_api = get_thread_local_fund_api()
                except CircuitOpenError:
                    if fail_fast:
                        raise
                    return {"funds": [], "benchmarks": {}, "common_start_date": None, "last_updated": None}
                except Exception:
                    raise

            funds_data = []
            all_nav_dates = set()

            # Set up sync database connection for NAV storage
            engine = get_sync_engine()

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
        """Fetch and process benchmark (VN-Index or VN30) data."""
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
