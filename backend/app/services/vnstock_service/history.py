from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import asyncio
import threading
from datetime import datetime, date, timedelta
import pandas as pd

from sqlalchemy import select, and_, func
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.database import async_session
from app.db.models import StockDailyPrice, StockCompany, StockHistoryBackfillState
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


@dataclass(frozen=True)
class BackfillJob:
    symbol: str
    reason: str


# Completeness backfill defaults
BACKFILL_DEFAULT_START_DATE = date(2010, 1, 1)
BACKFILL_RETRY_COOLDOWN = timedelta(hours=24)
BACKFILL_EXHAUSTED_RETRY_COOLDOWN = timedelta(days=7)
BACKFILL_NO_PROGRESS_LIMIT = 3
BACKFILL_CHUNK_DAYS = 730
BACKFILL_SCHEDULER_INTERVAL_SECONDS = 24 * 3600
BACKFILL_ACTIVITY_WINDOW_DAYS = 14
BACKFILL_LONG_TAIL_BATCH_SIZE = 30
BACKFILL_COVERAGE_TOLERANCE_DAYS = 30
BACKFILL_CORE_GROUPS = ("VN30", "VN100")


class HistoryService:
    """Historical price and volume data operations."""

    def __init__(self) -> None:
        # Track background sync task for weekly prices
        self._weekly_prices_sync_task: asyncio.Task | None = None
        self._weekly_prices_syncing_symbols = set()
        self._weekly_prices_retry_cooldown = timedelta(minutes=10)

        # Track completeness backfill tasks
        self._backfill_worker_task: asyncio.Task | None = None
        self._backfill_scheduler_task: asyncio.Task | None = None
        self._backfill_queue: asyncio.Queue[BackfillJob] = asyncio.Queue()
        self._backfill_queued_symbols = set()
        self._backfill_start_lock: asyncio.Lock | None = None
        self._symbol_sync_locks: Dict[str, threading.Lock] = {}
        self._symbol_sync_locks_guard = threading.Lock()

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
        """Legacy backfill workers are disabled in deterministic sync mode."""
        return

    async def stop_background_workers(self) -> None:
        """Legacy backfill workers are disabled in deterministic sync mode."""
        self._backfill_worker_task = None
        self._backfill_scheduler_task = None
        self._backfill_queued_symbols.clear()
        return

    async def schedule_completeness_backfill(
        self,
        symbols: List[str],
        target_start_date: date | None = None,
        reason: str = "request",
        mark_seen: bool = True
    ) -> None:
        """
        Register symbols for completeness backfill and enqueue eligible jobs.
        """
        clean_symbols = list(dict.fromkeys(s[:3].upper() for s in symbols if s))
        if not clean_symbols:
            return

        await self._ensure_backfill_workers()
        desired_start = target_start_date or BACKFILL_DEFAULT_START_DATE
        await self._enqueue_backfill_symbols(
            clean_symbols,
            desired_start,
            reason=reason,
            mark_seen=mark_seen
        )

    async def _schedule_completeness_backfill_safe(
        self,
        symbols: List[str],
        target_start_date: date,
        reason: str,
        mark_seen: bool
    ) -> None:
        try:
            await self.schedule_completeness_backfill(
                symbols=symbols,
                target_start_date=target_start_date,
                reason=reason,
                mark_seen=mark_seen
            )
        except Exception as e:
            logger.warning(f"Error scheduling completeness backfill ({reason}): {e}")

    async def _ensure_backfill_workers(self) -> None:
        if (
            self._backfill_worker_task
            and not self._backfill_worker_task.done()
            and self._backfill_scheduler_task
            and not self._backfill_scheduler_task.done()
        ):
            return

        if self._backfill_start_lock is None:
            self._backfill_start_lock = asyncio.Lock()

        async with self._backfill_start_lock:
            if not self._backfill_worker_task or self._backfill_worker_task.done():
                self._backfill_worker_task = asyncio.create_task(self._backfill_worker_loop())
            if not self._backfill_scheduler_task or self._backfill_scheduler_task.done():
                self._backfill_scheduler_task = asyncio.create_task(self._backfill_scheduler_loop())

    async def _backfill_worker_loop(self) -> None:
        while True:
            job = await self._backfill_queue.get()
            try:
                await self._process_backfill_job(job)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                bg_logger.error(f"Error in backfill worker for {job.symbol}: {e}")
            finally:
                self._backfill_queued_symbols.discard(job.symbol)
                self._backfill_queue.task_done()

    async def _backfill_scheduler_loop(self) -> None:
        while True:
            try:
                await self._run_backfill_sweep()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                bg_logger.error(f"Error in backfill scheduler sweep: {e}")

            await asyncio.sleep(BACKFILL_SCHEDULER_INTERVAL_SECONDS)

    async def _run_backfill_sweep(self) -> None:
        """Scheduled sweep: core symbols + recently active + trickle long-tail."""
        loop = asyncio.get_event_loop()

        core_symbols = await loop.run_in_executor(frontend_executor, self._fetch_core_symbols_sync)
        await self._enqueue_backfill_symbols(
            core_symbols,
            BACKFILL_DEFAULT_START_DATE,
            reason="scheduled_core",
            mark_seen=False
        )

        active_symbols = await self._get_recently_active_symbols(limit=200)
        await self._enqueue_backfill_symbols(
            active_symbols,
            BACKFILL_DEFAULT_START_DATE,
            reason="scheduled_active",
            mark_seen=False
        )

        exclude = set(core_symbols) | set(active_symbols)
        long_tail_symbols = await self._get_long_tail_symbols(
            limit=BACKFILL_LONG_TAIL_BATCH_SIZE,
            exclude_symbols=exclude
        )
        await self._enqueue_backfill_symbols(
            long_tail_symbols,
            BACKFILL_DEFAULT_START_DATE,
            reason="scheduled_long_tail",
            mark_seen=False
        )

    def _fetch_core_symbols_sync(self) -> List[str]:
        """Fetch VN30 and VN100 constituents for scheduled backfill priority."""
        from vnstock import Listing

        if not api_circuit_breaker.can_proceed():
            return []

        symbols = set()
        listing = Listing(source='VCI')

        for group in BACKFILL_CORE_GROUPS:
            try:
                series = listing.symbols_by_group(group)
                api_circuit_breaker.record_success()
                if series is None:
                    continue
                symbols.update(str(item).strip().upper()[:3] for item in series.tolist() if item)
            except (SystemExit, Exception) as e:
                if _is_rate_limit_error(e):
                    _record_rate_limit(reset_seconds=60.0)
                    break
                logger.warning(f"Error fetching core symbols for {group}: {e}")

        return sorted(symbols)

    async def _get_recently_active_symbols(self, limit: int = 200) -> List[str]:
        cutoff = datetime.utcnow() - timedelta(days=BACKFILL_ACTIVITY_WINDOW_DAYS)

        async with async_session() as session:
            stmt = (
                select(StockHistoryBackfillState.symbol)
                .where(
                    StockHistoryBackfillState.last_seen_at.is_not(None),
                    StockHistoryBackfillState.last_seen_at >= cutoff
                )
                .order_by(StockHistoryBackfillState.last_seen_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return [row[0] for row in result.all()]

    async def _get_long_tail_symbols(self, limit: int, exclude_symbols: set[str]) -> List[str]:
        async with async_session() as session:
            stmt = (
                select(StockCompany.symbol)
                .outerjoin(
                    StockHistoryBackfillState,
                    StockHistoryBackfillState.symbol == StockCompany.symbol
                )
            )
            if exclude_symbols:
                stmt = stmt.where(~StockCompany.symbol.in_(list(exclude_symbols)))

            stmt = (
                stmt.order_by(
                    StockHistoryBackfillState.last_attempt_at.asc().nullsfirst(),
                    StockCompany.symbol.asc()
                )
                .limit(limit)
            )

            result = await session.execute(stmt)
            return [row[0] for row in result.all() if row[0]]

    def _is_backfill_eligible(
        self,
        *,
        is_exhausted: bool,
        exhausted_until: datetime | None,
        next_attempt_at: datetime | None,
        now: datetime
    ) -> bool:
        if is_exhausted:
            if exhausted_until is None or exhausted_until > now:
                return False
        if next_attempt_at and next_attempt_at > now:
            return False
        return True

    async def _enqueue_backfill_symbols(
        self,
        symbols: List[str],
        target_start_date: date,
        reason: str,
        mark_seen: bool
    ) -> None:
        if not symbols:
            return

        clean_symbols = list(dict.fromkeys(s[:3].upper() for s in symbols if s))
        now = datetime.utcnow()
        to_queue: List[str] = []

        async with async_session() as session:
            existing_stmt = select(StockHistoryBackfillState).where(
                StockHistoryBackfillState.symbol.in_(clean_symbols)
            )
            existing_rows = await session.execute(existing_stmt)
            existing = {row.symbol: row for row in existing_rows.scalars().all()}

            for symbol in clean_symbols:
                state = existing.get(symbol)
                if state is None:
                    state = StockHistoryBackfillState(
                        symbol=symbol,
                        target_start_date=target_start_date,
                        last_seen_at=now if mark_seen else None,
                        next_attempt_at=now,
                        no_progress_attempts=0,
                        is_exhausted=False,
                    )
                    session.add(state)
                    existing[symbol] = state
                else:
                    if mark_seen:
                        state.last_seen_at = now

                    if state.target_start_date is None or target_start_date < state.target_start_date:
                        state.target_start_date = target_start_date

                    if state.is_exhausted and state.exhausted_until and state.exhausted_until <= now:
                        state.is_exhausted = False
                        state.exhausted_until = None
                        state.no_progress_attempts = 0

                if symbol in self._backfill_queued_symbols:
                    continue

                if not self._is_backfill_eligible(
                    is_exhausted=state.is_exhausted,
                    exhausted_until=state.exhausted_until,
                    next_attempt_at=state.next_attempt_at,
                    now=now
                ):
                    continue

                to_queue.append(symbol)

            await session.commit()

        for symbol in to_queue:
            self._backfill_queued_symbols.add(symbol)
            self._backfill_queue.put_nowait(BackfillJob(symbol=symbol, reason=reason))

    async def _get_symbol_price_bounds(self, symbol: str) -> tuple[date | None, date | None]:
        async with async_session() as session:
            stmt = select(
                func.min(StockDailyPrice.date),
                func.max(StockDailyPrice.date)
            ).where(StockDailyPrice.symbol == symbol)
            result = await session.execute(stmt)
            row = result.one()
            return row[0], row[1]

    async def _get_backfill_state_snapshot(
        self, symbol: str
    ) -> tuple[date, bool, datetime | None, datetime | None] | None:
        async with async_session() as session:
            stmt = select(StockHistoryBackfillState).where(StockHistoryBackfillState.symbol == symbol)
            state = (await session.execute(stmt)).scalar_one_or_none()
            if state is None:
                return None

            return (
                state.target_start_date or BACKFILL_DEFAULT_START_DATE,
                state.is_exhausted,
                state.exhausted_until,
                state.next_attempt_at
            )

    def _is_backfill_covered(self, oldest_date: date | None, target_start_date: date) -> bool:
        if oldest_date is None:
            return False
        threshold = target_start_date + timedelta(days=BACKFILL_COVERAGE_TOLERANCE_DAYS)
        return oldest_date <= threshold

    def _resolve_backfill_window(
        self,
        oldest_date: date | None,
        target_start_date: date
    ) -> tuple[date | None, date | None]:
        if oldest_date is None:
            return target_start_date, date.today()

        if self._is_backfill_covered(oldest_date, target_start_date):
            return None, None

        end_date = oldest_date - timedelta(days=1)
        start_date = max(target_start_date, end_date - timedelta(days=BACKFILL_CHUNK_DAYS))
        if end_date < start_date:
            return None, None
        return start_date, end_date

    async def _defer_backfill_attempt(self, symbol: str, error: str, cooldown: timedelta) -> None:
        now = datetime.utcnow()
        async with async_session() as session:
            stmt = select(StockHistoryBackfillState).where(StockHistoryBackfillState.symbol == symbol)
            state = (await session.execute(stmt)).scalar_one_or_none()
            if state is None:
                return

            state.last_attempt_at = now
            state.next_attempt_at = now + cooldown
            state.last_error = error[:500]
            await session.commit()

    async def _record_backfill_attempt(
        self,
        symbol: str,
        target_start_date: date,
        oldest_date: date | None,
        latest_date: date | None,
        progressed: bool,
        covered: bool,
        error: str | None = None
    ) -> None:
        now = datetime.utcnow()
        async with async_session() as session:
            stmt = select(StockHistoryBackfillState).where(StockHistoryBackfillState.symbol == symbol)
            state = (await session.execute(stmt)).scalar_one_or_none()
            if state is None:
                state = StockHistoryBackfillState(symbol=symbol, target_start_date=target_start_date)
                session.add(state)

            if state.target_start_date is None or target_start_date < state.target_start_date:
                state.target_start_date = target_start_date

            state.oldest_date = oldest_date
            state.latest_date = latest_date
            state.last_attempt_at = now
            state.last_error = error[:500] if error else None

            if covered or progressed:
                state.no_progress_attempts = 0
                state.is_exhausted = False
                state.exhausted_until = None
            else:
                state.no_progress_attempts = (state.no_progress_attempts or 0) + 1
                if state.no_progress_attempts >= BACKFILL_NO_PROGRESS_LIMIT:
                    state.is_exhausted = True
                    state.exhausted_until = now + BACKFILL_EXHAUSTED_RETRY_COOLDOWN

            if state.is_exhausted and state.exhausted_until:
                state.next_attempt_at = state.exhausted_until
            else:
                state.next_attempt_at = now + BACKFILL_RETRY_COOLDOWN

            await session.commit()

    async def _process_backfill_job(self, job: BackfillJob) -> None:
        now = datetime.utcnow()

        snapshot = await self._get_backfill_state_snapshot(job.symbol)
        if snapshot is None:
            return

        target_start_date, is_exhausted, exhausted_until, next_attempt_at = snapshot
        if not self._is_backfill_eligible(
            is_exhausted=is_exhausted,
            exhausted_until=exhausted_until,
            next_attempt_at=next_attempt_at,
            now=now
        ):
            return

        if sync_status.is_rate_limited or not api_circuit_breaker.can_proceed():
            await self._defer_backfill_attempt(
                job.symbol,
                "Rate limited - backfill deferred",
                BACKFILL_RETRY_COOLDOWN
            )
            return

        old_oldest, old_latest = await self._get_symbol_price_bounds(job.symbol)
        fetch_start, fetch_end = self._resolve_backfill_window(old_oldest, target_start_date)

        if fetch_start is None or fetch_end is None:
            await self._record_backfill_attempt(
                symbol=job.symbol,
                target_start_date=target_start_date,
                oldest_date=old_oldest,
                latest_date=old_latest,
                progressed=True,
                covered=True
            )
            return

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                background_executor,
                self._upsert_stock_price_history,
                job.symbol,
                fetch_start,
                fetch_end
            )
        except Exception as e:
            await self._record_backfill_attempt(
                symbol=job.symbol,
                target_start_date=target_start_date,
                oldest_date=old_oldest,
                latest_date=old_latest,
                progressed=False,
                covered=False,
                error=str(e)
            )
            return

        new_oldest, new_latest = await self._get_symbol_price_bounds(job.symbol)
        progressed = False
        if old_oldest is None and new_oldest is not None:
            progressed = True
        elif old_oldest and new_oldest and new_oldest < old_oldest:
            progressed = True

        covered = self._is_backfill_covered(new_oldest, target_start_date)
        await self._record_backfill_attempt(
            symbol=job.symbol,
            target_start_date=target_start_date,
            oldest_date=new_oldest,
            latest_date=new_latest,
            progressed=progressed,
            covered=covered
        )

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
        clean_symbols = list(dict.fromkeys(s[:3].upper() for s in symbols if s))

        # Calculate date range
        start_date = date(start_year, 1, 1)
        end_date = date.today()
        # Load from database
        stocks_data = await self._load_weekly_prices_from_db(clean_symbols, start_date, end_date)

        # Check freshness + requested historical coverage
        has_stale_latest = self._check_prices_staleness(stocks_data, clean_symbols, end_date)
        has_historical_gap = self._check_prices_historical_coverage(
            stocks_data=stocks_data,
            requested_symbols=clean_symbols,
            start_date=start_date
        )
        is_stale = has_stale_latest or has_historical_gap

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

        # Request-path sync is disabled in deterministic sync mode.
        is_syncing = False

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
        if not requested_symbols:
            return False

        if not stocks_data:
            return True

        stale_threshold = end_date - timedelta(days=7)

        for symbol in requested_symbols:
            prices = stocks_data.get(symbol, [])
            if not prices:
                return True

            # Check latest date (data freshness)
            latest_date_str = prices[-1]['date']
            try:
                latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d').date()
            except (TypeError, ValueError):
                return True

            if latest_date < stale_threshold:
                return True

        return False

    def _check_prices_historical_coverage(
        self,
        stocks_data: Dict[str, List[Dict[str, Any]]],
        requested_symbols: List[str],
        start_date: date
    ) -> bool:
        """
        Check whether cached history covers the requested start date.
        Returns True if any requested symbol:
        - Has no data
        - Has an earliest valid data point significantly after start_date
        """
        if not requested_symbols:
            return False

        coverage_threshold = start_date + timedelta(days=BACKFILL_COVERAGE_TOLERANCE_DAYS)

        for symbol in requested_symbols:
            prices = stocks_data.get(symbol, [])
            if not prices:
                return True

            oldest_date: date | None = None
            for point in prices:
                date_str = point.get('date')
                try:
                    point_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                except (TypeError, ValueError):
                    continue

                if oldest_date is None or point_date < oldest_date:
                    oldest_date = point_date

            if oldest_date is None:
                return True

            if oldest_date > coverage_threshold:
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
        end_date: date,
        force: bool = False
    ) -> bool:
        """
        Trigger background sync for price history.
        Returns True if sync was triggered, False if already syncing.
        """
        now = datetime.utcnow()
        cooldown_state = {}
        if not force:
            cooldown_state = await self._get_weekly_sync_cooldown_state(symbols)

        symbols_to_sync = []
        for symbol in symbols:
            if symbol in self._weekly_prices_syncing_symbols:
                continue

            if not force:
                state = cooldown_state.get(symbol, {})
                last_attempt = state.get("last_attempt_at")
                if last_attempt and (now - last_attempt) < self._weekly_prices_retry_cooldown:
                    previous_start_date = state.get("last_attempt_start_date")
                    # Allow immediate retry only when user requests older history than last attempt.
                    if previous_start_date is None or start_date >= previous_start_date:
                        continue

            symbols_to_sync.append(symbol)

        if not symbols_to_sync:
            # Only report syncing=True when there is an active sync task for requested symbols.
            return any(s in self._weekly_prices_syncing_symbols for s in symbols)

        # Mark as syncing
        self._weekly_prices_syncing_symbols.update(symbols_to_sync)
        await self._set_weekly_sync_cooldown_state(
            symbols=symbols_to_sync,
            attempted_at=now,
            attempted_start_date=start_date
        )

        # Create background task
        self._weekly_prices_sync_task = asyncio.create_task(
            self._sync_price_history_background(symbols_to_sync, start_date, end_date)
        )

        return True

    async def _get_weekly_sync_cooldown_state(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict[str, date | datetime | None]]:
        if not symbols:
            return {}

        try:
            async with async_session() as session:
                stmt = select(
                    StockHistoryBackfillState.symbol,
                    StockHistoryBackfillState.weekly_sync_last_attempt_at,
                    StockHistoryBackfillState.weekly_sync_last_attempt_start_date
                ).where(StockHistoryBackfillState.symbol.in_(symbols))
                result = await session.execute(stmt)
                rows = result.all()

            state: Dict[str, Dict[str, date | datetime | None]] = {}
            for symbol, last_attempt_at, last_attempt_start_date in rows:
                state[symbol] = {
                    "last_attempt_at": last_attempt_at,
                    "last_attempt_start_date": last_attempt_start_date,
                }
            return state
        except Exception as e:
            logger.warning(f"Error loading weekly sync cooldown state: {e}")
            return {}

    async def _set_weekly_sync_cooldown_state(
        self,
        symbols: List[str],
        attempted_at: datetime,
        attempted_start_date: date
    ) -> None:
        if not symbols:
            return

        try:
            async with async_session() as session:
                stmt = select(StockHistoryBackfillState).where(
                    StockHistoryBackfillState.symbol.in_(symbols)
                )
                existing_rows = await session.execute(stmt)
                existing = {row.symbol: row for row in existing_rows.scalars().all()}

                for symbol in symbols:
                    state = existing.get(symbol)
                    if state is None:
                        state = StockHistoryBackfillState(symbol=symbol)
                        session.add(state)
                        existing[symbol] = state

                    state.weekly_sync_last_attempt_at = attempted_at
                    state.weekly_sync_last_attempt_start_date = attempted_start_date

                await session.commit()
        except Exception as e:
            logger.warning(f"Error updating weekly sync cooldown state: {e}")

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
