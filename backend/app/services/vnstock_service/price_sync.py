from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pandas as pd
from sqlalchemy import func, select

from app.core.config import settings
from app.db.database import async_session
from app.db.models import StockCompany, StockDailyPrice, StockPriceSyncState
from app.services.sync_status import sync_status

from .core import (
    background_executor,
    bootstrap_logger,
    api_circuit_breaker,
    CircuitOpenError,
    _is_rate_limit_error,
)
from .history import HistoryService


@dataclass(frozen=True)
class SymbolBootstrapMeta:
    symbol: str
    listing_date: date | None


class PriceSyncService:
    """Deterministic price history sync orchestration.

    Workflow:
    - One-time bootstrap for all symbols (resumable via DB checkpoints)
    - Incremental sync with small heal window
    - Manual repair for specific symbols/date ranges
    """

    BOOTSTRAP_CHUNK_DAYS = 1095
    BOOTSTRAP_TARGET_RPM = 150
    BOOTSTRAP_MAX_CONCURRENCY = 6
    BOOTSTRAP_RATE_LIMIT_MAX_RETRIES = 12
    BOOTSTRAP_RETRY_BASE_DELAY_SECONDS = 5.0
    BOOTSTRAP_RETRY_MAX_DELAY_SECONDS = 60.0
    INCREMENTAL_HEAL_WINDOW_DAYS = 7
    FALLBACK_DISCOVERY_START_DATE = date(1990, 1, 1)

    def __init__(self, history: HistoryService) -> None:
        self._history = history
        self._bootstrap_task: asyncio.Task | None = None
        self._bootstrap_lock = asyncio.Lock()
        self._bootstrap_pacer_lock = asyncio.Lock()
        self._bootstrap_last_request_monotonic: float | None = None
        self._bootstrap_chunk_days = max(1, int(settings.price_bootstrap_chunk_days))
        self._bootstrap_target_rpm = max(1, int(settings.price_bootstrap_target_rpm))
        self._bootstrap_max_concurrency = max(1, int(settings.price_bootstrap_max_concurrency))
        self._bootstrap_executor_workers = max(1, int(settings.price_bootstrap_executor_workers))
        self._bootstrap_rate_limit_max_retries = max(0, int(settings.price_bootstrap_rate_limit_max_retries))
        self._bootstrap_retry_base_delay_seconds = max(
            0.1,
            float(settings.price_bootstrap_retry_base_delay_seconds),
        )
        self._bootstrap_retry_max_delay_seconds = max(
            self._bootstrap_retry_base_delay_seconds,
            float(settings.price_bootstrap_retry_max_delay_seconds),
        )
        self._bootstrap_executor: ThreadPoolExecutor | None = None

    async def start_background_tasks(self) -> None:
        """No-op hook kept for lifecycle symmetry."""
        return

    async def stop_background_tasks(self) -> None:
        """Cancel running bootstrap task on shutdown."""
        async with self._bootstrap_lock:
            if self._bootstrap_task and not self._bootstrap_task.done():
                self._bootstrap_task.cancel()
                try:
                    await self._bootstrap_task
                except asyncio.CancelledError:
                    pass
            self._bootstrap_task = None
            if self._bootstrap_executor is not None:
                self._bootstrap_executor.shutdown(wait=False, cancel_futures=True)
                self._bootstrap_executor = None

    async def start_bootstrap(self, force_restart: bool = False) -> Dict[str, Any]:
        """Start one-time bootstrap as background task."""
        async with self._bootstrap_lock:
            self._ensure_bootstrap_executor()
            if self._bootstrap_task and not self._bootstrap_task.done():
                if not force_restart:
                    return {
                        "started": False,
                        "message": "Bootstrap is already running",
                        "state": sync_status.price_bootstrap.state,
                    }

                self._bootstrap_task.cancel()
                try:
                    await self._bootstrap_task
                except asyncio.CancelledError:
                    pass

            self._bootstrap_task = asyncio.create_task(self._run_bootstrap())

        return {
            "started": True,
            "message": "Bootstrap started",
            "state": "running",
        }

    async def get_bootstrap_status(self) -> Dict[str, Any]:
        runtime = sync_status.price_bootstrap
        db_summary = await self._get_bootstrap_db_summary()

        return {
            "state": runtime.state,
            "total_symbols": runtime.total_symbols,
            "processed_symbols": runtime.processed_symbols,
            "success_symbols": runtime.success_symbols,
            "failed_symbols": runtime.failed_symbols,
            "current_symbol": runtime.current_symbol,
            "started_at": runtime.started_at,
            "completed_at": runtime.completed_at,
            "error": runtime.error,
            "progress": runtime.progress,
            "db_summary": db_summary,
        }

    async def run_incremental_sync(self, heal_window_days: int | None = None) -> Dict[str, Any]:
        """Run incremental sync for all active symbols."""
        if sync_status.price_bootstrap.state == "running":
            return {
                "started": False,
                "message": "Bootstrap is running. Incremental sync skipped.",
                "processed_symbols": 0,
                "success_symbols": 0,
                "failed_symbols": 0,
            }

        symbols = await self._get_active_symbols_for_incremental()
        total_symbols = len(symbols)
        window_days = heal_window_days or self.INCREMENTAL_HEAL_WINDOW_DAYS

        sync_status.start_price_incremental(total_symbols=total_symbols)

        today = date.today()
        start_date = today - timedelta(days=max(1, window_days))

        success_count = 0
        failure_count = 0

        try:
            loop = asyncio.get_event_loop()
            for idx, symbol in enumerate(symbols, start=1):
                sync_status.update_price_incremental_progress(
                    processed_symbols=idx - 1,
                    current_symbol=symbol,
                )

                try:
                    await loop.run_in_executor(
                        background_executor,
                        self._history._upsert_stock_price_history,
                        symbol,
                        start_date,
                        today,
                    )
                    await self._mark_incremental_result(symbol)
                    success_count += 1
                except Exception as e:
                    failure_count += 1
                    await self._mark_symbol_error(symbol, f"Incremental sync failed: {e}")

                sync_status.update_price_incremental_progress(
                    processed_symbols=idx,
                    current_symbol=symbol,
                )

            sync_status.complete_price_incremental(success=True)
            return {
                "started": True,
                "message": "Incremental sync completed",
                "processed_symbols": total_symbols,
                "success_symbols": success_count,
                "failed_symbols": failure_count,
                "window_start_date": start_date.isoformat(),
                "window_end_date": today.isoformat(),
            }
        except Exception as e:
            sync_status.complete_price_incremental(success=False, error=str(e)[:500])
            return {
                "started": False,
                "message": f"Incremental sync failed: {e}",
                "processed_symbols": success_count + failure_count,
                "success_symbols": success_count,
                "failed_symbols": failure_count,
            }

    async def run_repair_sync(self, symbols: List[str], start_date: date, end_date: date) -> Dict[str, Any]:
        """Run manual repair sync for selected symbols/date range."""
        clean_symbols = list(dict.fromkeys(self._normalize_symbol(s) for s in symbols if s))
        total_symbols = len(clean_symbols)

        if total_symbols == 0:
            return {
                "started": False,
                "message": "No symbols provided",
                "processed_symbols": 0,
                "success_symbols": 0,
                "failed_symbols": 0,
            }

        sync_status.start_price_repair(total_symbols=total_symbols)

        success_count = 0
        failure_count = 0

        try:
            loop = asyncio.get_event_loop()
            for idx, symbol in enumerate(clean_symbols, start=1):
                sync_status.update_price_repair_progress(
                    processed_symbols=idx - 1,
                    current_symbol=symbol,
                )

                try:
                    await loop.run_in_executor(
                        background_executor,
                        self._history._upsert_stock_price_history,
                        symbol,
                        start_date,
                        end_date,
                    )
                    await self._mark_incremental_result(symbol)
                    success_count += 1
                except Exception as e:
                    failure_count += 1
                    await self._mark_symbol_error(symbol, f"Repair sync failed: {e}")

                sync_status.update_price_repair_progress(
                    processed_symbols=idx,
                    current_symbol=symbol,
                )

            sync_status.complete_price_repair(success=True)
            return {
                "started": True,
                "message": "Repair sync completed",
                "processed_symbols": total_symbols,
                "success_symbols": success_count,
                "failed_symbols": failure_count,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }
        except Exception as e:
            sync_status.complete_price_repair(success=False, error=str(e)[:500])
            return {
                "started": False,
                "message": f"Repair sync failed: {e}",
                "processed_symbols": success_count + failure_count,
                "success_symbols": success_count,
                "failed_symbols": failure_count,
            }

    async def _run_bootstrap(self) -> None:
        """Background bootstrap execution."""
        symbols_meta = await self._build_symbol_universe()
        total = len(symbols_meta)

        sync_status.start_price_bootstrap(total_symbols=total)
        if total == 0:
            sync_status.complete_price_bootstrap(success=True)
            return

        await self._ensure_sync_state_rows(symbols_meta)

        await self._reset_bootstrap_pacer()
        bootstrap_logger.info(
            "Bootstrap runtime config: max_concurrency=%s executor_workers=%s target_rpm=%s chunk_days=%s",
            self._bootstrap_max_concurrency,
            self._bootstrap_executor_workers,
            self._bootstrap_target_rpm,
            self._bootstrap_chunk_days,
        )

        work_queue: asyncio.Queue[SymbolBootstrapMeta] = asyncio.Queue()
        for meta in symbols_meta:
            work_queue.put_nowait(meta)

        worker_count = min(total, self._bootstrap_max_concurrency)
        success_count = 0
        failure_count = 0
        processed_count = 0
        progress_lock = asyncio.Lock()

        async def _bootstrap_worker(worker_id: int) -> None:
            nonlocal processed_count, success_count, failure_count
            while True:
                try:
                    meta = work_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                symbol_failed = False
                was_cancelled = False
                try:
                    async with progress_lock:
                        sync_status.update_price_bootstrap_progress(
                            processed_symbols=processed_count,
                            success_symbols=success_count,
                            failed_symbols=failure_count,
                            current_symbol=meta.symbol,
                        )
                    await self._bootstrap_symbol(meta)
                except asyncio.CancelledError:
                    was_cancelled = True
                    raise
                except Exception as e:
                    symbol_failed = True
                    await self._mark_symbol_failed(meta.symbol, str(e))
                    bootstrap_logger.error(
                        f"Bootstrap worker {worker_id} failed symbol {meta.symbol}: {e}"
                    )
                finally:
                    if not was_cancelled:
                        async with progress_lock:
                            processed_count += 1
                            if symbol_failed:
                                failure_count += 1
                            else:
                                success_count += 1

                            sync_status.update_price_bootstrap_progress(
                                processed_symbols=processed_count,
                                success_symbols=success_count,
                                failed_symbols=failure_count,
                                current_symbol=meta.symbol,
                            )
                    work_queue.task_done()

        workers = [
            asyncio.create_task(_bootstrap_worker(worker_idx))
            for worker_idx in range(1, worker_count + 1)
        ]

        try:
            await asyncio.gather(*workers)
            if failure_count > 0:
                sync_status.complete_price_bootstrap(
                    success=True,
                    error=f"Bootstrap completed with {failure_count} failed symbols",
                )
            else:
                sync_status.complete_price_bootstrap(success=True)

        except asyncio.CancelledError:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            sync_status.complete_price_bootstrap(success=False, error="Bootstrap cancelled")
            raise
        except Exception as e:
            sync_status.complete_price_bootstrap(success=False, error=str(e)[:500])

    async def _bootstrap_symbol(self, meta: SymbolBootstrapMeta) -> None:
        started = time.monotonic()
        chunk_count = 0
        retry_count = 0

        state = await self._get_or_create_symbol_state(meta.symbol, meta.listing_date)

        if (
            state.bootstrap_status == "completed"
            and state.latest_synced_date is not None
            and state.earliest_synced_date is not None
        ):
            return

        symbol_start_date = state.listing_date or meta.listing_date
        if symbol_start_date is None:
            loop = asyncio.get_event_loop()
            bootstrap_executor = self._ensure_bootstrap_executor()
            symbol_start_date = await loop.run_in_executor(
                bootstrap_executor,
                self._discover_oldest_history_date,
                meta.symbol,
            )

        if symbol_start_date is None:
            raise RuntimeError("Unable to determine symbol start date")

        await self._set_symbol_bootstrap_running(meta.symbol, symbol_start_date)

        today = date.today()
        cursor = symbol_start_date
        if state.latest_synced_date and state.latest_synced_date >= symbol_start_date:
            cursor = state.latest_synced_date + timedelta(days=1)

        while cursor <= today:
            chunk_end = min(cursor + timedelta(days=self._bootstrap_chunk_days - 1), today)
            retries = await self._run_bootstrap_chunk_with_retry(
                meta.symbol,
                cursor,
                chunk_end,
            )
            retry_count += retries
            chunk_count += 1
            cursor = chunk_end + timedelta(days=1)

        oldest, latest = await self._get_symbol_bounds(meta.symbol)
        if oldest is None or latest is None:
            raise RuntimeError("No price history returned by source")

        await self._mark_symbol_bootstrap_completed(
            symbol=meta.symbol,
            listing_date=symbol_start_date,
            earliest_date=oldest,
            latest_date=latest,
        )

        elapsed = time.monotonic() - started
        bootstrap_logger.info(
            f"Bootstrap symbol {meta.symbol} completed: chunks={chunk_count}, "
            f"retries={retry_count}, elapsed={elapsed:.2f}s"
        )

    def _ensure_bootstrap_executor(self) -> ThreadPoolExecutor:
        if self._bootstrap_executor is None:
            self._bootstrap_executor = ThreadPoolExecutor(
                max_workers=self._bootstrap_executor_workers,
                thread_name_prefix="bootstrap_sync",
            )
        return self._bootstrap_executor

    async def _reset_bootstrap_pacer(self) -> None:
        async with self._bootstrap_pacer_lock:
            self._bootstrap_last_request_monotonic = None

    async def _acquire_bootstrap_request_slot(self) -> None:
        interval_seconds = 60.0 / float(self._bootstrap_target_rpm)
        async with self._bootstrap_pacer_lock:
            now = time.monotonic()
            if self._bootstrap_last_request_monotonic is not None:
                elapsed = now - self._bootstrap_last_request_monotonic
                if elapsed < interval_seconds:
                    await asyncio.sleep(interval_seconds - elapsed)
                    now = time.monotonic()
            self._bootstrap_last_request_monotonic = now

    async def _execute_bootstrap_chunk_upsert(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> int:
        loop = asyncio.get_event_loop()
        bootstrap_executor = self._ensure_bootstrap_executor()
        upsert_call = partial(
            self._history._upsert_stock_price_history,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            raise_on_error=True,
            log=bootstrap_logger,
        )
        return await loop.run_in_executor(bootstrap_executor, upsert_call)

    def _compute_bootstrap_retry_delay(self, retry_number: int) -> float:
        delay = self._bootstrap_retry_base_delay_seconds * (2 ** max(0, retry_number - 1))
        return min(delay, self._bootstrap_retry_max_delay_seconds)

    async def _run_bootstrap_chunk_with_retry(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> int:
        attempt = 0
        while True:
            attempt += 1
            retries_so_far = attempt - 1
            await self._acquire_bootstrap_request_slot()
            started = time.monotonic()

            try:
                await self._execute_bootstrap_chunk_upsert(symbol, start_date, end_date)
                elapsed = time.monotonic() - started
                bootstrap_logger.debug(
                    f"Bootstrap chunk synced for {symbol} ({start_date} -> {end_date}) "
                    f"attempt={attempt} elapsed={elapsed:.2f}s"
                )
                return retries_so_far
            except Exception as e:
                elapsed = time.monotonic() - started
                if not (_is_rate_limit_error(e) or isinstance(e, CircuitOpenError)):
                    bootstrap_logger.debug(
                        f"Bootstrap chunk failed for {symbol} ({start_date} -> {end_date}) "
                        f"attempt={attempt} elapsed={elapsed:.2f}s error={e}"
                    )
                    raise

                if retries_so_far >= self._bootstrap_rate_limit_max_retries:
                    raise RuntimeError(
                        f"Rate limit persisted for {symbol} ({start_date} -> {end_date}) "
                        f"after {attempt} attempts"
                    ) from e

                backoff_delay = self._compute_bootstrap_retry_delay(retries_so_far + 1)
                status_delay = sync_status.rate_limit_seconds_remaining or 0.0
                circuit_delay = api_circuit_breaker.time_until_half_open or 0.0
                wait_base = max(backoff_delay, status_delay, circuit_delay)
                jitter = random.uniform(0.0, min(1.0, wait_base * 0.2))
                sleep_seconds = wait_base + jitter

                bootstrap_logger.debug(
                    f"Bootstrap chunk rate-limited for {symbol} ({start_date} -> {end_date}) "
                    f"attempt={attempt} elapsed={elapsed:.2f}s sleep={sleep_seconds:.2f}s"
                )
                await asyncio.sleep(sleep_seconds)

    async def _build_symbol_universe(self) -> List[SymbolBootstrapMeta]:
        listing_map = await self._fetch_listing_symbol_map()
        db_symbols = await self._fetch_db_symbols()

        all_symbols = set(listing_map.keys()) | set(db_symbols)
        sorted_symbols = sorted(all_symbols)

        return [
            SymbolBootstrapMeta(symbol=symbol, listing_date=listing_map.get(symbol))
            for symbol in sorted_symbols
        ]

    async def _fetch_listing_symbol_map(self) -> Dict[str, date | None]:
        loop = asyncio.get_event_loop()
        bootstrap_executor = self._ensure_bootstrap_executor()
        return await loop.run_in_executor(bootstrap_executor, self._fetch_listing_symbol_map_sync)

    def _fetch_listing_symbol_map_sync(self) -> Dict[str, date | None]:
        from vnstock import Listing

        result: Dict[str, date | None] = {}
        try:
            df = Listing(source='VCI').all_symbols()
            if df is None or df.empty:
                return result

            for _, row in df.iterrows():
                symbol = self._normalize_symbol(row.get('symbol'))
                if not symbol:
                    continue
                listing_date = self._extract_listing_date_from_row(row)
                result[symbol] = listing_date

        except Exception as e:
            bootstrap_logger.warning(f"Error fetching listing symbols for bootstrap: {e}")

        return result

    def _extract_listing_date_from_row(self, row) -> Optional[date]:
        candidates = [
            'listing_date',
            'listed_date',
            'listing_first_trade_date',
            'first_trade_date',
            'ipo_date',
            'trading_date',
        ]

        for column in candidates:
            if column not in row.index:
                continue
            raw_value = row.get(column)
            if pd.isna(raw_value):
                continue
            parsed = pd.to_datetime(raw_value, errors='coerce')
            if pd.isna(parsed):
                continue
            parsed_date = parsed.date()
            if parsed_date.year >= 1990:
                return parsed_date

        return None

    def _discover_oldest_history_date(self, symbol: str) -> Optional[date]:
        from vnstock import Vnstock

        try:
            stock = Vnstock().stock(symbol=symbol, source='VCI')
            hist = stock.quote.history(
                start=self.FALLBACK_DISCOVERY_START_DATE.strftime('%Y-%m-%d'),
                end=date.today().strftime('%Y-%m-%d'),
                interval='1D',
            )
            if hist is None or hist.empty:
                return None

            times = pd.to_datetime(hist['time'], errors='coerce').dropna()
            if times.empty:
                return None
            return times.min().date()
        except Exception as e:
            bootstrap_logger.warning(f"Error discovering oldest history date for {symbol}: {e}")
            return None

    async def _fetch_db_symbols(self) -> List[str]:
        async with async_session() as session:
            company_stmt = select(StockCompany.symbol)
            company_result = await session.execute(company_stmt)
            company_symbols = [self._normalize_symbol(row[0]) for row in company_result.all() if row[0]]

            state_stmt = select(StockPriceSyncState.symbol)
            state_result = await session.execute(state_stmt)
            state_symbols = [self._normalize_symbol(row[0]) for row in state_result.all() if row[0]]

        return list(dict.fromkeys([*company_symbols, *state_symbols]))

    async def _ensure_sync_state_rows(self, symbols_meta: List[SymbolBootstrapMeta]) -> None:
        if not symbols_meta:
            return

        symbols = [meta.symbol for meta in symbols_meta]
        listing_map = {meta.symbol: meta.listing_date for meta in symbols_meta}

        async with async_session() as session:
            stmt = select(StockPriceSyncState).where(StockPriceSyncState.symbol.in_(symbols))
            existing_rows = await session.execute(stmt)
            existing = {row.symbol: row for row in existing_rows.scalars().all()}

            now = datetime.utcnow()
            for symbol in symbols:
                row = existing.get(symbol)
                listing_date = listing_map.get(symbol)
                if row is None:
                    session.add(
                        StockPriceSyncState(
                            symbol=symbol,
                            listing_date=listing_date,
                            bootstrap_status='idle',
                            updated_at=now,
                        )
                    )
                    continue

                if row.listing_date is None and listing_date is not None:
                    row.listing_date = listing_date
                row.updated_at = now

            await session.commit()

    async def _get_or_create_symbol_state(self, symbol: str, listing_date: date | None) -> StockPriceSyncState:
        async with async_session() as session:
            stmt = select(StockPriceSyncState).where(StockPriceSyncState.symbol == symbol)
            row = (await session.execute(stmt)).scalar_one_or_none()

            if row is None:
                row = StockPriceSyncState(
                    symbol=symbol,
                    listing_date=listing_date,
                    bootstrap_status='idle',
                    retry_count=0,
                )
                session.add(row)
                await session.commit()
                await session.refresh(row)
                return row

            if row.listing_date is None and listing_date is not None:
                row.listing_date = listing_date
                row.updated_at = datetime.utcnow()
                await session.commit()
                await session.refresh(row)

            return row

    async def _set_symbol_bootstrap_running(self, symbol: str, listing_date: date) -> None:
        now = datetime.utcnow()
        async with async_session() as session:
            stmt = select(StockPriceSyncState).where(StockPriceSyncState.symbol == symbol)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                row = StockPriceSyncState(symbol=symbol)
                session.add(row)

            row.listing_date = row.listing_date or listing_date
            row.bootstrap_status = 'running'
            row.bootstrap_started_at = now
            row.last_error = None
            row.updated_at = now
            await session.commit()

    async def _mark_symbol_bootstrap_completed(
        self,
        symbol: str,
        listing_date: date,
        earliest_date: date,
        latest_date: date,
    ) -> None:
        now = datetime.utcnow()
        async with async_session() as session:
            stmt = select(StockPriceSyncState).where(StockPriceSyncState.symbol == symbol)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                row = StockPriceSyncState(symbol=symbol)
                session.add(row)

            row.listing_date = row.listing_date or listing_date
            row.bootstrap_status = 'completed'
            row.bootstrap_completed_at = now
            row.earliest_synced_date = earliest_date
            row.latest_synced_date = latest_date
            row.last_error = None
            row.updated_at = now
            await session.commit()

    async def _mark_symbol_failed(self, symbol: str, error_message: str) -> None:
        now = datetime.utcnow()
        async with async_session() as session:
            stmt = select(StockPriceSyncState).where(StockPriceSyncState.symbol == symbol)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                row = StockPriceSyncState(symbol=symbol)
                session.add(row)

            row.bootstrap_status = 'failed'
            row.last_error = error_message[:500]
            row.retry_count = (row.retry_count or 0) + 1
            row.updated_at = now
            await session.commit()

    async def _mark_symbol_error(self, symbol: str, error_message: str) -> None:
        now = datetime.utcnow()
        async with async_session() as session:
            stmt = select(StockPriceSyncState).where(StockPriceSyncState.symbol == symbol)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                row = StockPriceSyncState(symbol=symbol)
                session.add(row)

            row.last_error = error_message[:500]
            row.retry_count = (row.retry_count or 0) + 1
            row.updated_at = now
            await session.commit()

    async def _mark_incremental_result(self, symbol: str) -> None:
        now = datetime.utcnow()
        oldest, latest = await self._get_symbol_bounds(symbol)

        async with async_session() as session:
            stmt = select(StockPriceSyncState).where(StockPriceSyncState.symbol == symbol)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                row = StockPriceSyncState(symbol=symbol)
                session.add(row)

            row.last_incremental_sync_at = now
            if oldest is not None:
                row.earliest_synced_date = oldest
            if latest is not None:
                row.latest_synced_date = latest
            row.last_error = None
            row.updated_at = now
            await session.commit()

    async def _get_symbol_bounds(self, symbol: str) -> tuple[date | None, date | None]:
        async with async_session() as session:
            stmt = select(
                func.min(StockDailyPrice.date),
                func.max(StockDailyPrice.date),
            ).where(StockDailyPrice.symbol == symbol)
            result = await session.execute(stmt)
            row = result.one()
            return row[0], row[1]

    async def _get_active_symbols_for_incremental(self) -> List[str]:
        async with async_session() as session:
            stmt = select(StockPriceSyncState.symbol).where(
                StockPriceSyncState.bootstrap_status.in_(['completed', 'failed', 'idle', 'running'])
            )
            result = await session.execute(stmt)
            symbols = [self._normalize_symbol(row[0]) for row in result.all() if row[0]]

        if symbols:
            return list(dict.fromkeys(symbols))

        # Fallback before any bootstrap state exists
        return await self._fetch_db_symbols()

    async def _get_bootstrap_db_summary(self) -> Dict[str, Any]:
        async with async_session() as session:
            total_stmt = select(func.count(StockPriceSyncState.id))
            total = (await session.execute(total_stmt)).scalar() or 0

            grouped_stmt = (
                select(StockPriceSyncState.bootstrap_status, func.count(StockPriceSyncState.id))
                .group_by(StockPriceSyncState.bootstrap_status)
            )
            grouped = await session.execute(grouped_stmt)

            by_status: Dict[str, int] = {}
            for status_value, count_value in grouped.all():
                by_status[str(status_value)] = int(count_value)

        return {
            "total_symbols": int(total),
            "by_status": by_status,
        }

    def _normalize_symbol(self, symbol: Any) -> str:
        if symbol is None:
            return ""
        return str(symbol).strip().upper()[:3]
