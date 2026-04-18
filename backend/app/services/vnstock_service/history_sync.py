from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pandas as pd
from sqlalchemy import func, select

from app.core.config import settings
from app.db.database import async_session
from app.db.models import StockCompany, StockDailyHistory, StockHistorySyncState
from app.services.sync_status import sync_status

from .core import (
    history_sync_logger,
    api_circuit_breaker,
    CircuitOpenError,
    _is_rate_limit_error,
)
from .history import HistoryService
from .rate_limit_pause import shared_rate_limit_pause_controller
from .symbols import VALID_GROUPS, get_group_code_for_index


@dataclass(frozen=True)
class SymbolSyncMeta:
    symbol: str
    listing_date: date | None


@dataclass(frozen=True)
class RequestHistorySyncResult:
    sync_performed: bool = False
    sync_timed_out: bool = False
    sync_error: str | None = None
    updated_through: str | None = None
    repaired_missing_dates: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sync_performed": self.sync_performed,
            "sync_timed_out": self.sync_timed_out,
            "sync_error": self.sync_error,
            "updated_through": self.updated_through,
            "repaired_missing_dates": self.repaired_missing_dates,
        }


class HistorySyncService:
    """Deterministic stock history synchronization and gap-audit orchestration."""

    SYNC_OVERLAP_DAYS = 2
    FALLBACK_DISCOVERY_START_DATE = date(1990, 1, 1)

    def __init__(self, history: HistoryService) -> None:
        self._history = history
        self._sync_task: asyncio.Task | None = None
        self._sync_lock = asyncio.Lock()
        self._sync_pacer_lock = asyncio.Lock()
        self._sync_last_request_monotonic: float | None = None
        self._request_sync_tasks: Dict[str, asyncio.Task[RequestHistorySyncResult]] = {}
        self._request_sync_tasks_lock = asyncio.Lock()
        self._request_sync_timeout_default_seconds = 12.0

        self._sync_chunk_days = max(1, int(settings.sync_chunk_days))
        self._sync_target_rpm = max(1, int(settings.sync_target_rpm))
        self._sync_max_workers = max(1, int(settings.sync_max_workers))
        self._sync_rate_limit_fixed_wait_seconds = max(
            0.1,
            float(settings.sync_rate_limit_fixed_wait_seconds),
        )
        self._sync_rate_limit_max_wait_seconds = max(
            0.0,
            float(settings.sync_rate_limit_max_wait_seconds),
        )
        self._sync_executor: ThreadPoolExecutor | None = None
        self._operation_worker_semaphore = asyncio.Semaphore(self._sync_max_workers)

    async def start_background_tasks(self) -> None:
        """No-op hook kept for lifecycle symmetry."""
        return

    async def stop_background_tasks(self) -> None:
        """Cancel running sync task on shutdown."""
        request_tasks_to_cancel: List[asyncio.Task[RequestHistorySyncResult]] = []
        async with self._request_sync_tasks_lock:
            request_tasks_to_cancel = [
                task for task in self._request_sync_tasks.values()
                if not task.done()
            ]
            self._request_sync_tasks.clear()
        for task in request_tasks_to_cancel:
            task.cancel()
        if request_tasks_to_cancel:
            await asyncio.gather(*request_tasks_to_cancel, return_exceptions=True)

        async with self._sync_lock:
            if self._sync_task and not self._sync_task.done():
                self._sync_task.cancel()
                try:
                    await self._sync_task
                except asyncio.CancelledError:
                    pass
            self._sync_task = None
            if self._sync_executor is not None:
                self._sync_executor.shutdown(wait=False, cancel_futures=True)
                self._sync_executor = None

    async def run_sync(
        self,
        force_restart: bool = False,
        symbols: Optional[List[str]] = None,
        index_symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start unified history sync as background task."""
        symbols_filter = await self._resolve_symbols_filter(
            symbols=symbols,
            index_symbol=index_symbol,
        )

        async with self._sync_lock:
            self._ensure_sync_executor()
            if self._sync_task and not self._sync_task.done():
                if not force_restart:
                    return {
                        "started": False,
                        "message": "History sync is already running",
                        "state": "running",
                    }

                self._sync_task.cancel()
                try:
                    await self._sync_task
                except asyncio.CancelledError:
                    pass

            self._sync_task = asyncio.create_task(self._run_sync(symbols_filter))

        return {
            "started": True,
            "message": "History sync started",
            "state": "running",
        }

    async def sync_symbol_history_for_request(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        timeout_seconds: float | None = None,
    ) -> Dict[str, Any]:
        symbol_clean = self._normalize_symbol(symbol)
        if not symbol_clean:
            return RequestHistorySyncResult(
                sync_error="Invalid symbol",
            ).to_dict()

        safe_timeout = self._request_sync_timeout_default_seconds
        if timeout_seconds is not None:
            try:
                safe_timeout = float(timeout_seconds)
            except (TypeError, ValueError):
                safe_timeout = self._request_sync_timeout_default_seconds
            if safe_timeout <= 0:
                safe_timeout = self._request_sync_timeout_default_seconds

        task = await self._get_or_create_request_sync_task(
            symbol=symbol_clean,
            start_date=start_date,
            end_date=end_date,
        )
        try:
            result = await asyncio.wait_for(asyncio.shield(task), timeout=safe_timeout)
            return result.to_dict()
        except asyncio.TimeoutError:
            return RequestHistorySyncResult(
                sync_timed_out=True,
                updated_through=await self._get_symbol_latest_date_iso(symbol_clean),
            ).to_dict()
        except Exception as e:
            message = str(e)[:500]
            history_sync_logger.warning(
                "Request-path history sync failed for %s (%s -> %s): %s",
                symbol_clean,
                start_date,
                end_date,
                message,
            )
            return RequestHistorySyncResult(
                sync_error=message,
                updated_through=await self._get_symbol_latest_date_iso(symbol_clean),
            ).to_dict()

    async def _get_or_create_request_sync_task(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> asyncio.Task[RequestHistorySyncResult]:
        async with self._request_sync_tasks_lock:
            existing_task = self._request_sync_tasks.get(symbol)
            if existing_task and not existing_task.done():
                return existing_task

            task = asyncio.create_task(
                self._sync_symbol_history_for_request_task(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
            self._request_sync_tasks[symbol] = task

            def _cleanup(done_task: asyncio.Task[RequestHistorySyncResult]) -> None:
                current_task = self._request_sync_tasks.get(symbol)
                if current_task is done_task:
                    self._request_sync_tasks.pop(symbol, None)

            task.add_done_callback(_cleanup)
            return task

    async def _sync_symbol_history_for_request_task(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> RequestHistorySyncResult:
        try:
            if end_date < start_date:
                start_date, end_date = end_date, start_date

            today = date.today()
            bounded_end = min(end_date, today)
            bounded_start = min(start_date, bounded_end)

            state = await self._get_or_create_symbol_state(symbol, listing_date=None)
            symbol_start_date = state.listing_date
            _, latest_local = await self._get_symbol_bounds(symbol)

            if latest_local is None:
                if symbol_start_date is None:
                    loop = asyncio.get_event_loop()
                    sync_executor = self._ensure_sync_executor()
                    symbol_start_date = await loop.run_in_executor(
                        sync_executor,
                        self._discover_oldest_history_date,
                        symbol,
                    )
                if symbol_start_date is None:
                    raise RuntimeError("Unable to determine symbol start date")
                incremental_start = symbol_start_date
            else:
                overlap_days = max(0, self.SYNC_OVERLAP_DAYS - 1)
                incremental_start = latest_local - timedelta(days=overlap_days)
                if symbol_start_date is not None:
                    incremental_start = max(symbol_start_date, incremental_start)

            await self._set_symbol_sync_running(
                symbol=symbol,
                listing_date=symbol_start_date or bounded_start,
            )

            cursor = incremental_start
            while cursor <= today:
                chunk_end = min(cursor + timedelta(days=self._sync_chunk_days - 1), today)
                await self._run_sync_chunk_with_retry(symbol, cursor, chunk_end)
                cursor = chunk_end + timedelta(days=1)

            local_dates = await self._get_local_history_dates(symbol, bounded_start, bounded_end)
            upstream_dates = await self._fetch_remote_history_dates(symbol, bounded_start, bounded_end)
            missing_dates = sorted(upstream_dates - local_dates)

            repaired_missing_dates = 0
            if missing_dates:
                repaired_missing_dates = await self._repair_missing_dates(symbol, missing_dates)
            else:
                await self._mark_symbol_sync_result(symbol)

            return RequestHistorySyncResult(
                sync_performed=True,
                updated_through=await self._get_symbol_latest_date_iso(symbol),
                repaired_missing_dates=repaired_missing_dates,
            )
        except Exception as e:
            message = str(e)[:500]
            await self._mark_symbol_error(symbol, f"Request sync failed: {message}")
            history_sync_logger.warning(
                "Request-path history sync task failed for %s (%s -> %s): %s",
                symbol,
                start_date,
                end_date,
                message,
            )
            return RequestHistorySyncResult(
                sync_error=message,
                updated_through=await self._get_symbol_latest_date_iso(symbol),
            )

    async def run_audit_sync(
        self,
        symbols: Optional[List[str]],
        start_date: date,
        end_date: date,
        auto_repair: bool = False,
        index_symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run on-demand gap audit against upstream history and optionally repair gaps."""
        symbols_filter = await self._resolve_symbols_filter(
            symbols=symbols,
            index_symbol=index_symbol,
        )
        if symbols_filter:
            symbols_to_audit = symbols_filter
        else:
            symbols_meta = await self._build_symbol_universe()
            symbols_to_audit = [meta.symbol for meta in symbols_meta]

        total_symbols = len(symbols_to_audit)
        if total_symbols == 0:
            return {
                "started": False,
                "message": "No symbols available for audit",
                "processed_symbols": 0,
                "success_symbols": 0,
                "failed_symbols": 0,
                "audited_symbols": 0,
                "symbols_with_gaps": 0,
                "total_missing_dates": 0,
                "total_repaired_dates": 0,
                "results": [],
            }

        sync_status.start_history_audit(total_symbols=total_symbols)

        processed_symbols = 0
        success_symbols = 0
        failed_symbols = 0
        failed_tickers: List[str] = []
        symbols_with_gaps = 0
        total_missing_dates = 0
        total_repaired_dates = 0
        results: List[Optional[Dict[str, Any]]] = [None] * total_symbols
        progress_lock = asyncio.Lock()

        work_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
        for index, symbol in enumerate(symbols_to_audit):
            work_queue.put_nowait((index, symbol))

        worker_count = min(total_symbols, self._sync_max_workers)

        async def _audit_worker() -> None:
            nonlocal processed_symbols, success_symbols, failed_symbols
            nonlocal symbols_with_gaps, total_missing_dates, total_repaired_dates

            while True:
                try:
                    result_index, symbol = work_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                symbol_result: Dict[str, Any]
                symbol_failed = False
                symbol_missing_count = 0
                symbol_repaired_count = 0

                try:
                    async with self._operation_worker_semaphore:
                        async with progress_lock:
                            sync_status.update_history_audit_progress(
                                processed_symbols=processed_symbols,
                                success_symbols=success_symbols,
                                failed_symbols=failed_symbols,
                                current_symbol=symbol,
                            )

                        local_dates = await self._get_local_history_dates(symbol, start_date, end_date)
                        upstream_dates = await self._fetch_remote_history_dates(symbol, start_date, end_date)

                        missing_dates = sorted(upstream_dates - local_dates)
                        symbol_missing_count = len(missing_dates)

                        if symbol_missing_count > 0 and auto_repair:
                            symbol_repaired_count = await self._repair_missing_dates(symbol, missing_dates)

                        symbol_result = {
                            "symbol": symbol,
                            "local_dates": len(local_dates),
                            "upstream_dates": len(upstream_dates),
                            "missing_dates": symbol_missing_count,
                            "repaired_dates": symbol_repaired_count,
                            "missing_date_samples": [d.isoformat() for d in missing_dates[:20]],
                            "error": None,
                        }
                except Exception as e:
                    symbol_failed = True
                    await self._mark_symbol_error(symbol, f"Audit sync failed: {e}")
                    symbol_result = {
                        "symbol": symbol,
                        "local_dates": 0,
                        "upstream_dates": 0,
                        "missing_dates": 0,
                        "repaired_dates": 0,
                        "missing_date_samples": [],
                        "error": str(e)[:500],
                    }
                finally:
                    async with progress_lock:
                        processed_symbols += 1

                        if symbol_failed:
                            failed_symbols += 1
                            failed_tickers.append(symbol)
                        else:
                            success_symbols += 1
                            if symbol_missing_count > 0:
                                symbols_with_gaps += 1
                                total_missing_dates += symbol_missing_count
                                total_repaired_dates += symbol_repaired_count

                        results[result_index] = symbol_result
                        sync_status.update_history_audit_progress(
                            processed_symbols=processed_symbols,
                            success_symbols=success_symbols,
                            failed_symbols=failed_symbols,
                            current_symbol=symbol,
                            failed_tickers=failed_tickers,
                        )
                    work_queue.task_done()

        workers = [
            asyncio.create_task(_audit_worker())
            for _ in range(worker_count)
        ]

        try:
            await asyncio.gather(*workers)

            normalized_results: List[Dict[str, Any]] = []
            for index, symbol in enumerate(symbols_to_audit):
                result = results[index]
                if result is None:
                    result = {
                        "symbol": symbol,
                        "local_dates": 0,
                        "upstream_dates": 0,
                        "missing_dates": 0,
                        "repaired_dates": 0,
                        "missing_date_samples": [],
                        "error": "Audit worker did not complete",
                    }
                normalized_results.append(result)

            if failed_symbols > 0:
                sync_status.complete_history_audit(
                    success=True,
                    error=f"History audit completed with {failed_symbols} failed symbols",
                )
            else:
                sync_status.complete_history_audit(success=True)

            return {
                "started": True,
                "message": "History audit completed",
                "processed_symbols": processed_symbols,
                "success_symbols": success_symbols,
                "failed_symbols": failed_symbols,
                "audited_symbols": total_symbols,
                "symbols_with_gaps": symbols_with_gaps,
                "total_missing_dates": total_missing_dates,
                "total_repaired_dates": total_repaired_dates,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "results": normalized_results,
            }
        except Exception as e:
            sync_status.complete_history_audit(success=False, error=str(e)[:500])
            return {
                "started": False,
                "message": f"History audit failed: {e}",
                "processed_symbols": processed_symbols,
                "success_symbols": success_symbols,
                "failed_symbols": failed_symbols,
                "audited_symbols": total_symbols,
                "symbols_with_gaps": symbols_with_gaps,
                "total_missing_dates": total_missing_dates,
                "total_repaired_dates": total_repaired_dates,
                "results": [row for row in results if row is not None],
            }

    async def run_repair_sync(
        self,
        symbols: Optional[List[str]],
        start_date: date,
        end_date: date,
        index_symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run manual repair sync for selected symbols/date range."""
        clean_symbols = await self._resolve_symbols_filter(
            symbols=symbols,
            index_symbol=index_symbol,
        )
        clean_symbols = clean_symbols or []
        total_symbols = len(clean_symbols)

        if total_symbols == 0:
            return {
                "started": False,
                "message": "No symbols available for repair",
                "processed_symbols": 0,
                "success_symbols": 0,
                "failed_symbols": 0,
            }

        sync_status.start_history_repair(total_symbols=total_symbols)

        success_count = 0
        failure_count = 0
        failed_tickers: List[str] = []
        processed_count = 0
        progress_lock = asyncio.Lock()

        work_queue: asyncio.Queue[str] = asyncio.Queue()
        for symbol in clean_symbols:
            work_queue.put_nowait(symbol)

        worker_count = min(total_symbols, self._sync_max_workers)

        try:
            loop = asyncio.get_running_loop()
            sync_executor = self._ensure_sync_executor()

            async def _repair_worker() -> None:
                nonlocal processed_count, success_count, failure_count
                while True:
                    try:
                        symbol = work_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return

                    symbol_failed = False
                    try:
                        async with self._operation_worker_semaphore:
                            async with progress_lock:
                                sync_status.update_history_repair_progress(
                                    processed_symbols=processed_count,
                                    success_symbols=success_count,
                                    failed_symbols=failure_count,
                                    current_symbol=symbol,
                                )

                            await loop.run_in_executor(
                                sync_executor,
                                self._history._upsert_stock_daily_history,
                                symbol,
                                start_date,
                                end_date,
                            )
                            await self._mark_symbol_sync_result(symbol)
                    except Exception as e:
                        symbol_failed = True
                        await self._mark_symbol_error(symbol, f"Repair sync failed: {e}")
                    finally:
                        async with progress_lock:
                            processed_count += 1
                            if symbol_failed:
                                failure_count += 1
                                failed_tickers.append(symbol)
                            else:
                                success_count += 1

                            sync_status.update_history_repair_progress(
                                processed_symbols=processed_count,
                                success_symbols=success_count,
                                failed_symbols=failure_count,
                                current_symbol=symbol,
                                failed_tickers=failed_tickers,
                            )
                        work_queue.task_done()

            workers = [
                asyncio.create_task(_repair_worker())
                for _ in range(worker_count)
            ]
            await asyncio.gather(*workers)

            if failure_count > 0:
                sync_status.complete_history_repair(
                    success=True,
                    error=f"Repair sync completed with {failure_count} failed symbols",
                )
            else:
                sync_status.complete_history_repair(success=True)

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
            sync_status.complete_history_repair(success=False, error=str(e)[:500])
            return {
                "started": False,
                "message": f"Repair sync failed: {e}",
                "processed_symbols": success_count + failure_count,
                "success_symbols": success_count,
                "failed_symbols": failure_count,
            }

    async def _run_sync(self, symbols: Optional[List[str]]) -> None:
        """Background unified sync execution."""
        symbols_meta = await self._build_symbol_universe(symbols)
        total = len(symbols_meta)

        sync_status.start_history_sync(total_symbols=total)
        if total == 0:
            sync_status.complete_history_sync(success=True)
            return

        await self._ensure_sync_state_rows(symbols_meta)

        await self._reset_sync_pacer()
        history_sync_logger.info(
            "History sync runtime config: max_workers=%s target_rpm=%s chunk_days=%s "
            "fixed_wait=%.1fs max_wait=%.1fs",
            self._sync_max_workers,
            self._sync_target_rpm,
            self._sync_chunk_days,
            self._sync_rate_limit_fixed_wait_seconds,
            self._sync_rate_limit_max_wait_seconds,
        )

        work_queue: asyncio.Queue[SymbolSyncMeta] = asyncio.Queue()
        for meta in symbols_meta:
            work_queue.put_nowait(meta)

        worker_count = min(total, self._sync_max_workers)
        success_count = 0
        failure_count = 0
        failed_tickers: List[str] = []
        processed_count = 0
        progress_lock = asyncio.Lock()

        async def _sync_worker(worker_id: int) -> None:
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
                        sync_status.update_history_sync_progress(
                            processed_symbols=processed_count,
                            success_symbols=success_count,
                            failed_symbols=failure_count,
                            current_symbol=meta.symbol,
                        )
                    async with self._operation_worker_semaphore:
                        await self._sync_symbol(meta)
                except asyncio.CancelledError:
                    was_cancelled = True
                    raise
                except Exception as e:
                    symbol_failed = True
                    await self._mark_symbol_failed(meta.symbol, str(e))
                    history_sync_logger.error(
                        f"History sync worker {worker_id} failed symbol {meta.symbol}: {e}"
                    )
                finally:
                    if not was_cancelled:
                        async with progress_lock:
                            processed_count += 1
                            if symbol_failed:
                                failure_count += 1
                                failed_tickers.append(meta.symbol)
                            else:
                                success_count += 1

                            sync_status.update_history_sync_progress(
                                processed_symbols=processed_count,
                                success_symbols=success_count,
                                failed_symbols=failure_count,
                                current_symbol=meta.symbol,
                                failed_tickers=failed_tickers,
                            )
                    work_queue.task_done()

        workers = [
            asyncio.create_task(_sync_worker(worker_idx))
            for worker_idx in range(1, worker_count + 1)
        ]

        try:
            await asyncio.gather(*workers)
            if failure_count > 0:
                sync_status.complete_history_sync(
                    success=True,
                    error=f"History sync completed with {failure_count} failed symbols",
                )
            else:
                sync_status.complete_history_sync(success=True)

        except asyncio.CancelledError:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            sync_status.complete_history_sync(success=False, error="History sync cancelled")
            raise
        except Exception as e:
            sync_status.complete_history_sync(success=False, error=str(e)[:500])

    async def _sync_symbol(self, meta: SymbolSyncMeta) -> None:
        started = time.monotonic()
        chunk_count = 0
        retry_count = 0

        state = await self._get_or_create_symbol_state(meta.symbol, meta.listing_date)

        symbol_start_date = state.listing_date or meta.listing_date
        if symbol_start_date is None:
            loop = asyncio.get_event_loop()
            sync_executor = self._ensure_sync_executor()
            symbol_start_date = await loop.run_in_executor(
                sync_executor,
                self._discover_oldest_history_date,
                meta.symbol,
            )

        if symbol_start_date is None:
            raise RuntimeError("Unable to determine symbol start date")

        await self._set_symbol_sync_running(meta.symbol, symbol_start_date)

        today = date.today()
        _, latest_local = await self._get_symbol_bounds(meta.symbol)

        cursor = symbol_start_date
        if latest_local is not None:
            overlap_days = max(0, self.SYNC_OVERLAP_DAYS - 1)
            overlap_start = latest_local - timedelta(days=overlap_days)
            cursor = max(symbol_start_date, overlap_start)

        while cursor <= today:
            chunk_end = min(cursor + timedelta(days=self._sync_chunk_days - 1), today)
            retries = await self._run_sync_chunk_with_retry(
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

        await self._mark_symbol_sync_completed(
            symbol=meta.symbol,
            listing_date=symbol_start_date,
            earliest_date=oldest,
            latest_date=latest,
        )

        elapsed = time.monotonic() - started
        history_sync_logger.info(
            f"History sync symbol {meta.symbol} completed: chunks={chunk_count}, "
            f"retries={retry_count}, elapsed={elapsed:.2f}s"
        )

    def _ensure_sync_executor(self) -> ThreadPoolExecutor:
        if self._sync_executor is None:
            self._sync_executor = ThreadPoolExecutor(
                max_workers=self._sync_max_workers,
                thread_name_prefix="history_sync",
            )
        return self._sync_executor

    async def _reset_sync_pacer(self) -> None:
        async with self._sync_pacer_lock:
            self._sync_last_request_monotonic = None

    async def _acquire_sync_request_slot(self) -> None:
        interval_seconds = 60.0 / float(self._sync_target_rpm)
        async with self._sync_pacer_lock:
            now = time.monotonic()
            if self._sync_last_request_monotonic is not None:
                elapsed = now - self._sync_last_request_monotonic
                if elapsed < interval_seconds:
                    await asyncio.sleep(interval_seconds - elapsed)
                    now = time.monotonic()
            self._sync_last_request_monotonic = now

    async def _execute_sync_chunk_upsert(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> int:
        loop = asyncio.get_event_loop()
        sync_executor = self._ensure_sync_executor()
        upsert_call = partial(
            self._history._upsert_stock_daily_history,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            raise_on_error=True,
            log=history_sync_logger,
        )
        return await loop.run_in_executor(sync_executor, upsert_call)

    async def _run_sync_chunk_with_retry(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> int:
        attempt = 0
        total_rate_limit_wait_seconds = 0.0
        while True:
            attempt += 1
            retries_so_far = attempt - 1
            await shared_rate_limit_pause_controller.wait_if_paused()
            await self._acquire_sync_request_slot()
            started = time.monotonic()

            try:
                await self._execute_sync_chunk_upsert(symbol, start_date, end_date)
                elapsed = time.monotonic() - started
                history_sync_logger.debug(
                    f"History sync chunk synced for {symbol} ({start_date} -> {end_date}) "
                    f"attempt={attempt} elapsed={elapsed:.2f}s"
                )
                return retries_so_far
            except Exception as e:
                elapsed = time.monotonic() - started
                if not (_is_rate_limit_error(e) or isinstance(e, CircuitOpenError)):
                    history_sync_logger.debug(
                        f"History sync chunk failed for {symbol} ({start_date} -> {end_date}) "
                        f"attempt={attempt} elapsed={elapsed:.2f}s error={e}"
                    )
                    raise

                sleep_seconds = await shared_rate_limit_pause_controller.register_rate_limit_and_get_wait(
                    self._sync_rate_limit_fixed_wait_seconds
                )
                total_rate_limit_wait_seconds += sleep_seconds

                circuit_state = api_circuit_breaker.state.value
                circuit_delay = api_circuit_breaker.time_until_half_open or 0.0
                history_sync_logger.warning(
                    "History sync rate-limit pause symbol=%s start_date=%s end_date=%s attempt=%s "
                    "elapsed=%.2fs wait_seconds=%.2fs total_wait_seconds=%.2fs "
                    "circuit_state=%s circuit_time_until_half_open=%.2fs",
                    symbol,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    attempt,
                    elapsed,
                    sleep_seconds,
                    total_rate_limit_wait_seconds,
                    circuit_state,
                    circuit_delay,
                )

                if (
                    self._sync_rate_limit_max_wait_seconds > 0
                    and total_rate_limit_wait_seconds > self._sync_rate_limit_max_wait_seconds
                ):
                    history_sync_logger.error(
                        "History sync max rate-limit wait exceeded symbol=%s start_date=%s end_date=%s "
                        "total_wait_seconds=%.2fs cap_seconds=%.2fs attempts=%s",
                        symbol,
                        start_date.isoformat(),
                        end_date.isoformat(),
                        total_rate_limit_wait_seconds,
                        self._sync_rate_limit_max_wait_seconds,
                        attempt,
                    )
                    raise RuntimeError(
                        f"Rate limit persisted for {symbol} ({start_date} -> {end_date}) "
                        f"for {total_rate_limit_wait_seconds:.1f}s "
                        f"(cap={self._sync_rate_limit_max_wait_seconds:.1f}s)"
                    ) from e

                await asyncio.sleep(sleep_seconds)

    async def _repair_missing_dates(self, symbol: str, missing_dates: List[date]) -> int:
        fetch_windows = self._group_dates_into_fetch_windows(
            missing_dates,
            max_span_days=self._sync_chunk_days,
        )
        for range_start, range_end in fetch_windows:
            await self._run_sync_chunk_with_retry(symbol, range_start, range_end)

        await self._mark_symbol_sync_result(symbol)
        return len(missing_dates)

    async def _resolve_symbols_filter(
        self,
        symbols: Optional[List[str]],
        index_symbol: Optional[str],
    ) -> Optional[List[str]]:
        normalized_symbols = [
            self._normalize_symbol(s)
            for s in (symbols or [])
            if s
        ]
        normalized_symbols = [s for s in normalized_symbols if s]

        group_symbols: List[str] = []
        if index_symbol:
            group_symbols = await self._fetch_symbols_for_index(index_symbol)

        merged = list(dict.fromkeys([*normalized_symbols, *group_symbols]))
        return merged if merged else None

    async def _fetch_symbols_for_index(self, index_symbol: str) -> List[str]:
        loop = asyncio.get_event_loop()
        sync_executor = self._ensure_sync_executor()
        return await loop.run_in_executor(
            sync_executor,
            self._fetch_symbols_for_index_sync,
            index_symbol,
        )

    def _fetch_symbols_for_index_sync(self, index_symbol: str) -> List[str]:
        from vnstock import Listing

        normalized_index = str(index_symbol or "").strip().upper()
        mapped_group_code = get_group_code_for_index(normalized_index)
        valid_group_lookup = {group.upper(): group for group in VALID_GROUPS}
        group_code = valid_group_lookup.get(str(mapped_group_code).strip().upper())
        if group_code is None:
            raise ValueError(
                f"Unsupported index symbol/group '{index_symbol}'. "
                f"Supported groups include: {', '.join(sorted(VALID_GROUPS))}"
            )

        listing = Listing(source='VCI')
        symbols_df = listing.symbols_by_group(group_code)
        if symbols_df is None or symbols_df.empty:
            return []

        symbols = [self._normalize_symbol(value) for value in symbols_df.tolist()]
        return [symbol for symbol in symbols if symbol]

    async def _fetch_remote_history_dates(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> set[date]:
        loop = asyncio.get_event_loop()
        sync_executor = self._ensure_sync_executor()
        return await loop.run_in_executor(
            sync_executor,
            self._fetch_remote_history_dates_sync,
            symbol,
            start_date,
            end_date,
        )

    def _fetch_remote_history_dates_sync(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> set[date]:
        hist = self._history._fetch_ohlcv_history(
            symbol,
            start_date,
            end_date,
            source="VCI",
        )
        if hist is None or hist.empty:
            return set()

        times = pd.to_datetime(hist['time'], errors='coerce').dropna()
        return {
            ts.date()
            for ts in times
            if start_date <= ts.date() <= end_date
        }

    async def _get_local_history_dates(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> set[date]:
        async with async_session() as session:
            stmt = select(StockDailyHistory.date).where(
                StockDailyHistory.symbol == symbol,
                StockDailyHistory.date >= start_date,
                StockDailyHistory.date <= end_date,
            )
            result = await session.execute(stmt)
            return set(result.scalars().all())

    @staticmethod
    def _group_dates_into_ranges(dates_list: List[date]) -> List[tuple[date, date]]:
        if not dates_list:
            return []

        sorted_dates = sorted(dates_list)
        ranges: List[tuple[date, date]] = []

        start = sorted_dates[0]
        end = sorted_dates[0]

        for current in sorted_dates[1:]:
            if current == end + timedelta(days=1):
                end = current
                continue
            ranges.append((start, end))
            start = current
            end = current

        ranges.append((start, end))
        return ranges

    @staticmethod
    def _group_dates_into_fetch_windows(
        dates_list: List[date],
        max_span_days: int,
    ) -> List[tuple[date, date]]:
        if not dates_list:
            return []

        safe_span_days = max(1, int(max_span_days))
        sorted_dates = sorted(dates_list)
        windows: List[tuple[date, date]] = []

        window_start = sorted_dates[0]
        window_end = sorted_dates[0]

        for current in sorted_dates[1:]:
            if (current - window_start).days <= safe_span_days - 1:
                window_end = current
                continue

            windows.append((window_start, window_end))
            window_start = current
            window_end = current

        windows.append((window_start, window_end))
        return windows

    async def _build_symbol_universe(self, symbols: Optional[List[str]] = None) -> List[SymbolSyncMeta]:
        listing_map = await self._fetch_listing_symbol_map()

        if symbols is not None:
            normalized = [self._normalize_symbol(s) for s in symbols if s]
            filtered_symbols = sorted(set(s for s in normalized if s))
            return [
                SymbolSyncMeta(symbol=symbol, listing_date=listing_map.get(symbol))
                for symbol in filtered_symbols
            ]

        db_symbols = await self._fetch_db_symbols()
        all_symbols = set(listing_map.keys()) | set(db_symbols)
        sorted_symbols = sorted(all_symbols)

        return [
            SymbolSyncMeta(symbol=symbol, listing_date=listing_map.get(symbol))
            for symbol in sorted_symbols
        ]

    async def _fetch_listing_symbol_map(self) -> Dict[str, date | None]:
        loop = asyncio.get_event_loop()
        sync_executor = self._ensure_sync_executor()
        return await loop.run_in_executor(sync_executor, self._fetch_listing_symbol_map_sync)

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
            history_sync_logger.warning(f"Error fetching listing symbols for history sync: {e}")

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
        try:
            hist = self._history._fetch_ohlcv_history(
                symbol,
                self.FALLBACK_DISCOVERY_START_DATE,
                date.today(),
                source="VCI",
            )
            if hist is None or hist.empty:
                return None

            times = pd.to_datetime(hist['time'], errors='coerce').dropna()
            if times.empty:
                return None
            return times.min().date()
        except Exception as e:
            history_sync_logger.warning(f"Error discovering oldest history date for {symbol}: {e}")
            return None

    async def _fetch_db_symbols(self) -> List[str]:
        async with async_session() as session:
            company_stmt = select(StockCompany.symbol)
            company_result = await session.execute(company_stmt)
            company_symbols = [self._normalize_symbol(row[0]) for row in company_result.all() if row[0]]

            state_stmt = select(StockHistorySyncState.symbol)
            state_result = await session.execute(state_stmt)
            state_symbols = [self._normalize_symbol(row[0]) for row in state_result.all() if row[0]]

        return list(dict.fromkeys([*company_symbols, *state_symbols]))

    async def _ensure_sync_state_rows(self, symbols_meta: List[SymbolSyncMeta]) -> None:
        if not symbols_meta:
            return

        symbols = [meta.symbol for meta in symbols_meta]
        listing_map = {meta.symbol: meta.listing_date for meta in symbols_meta}

        async with async_session() as session:
            stmt = select(StockHistorySyncState).where(StockHistorySyncState.symbol.in_(symbols))
            existing_rows = await session.execute(stmt)
            existing = {row.symbol: row for row in existing_rows.scalars().all()}

            now = datetime.utcnow()
            for symbol in symbols:
                row = existing.get(symbol)
                listing_date = listing_map.get(symbol)
                if row is None:
                    session.add(
                        StockHistorySyncState(
                            symbol=symbol,
                            listing_date=listing_date,
                            sync_status='idle',
                            updated_at=now,
                        )
                    )
                    continue

                if row.listing_date is None and listing_date is not None:
                    row.listing_date = listing_date
                row.updated_at = now

            await session.commit()

    async def _get_or_create_symbol_state(self, symbol: str, listing_date: date | None) -> StockHistorySyncState:
        async with async_session() as session:
            stmt = select(StockHistorySyncState).where(StockHistorySyncState.symbol == symbol)
            row = (await session.execute(stmt)).scalar_one_or_none()

            if row is None:
                row = StockHistorySyncState(
                    symbol=symbol,
                    listing_date=listing_date,
                    sync_status='idle',
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

    async def _set_symbol_sync_running(self, symbol: str, listing_date: date) -> None:
        now = datetime.utcnow()
        async with async_session() as session:
            stmt = select(StockHistorySyncState).where(StockHistorySyncState.symbol == symbol)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                row = StockHistorySyncState(symbol=symbol)
                session.add(row)

            row.listing_date = row.listing_date or listing_date
            row.sync_status = 'running'
            row.sync_started_at = now
            row.last_error = None
            row.updated_at = now
            await session.commit()

    async def _mark_symbol_sync_completed(
        self,
        symbol: str,
        listing_date: date,
        earliest_date: date,
        latest_date: date,
    ) -> None:
        now = datetime.utcnow()
        async with async_session() as session:
            stmt = select(StockHistorySyncState).where(StockHistorySyncState.symbol == symbol)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                row = StockHistorySyncState(symbol=symbol)
                session.add(row)

            row.listing_date = row.listing_date or listing_date
            row.sync_status = 'completed'
            row.sync_completed_at = now
            row.earliest_synced_date = earliest_date
            row.latest_synced_date = latest_date
            row.last_incremental_sync_at = now
            row.last_error = None
            row.updated_at = now
            await session.commit()

    async def _mark_symbol_failed(self, symbol: str, error_message: str) -> None:
        now = datetime.utcnow()
        async with async_session() as session:
            stmt = select(StockHistorySyncState).where(StockHistorySyncState.symbol == symbol)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                row = StockHistorySyncState(symbol=symbol)
                session.add(row)

            row.sync_status = 'failed'
            row.last_error = error_message[:500]
            row.retry_count = (row.retry_count or 0) + 1
            row.updated_at = now
            await session.commit()

    async def _mark_symbol_error(self, symbol: str, error_message: str) -> None:
        now = datetime.utcnow()
        async with async_session() as session:
            stmt = select(StockHistorySyncState).where(StockHistorySyncState.symbol == symbol)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                row = StockHistorySyncState(symbol=symbol)
                session.add(row)

            row.last_error = error_message[:500]
            row.retry_count = (row.retry_count or 0) + 1
            row.updated_at = now
            await session.commit()

    async def _mark_symbol_sync_result(self, symbol: str) -> None:
        now = datetime.utcnow()
        oldest, latest = await self._get_symbol_bounds(symbol)

        async with async_session() as session:
            stmt = select(StockHistorySyncState).where(StockHistorySyncState.symbol == symbol)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                row = StockHistorySyncState(symbol=symbol)
                session.add(row)

            row.sync_status = 'completed'
            row.sync_completed_at = now
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
                func.min(StockDailyHistory.date),
                func.max(StockDailyHistory.date),
            ).where(StockDailyHistory.symbol == symbol)
            result = await session.execute(stmt)
            row = result.one()
            return row[0], row[1]

    async def _get_symbol_latest_date_iso(self, symbol: str) -> str | None:
        _, latest = await self._get_symbol_bounds(symbol)
        if latest is None:
            return None
        return latest.isoformat()

    def _normalize_symbol(self, symbol: Any) -> str:
        if symbol is None:
            return ""
        return str(symbol).strip().upper()[:3]
