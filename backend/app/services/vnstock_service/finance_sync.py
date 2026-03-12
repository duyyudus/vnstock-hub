from __future__ import annotations

from typing import Any, Dict, List, Optional
import asyncio
import time

from sqlalchemy import distinct, func, select

from app.core.config import settings
from app.db.database import async_session
from app.db.models import StockCompany, StockFinancialDataCache
from app.services.sync_status import sync_status

from .core import (
    finance_sync_logger,
    frontend_executor,
    api_circuit_breaker,
    CircuitOpenError,
    _is_rate_limit_error,
)
from .finance import FinanceService
from .rate_limit_pause import shared_rate_limit_pause_controller
from .symbols import VALID_GROUPS, get_group_code_for_index


class FinanceDataSyncService:
    """Background synchronization for company finance datasets."""

    DATA_TYPES = (
        FinanceService.DATA_TYPE_INCOME,
        FinanceService.DATA_TYPE_CASHFLOW,
        FinanceService.DATA_TYPE_BALANCE_SHEET,
        FinanceService.DATA_TYPE_RATIOS,
    )

    def __init__(self, finance: FinanceService) -> None:
        self._finance = finance
        self._sync_task: asyncio.Task | None = None
        self._sync_lock = asyncio.Lock()
        self._sync_max_workers = max(1, int(settings.sync_max_workers))
        self._sync_target_rpm = max(1, int(settings.sync_target_rpm))
        self._sync_rate_limit_fixed_wait_seconds = max(
            0.1,
            float(settings.sync_rate_limit_fixed_wait_seconds),
        )
        self._sync_rate_limit_max_wait_seconds = max(
            0.0,
            float(settings.sync_rate_limit_max_wait_seconds),
        )
        self._sync_pacer_lock = asyncio.Lock()
        self._sync_last_request_monotonic: float | None = None
        self._operation_worker_semaphore = asyncio.Semaphore(self._sync_max_workers)

    async def start_background_tasks(self) -> None:
        """No-op hook kept for lifecycle symmetry."""
        return

    async def stop_background_tasks(self) -> None:
        """Cancel running finance sync task on shutdown."""
        async with self._sync_lock:
            if self._sync_task and not self._sync_task.done():
                self._sync_task.cancel()
                try:
                    await self._sync_task
                except asyncio.CancelledError:
                    pass
            self._sync_task = None

    async def run_sync(
        self,
        force_restart: bool = False,
        symbols: Optional[List[str]] = None,
        index_symbol: Optional[str] = None,
        quick_sync: bool = False,
    ) -> Dict[str, Any]:
        """Start finance data sync as background task."""
        symbols_filter = await self._resolve_symbols_filter(symbols=symbols, index_symbol=index_symbol)

        async with self._sync_lock:
            if self._sync_task and not self._sync_task.done():
                if not force_restart:
                    return {
                        "started": False,
                        "message": "Finance sync is already running",
                        "state": "running",
                    }

                self._sync_task.cancel()
                try:
                    await self._sync_task
                except asyncio.CancelledError:
                    pass

            finance_sync_logger.info(
                "Finance sync requested (force_restart=%s, symbols=%s, index_symbol=%s, quick_sync=%s)",
                force_restart,
                len(symbols_filter) if symbols_filter else 0,
                index_symbol or "",
                quick_sync,
            )
            self._sync_task = asyncio.create_task(
                self._run_sync(symbols_filter, quick_sync=quick_sync)
            )

        return {
            "started": True,
            "message": "Finance sync started",
            "state": "running",
        }

    async def _run_sync(self, symbols: Optional[List[str]], quick_sync: bool = False) -> None:
        symbols_to_sync = await self._build_symbol_universe(symbols)
        if quick_sync:
            symbols_to_sync = await self._filter_quick_sync_symbols(symbols_to_sync)
        total = len(symbols_to_sync)

        finance_sync_logger.info(
            "Finance sync started for %s symbols (quick_sync=%s)",
            total,
            quick_sync,
        )
        sync_status.start_finance_sync(total_symbols=total)
        if total == 0:
            finance_sync_logger.info("Finance sync completed immediately (no symbols)")
            sync_status.complete_finance_sync(success=True)
            return

        await self._reset_sync_pacer()
        finance_sync_logger.info(
            "Finance sync runtime config: max_workers=%s target_rpm=%s fixed_wait=%.1fs max_wait=%.1fs",
            self._sync_max_workers,
            self._sync_target_rpm,
            self._sync_rate_limit_fixed_wait_seconds,
            self._sync_rate_limit_max_wait_seconds,
        )

        work_queue: asyncio.Queue[str] = asyncio.Queue()
        for symbol in symbols_to_sync:
            work_queue.put_nowait(symbol)

        worker_count = min(total, self._sync_max_workers)
        success_count = 0
        failure_count = 0
        failed_tickers: list[str] = []
        processed_count = 0
        progress_lock = asyncio.Lock()

        async def _sync_worker(worker_id: int) -> None:
            nonlocal processed_count, success_count, failure_count

            while True:
                try:
                    symbol = work_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                symbol_failed = False
                was_cancelled = False
                try:
                    async with progress_lock:
                        sync_status.update_finance_sync_progress(
                            processed_symbols=processed_count,
                            success_symbols=success_count,
                            failed_symbols=failure_count,
                            current_symbol=symbol,
                        )

                    async with self._operation_worker_semaphore:
                        await self._sync_symbol(symbol)
                except asyncio.CancelledError:
                    was_cancelled = True
                    raise
                except Exception as e:
                    symbol_failed = True
                    finance_sync_logger.error(
                        f"Finance sync worker {worker_id} failed symbol {symbol}: {e}"
                    )
                finally:
                    if not was_cancelled:
                        async with progress_lock:
                            processed_count += 1
                            if symbol_failed:
                                failure_count += 1
                                failed_tickers.append(symbol)
                            else:
                                success_count += 1

                            sync_status.update_finance_sync_progress(
                                processed_symbols=processed_count,
                                success_symbols=success_count,
                                failed_symbols=failure_count,
                                current_symbol=symbol,
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
                finance_sync_logger.warning(
                    "Finance sync completed with failures: success=%s failed=%s",
                    success_count,
                    failure_count,
                )
                sync_status.complete_finance_sync(
                    success=True,
                    error=f"Finance sync completed with {failure_count} failed symbols",
                )
            else:
                finance_sync_logger.info(
                    "Finance sync completed successfully: success=%s failed=%s",
                    success_count,
                    failure_count,
                )
                sync_status.complete_finance_sync(success=True)
        except asyncio.CancelledError:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            finance_sync_logger.warning("Finance sync cancelled")
            sync_status.complete_finance_sync(success=False, error="Finance sync cancelled")
            raise
        except Exception as e:
            finance_sync_logger.error("Finance sync failed: %s", e)
            sync_status.complete_finance_sync(success=False, error=str(e)[:500])

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

    async def _run_finance_fetch_with_retry(
        self,
        symbol: str,
        data_type: str,
        lang: str = "en",
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
                await self._finance.refresh_financial_dataset(
                    symbol=symbol,
                    data_type=data_type,
                    lang=lang,
                    raise_on_failure=True,
                )
                elapsed = time.monotonic() - started
                finance_sync_logger.debug(
                    "Finance sync dataset synced for %s (%s) attempt=%s elapsed=%.2fs",
                    symbol,
                    data_type,
                    attempt,
                    elapsed,
                )
                return retries_so_far
            except Exception as e:
                elapsed = time.monotonic() - started
                if not (_is_rate_limit_error(e) or isinstance(e, CircuitOpenError)):
                    finance_sync_logger.debug(
                        "Finance sync dataset failed for %s (%s) attempt=%s elapsed=%.2fs error=%s",
                        symbol,
                        data_type,
                        attempt,
                        elapsed,
                        e,
                    )
                    raise

                sleep_seconds = await shared_rate_limit_pause_controller.register_rate_limit_and_get_wait(
                    self._sync_rate_limit_fixed_wait_seconds
                )
                total_rate_limit_wait_seconds += sleep_seconds

                circuit_state = api_circuit_breaker.state.value
                circuit_delay = api_circuit_breaker.time_until_half_open or 0.0
                finance_sync_logger.warning(
                    "Finance sync rate-limit pause symbol=%s data_type=%s attempt=%s "
                    "elapsed=%.2fs wait_seconds=%.2fs total_wait_seconds=%.2fs "
                    "circuit_state=%s circuit_time_until_half_open=%.2fs",
                    symbol,
                    data_type,
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
                    finance_sync_logger.error(
                        "Finance sync max rate-limit wait exceeded symbol=%s data_type=%s "
                        "total_wait_seconds=%.2fs cap_seconds=%.2fs attempts=%s",
                        symbol,
                        data_type,
                        total_rate_limit_wait_seconds,
                        self._sync_rate_limit_max_wait_seconds,
                        attempt,
                    )
                    raise RuntimeError(
                        f"Rate limit persisted for {symbol} ({data_type}) "
                        f"for {total_rate_limit_wait_seconds:.1f}s "
                        f"(cap={self._sync_rate_limit_max_wait_seconds:.1f}s)"
                    ) from e

                await asyncio.sleep(sleep_seconds)

    async def _sync_symbol(self, symbol: str) -> None:
        for data_type in self.DATA_TYPES:
            retries = await self._run_finance_fetch_with_retry(
                symbol=symbol,
                data_type=data_type,
                lang="en",
            )
            if retries > 0:
                finance_sync_logger.debug(
                    "Finance sync dataset completed after retries for %s (%s): retries=%s",
                    symbol,
                    data_type,
                    retries,
                )

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
        return await loop.run_in_executor(frontend_executor, self._fetch_symbols_for_index_sync, index_symbol)

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

    async def _build_symbol_universe(self, symbols: Optional[List[str]] = None) -> List[str]:
        if symbols is not None:
            normalized = [self._normalize_symbol(s) for s in symbols if s]
            return sorted(set(s for s in normalized if s))

        listing_symbols = await self._fetch_listing_symbols()
        db_symbols = await self._fetch_db_symbols()
        return sorted(set(listing_symbols) | set(db_symbols))

    async def _fetch_listing_symbols(self) -> List[str]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(frontend_executor, self._fetch_listing_symbols_sync)

    def _fetch_listing_symbols_sync(self) -> List[str]:
        from vnstock import Listing

        result: List[str] = []
        try:
            df = Listing(source='VCI').all_symbols()
            if df is None or df.empty:
                return result

            for _, row in df.iterrows():
                symbol = self._normalize_symbol(row.get('symbol'))
                if symbol:
                    result.append(symbol)

        except Exception as e:
            finance_sync_logger.warning(f"Error fetching listing symbols for finance sync: {e}")

        return result

    async def _fetch_db_symbols(self) -> List[str]:
        async with async_session() as session:
            company_stmt = select(StockCompany.symbol)
            company_result = await session.execute(company_stmt)
            company_symbols = [self._normalize_symbol(row[0]) for row in company_result.all() if row[0]]

            finance_stmt = select(StockFinancialDataCache.symbol)
            finance_result = await session.execute(finance_stmt)
            finance_symbols = [self._normalize_symbol(row[0]) for row in finance_result.all() if row[0]]

        return list(dict.fromkeys([*company_symbols, *finance_symbols]))

    async def _filter_quick_sync_symbols(self, symbols: List[str]) -> List[str]:
        if not symbols:
            return symbols

        fully_synced_symbols = await self._fetch_fully_synced_symbols(symbols)
        filtered_symbols = [symbol for symbol in symbols if symbol not in fully_synced_symbols]

        finance_sync_logger.info(
            "Finance quick sync filter applied: input=%s skipped_fully_synced=%s queued=%s",
            len(symbols),
            len(fully_synced_symbols),
            len(filtered_symbols),
        )
        return filtered_symbols

    async def _fetch_fully_synced_symbols(self, symbols: List[str]) -> set[str]:
        async with async_session() as session:
            stmt = (
                select(StockFinancialDataCache.symbol)
                .where(
                    StockFinancialDataCache.symbol.in_(symbols),
                    StockFinancialDataCache.period == self._finance.DEFAULT_PERIOD,
                    StockFinancialDataCache.lang == "en",
                    StockFinancialDataCache.data_type.in_(self.DATA_TYPES),
                )
                .group_by(StockFinancialDataCache.symbol)
                .having(func.count(distinct(StockFinancialDataCache.data_type)) == len(self.DATA_TYPES))
            )
            result = await session.execute(stmt)

        return {
            self._normalize_symbol(row[0])
            for row in result.all()
            if row[0]
        }

    def _normalize_symbol(self, symbol: Any) -> str:
        if symbol is None:
            return ""
        return str(symbol).strip().upper()[:3]
