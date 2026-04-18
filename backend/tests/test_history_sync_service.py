from datetime import date
from types import SimpleNamespace
import asyncio
import threading
import time

import pandas as pd
import pytest
from sqlalchemy import select

import app.services.vnstock_service.history_sync as history_sync_module
from app.db.models import StockHistorySyncState
from app.services.sync_status import sync_status
from app.services.vnstock_service.history import HistoryService
from app.services.vnstock_service.history_sync import (
    HistorySyncService,
    RequestHistorySyncResult,
    SymbolSyncMeta,
)


@pytest.mark.asyncio
async def test_sync_chunk_rate_limit_then_success(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    service._sync_rate_limit_fixed_wait_seconds = 0.1

    attempts = []
    sleeps = []

    async def _no_pace():
        return None

    async def _no_pause():
        return None

    async def _fixed_wait(_fixed_wait_seconds: float):
        return 0.1

    async def _fake_execute(symbol: str, start_date: date, end_date: date):
        attempts.append((symbol, start_date, end_date))
        if len(attempts) <= 2:
            raise RuntimeError("Rate limit exceeded")
        return 500

    async def _fake_sleep(seconds: float):
        sleeps.append(seconds)

    monkeypatch.setattr(service, "_acquire_sync_request_slot", _no_pace)
    monkeypatch.setattr(service, "_execute_sync_chunk_upsert", _fake_execute)
    monkeypatch.setattr(
        history_sync_module.shared_rate_limit_pause_controller,
        "wait_if_paused",
        _no_pause,
    )
    monkeypatch.setattr(
        history_sync_module.shared_rate_limit_pause_controller,
        "register_rate_limit_and_get_wait",
        _fixed_wait,
    )
    monkeypatch.setattr(history_sync_module.asyncio, "sleep", _fake_sleep)

    retries = await service._run_sync_chunk_with_retry(
        symbol="AAA",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 12, 31),
    )

    assert retries == 2
    assert len(attempts) == 3
    assert attempts[0][1:] == attempts[1][1:] == attempts[2][1:]
    assert len(sleeps) == 2


@pytest.mark.asyncio
async def test_sync_chunk_rate_limit_exceeds_max_wait_cap(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    service._sync_rate_limit_fixed_wait_seconds = 0.1
    service._sync_rate_limit_max_wait_seconds = 0.25

    attempts = []
    sleeps = []

    async def _no_pace():
        return None

    async def _no_pause():
        return None

    async def _fixed_wait(_fixed_wait_seconds: float):
        return 0.1

    async def _fake_execute(symbol: str, start_date: date, end_date: date):
        attempts.append((symbol, start_date, end_date))
        raise RuntimeError("Rate limit exceeded")

    async def _fake_sleep(seconds: float):
        sleeps.append(seconds)

    monkeypatch.setattr(service, "_acquire_sync_request_slot", _no_pace)
    monkeypatch.setattr(service, "_execute_sync_chunk_upsert", _fake_execute)
    monkeypatch.setattr(
        history_sync_module.shared_rate_limit_pause_controller,
        "wait_if_paused",
        _no_pause,
    )
    monkeypatch.setattr(
        history_sync_module.shared_rate_limit_pause_controller,
        "register_rate_limit_and_get_wait",
        _fixed_wait,
    )
    monkeypatch.setattr(history_sync_module.asyncio, "sleep", _fake_sleep)

    with pytest.raises(RuntimeError, match=r"Rate limit persisted for AAA .*cap=0.2s"):
        await service._run_sync_chunk_with_retry(
            symbol="AAA",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
        )

    assert len(attempts) == 3
    assert sleeps == [0.1, 0.1]


@pytest.mark.asyncio
async def test_sync_chunk_non_rate_limit_error_does_not_retry(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    attempts = []
    sleeps = []

    async def _no_pace():
        return None

    async def _fake_execute(symbol: str, start_date: date, end_date: date):
        attempts.append((symbol, start_date, end_date))
        raise RuntimeError("database exploded")

    async def _fake_sleep(seconds: float):
        sleeps.append(seconds)

    monkeypatch.setattr(service, "_acquire_sync_request_slot", _no_pace)
    monkeypatch.setattr(service, "_execute_sync_chunk_upsert", _fake_execute)
    monkeypatch.setattr(history_sync_module.asyncio, "sleep", _fake_sleep)

    with pytest.raises(RuntimeError, match="database exploded"):
        await service._run_sync_chunk_with_retry(
            symbol="AAA",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
        )

    assert len(attempts) == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_sync_pacer_respects_min_interval(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    service._sync_target_rpm = 120  # 0.5s min interval

    fake_clock = {"now": 0.0}
    sleep_calls = []

    def _fake_monotonic() -> float:
        return fake_clock["now"]

    async def _fake_sleep(seconds: float):
        sleep_calls.append(seconds)
        fake_clock["now"] += seconds

    monkeypatch.setattr(history_sync_module.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(history_sync_module.asyncio, "sleep", _fake_sleep)

    await service._reset_sync_pacer()
    await service._acquire_sync_request_slot()
    fake_clock["now"] += 0.1
    await service._acquire_sync_request_slot()
    fake_clock["now"] += 0.49
    await service._acquire_sync_request_slot()

    assert len(sleep_calls) == 2
    assert sleep_calls[0] == pytest.approx(0.4, abs=1e-9)
    assert sleep_calls[1] == pytest.approx(0.01, abs=1e-9)


@pytest.mark.asyncio
async def test_sync_parallel_workers_update_counters(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    service._sync_max_workers = 3
    service._operation_worker_semaphore = asyncio.Semaphore(service._sync_max_workers)

    symbols_meta = [
        SymbolSyncMeta(symbol="AAA", listing_date=date(2020, 1, 1)),
        SymbolSyncMeta(symbol="AAB", listing_date=date(2020, 1, 1)),
        SymbolSyncMeta(symbol="AAC", listing_date=date(2020, 1, 1)),
        SymbolSyncMeta(symbol="AAD", listing_date=date(2020, 1, 1)),
    ]

    in_flight = 0
    max_in_flight = 0
    counter_lock = asyncio.Lock()

    async def _fake_build_symbol_universe(_symbols=None):
        return symbols_meta

    async def _fake_ensure_sync_state_rows(_symbols_meta):
        return None

    async def _fake_mark_symbol_failed(symbol: str, error_message: str):
        raise AssertionError(f"Did not expect failure for {symbol}: {error_message}")

    async def _fake_sync_symbol(meta: SymbolSyncMeta):
        nonlocal in_flight, max_in_flight
        async with counter_lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.02)
        async with counter_lock:
            in_flight -= 1

    monkeypatch.setattr(service, "_build_symbol_universe", _fake_build_symbol_universe)
    monkeypatch.setattr(service, "_ensure_sync_state_rows", _fake_ensure_sync_state_rows)
    monkeypatch.setattr(service, "_mark_symbol_failed", _fake_mark_symbol_failed)
    monkeypatch.setattr(service, "_sync_symbol", _fake_sync_symbol)

    await service._run_sync(symbols=None)

    runtime = sync_status.history_sync
    assert runtime.is_running is False
    assert runtime.total_symbols == 4
    assert runtime.processed_symbols == 4
    assert runtime.success_symbols == 4
    assert runtime.failed_symbols == 0
    assert runtime.failed_tickers == []
    assert runtime.progress == 1.0
    assert max_in_flight > 1


@pytest.mark.asyncio
async def test_sync_tracks_failed_tickers_and_resets(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    service._sync_max_workers = 1
    service._operation_worker_semaphore = asyncio.Semaphore(service._sync_max_workers)

    symbols_meta = [
        SymbolSyncMeta(symbol="AAA", listing_date=date(2020, 1, 1)),
        SymbolSyncMeta(symbol="BBB", listing_date=date(2020, 1, 1)),
        SymbolSyncMeta(symbol="CCC", listing_date=date(2020, 1, 1)),
    ]

    async def _fake_build_symbol_universe(_symbols=None):
        return symbols_meta

    async def _fake_ensure_sync_state_rows(_symbols_meta):
        return None

    async def _mark_symbol_failed(_symbol: str, _error_message: str):
        return None

    async def _fail_on_bbb(meta: SymbolSyncMeta):
        if meta.symbol == "BBB":
            raise RuntimeError("sync failed")

    async def _all_success(_meta: SymbolSyncMeta):
        return None

    monkeypatch.setattr(service, "_build_symbol_universe", _fake_build_symbol_universe)
    monkeypatch.setattr(service, "_ensure_sync_state_rows", _fake_ensure_sync_state_rows)
    monkeypatch.setattr(service, "_mark_symbol_failed", _mark_symbol_failed)
    monkeypatch.setattr(service, "_sync_symbol", _fail_on_bbb)

    await service._run_sync(symbols=None)

    runtime = sync_status.history_sync
    assert runtime.failed_symbols == 1
    assert runtime.failed_tickers == ["BBB"]

    monkeypatch.setattr(service, "_sync_symbol", _all_success)

    await service._run_sync(symbols=None)

    runtime = sync_status.history_sync
    assert runtime.failed_symbols == 0
    assert runtime.failed_tickers == []


@pytest.mark.asyncio
async def test_audit_parallel_workers_update_counters(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    service._sync_max_workers = 3
    service._operation_worker_semaphore = asyncio.Semaphore(service._sync_max_workers)

    symbols_to_audit = ["AAA", "AAB", "AAC", "AAD"]
    in_flight = 0
    max_in_flight = 0
    counter_lock = asyncio.Lock()

    async def _fake_resolve_symbols_filter(symbols=None, index_symbol=None):
        return symbols_to_audit

    async def _fake_get_local_history_dates(_symbol: str, _start_date: date, _end_date: date):
        return {date(2025, 1, 2)}

    async def _fake_fetch_remote_history_dates(_symbol: str, _start_date: date, _end_date: date):
        nonlocal in_flight, max_in_flight
        async with counter_lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.02)
        async with counter_lock:
            in_flight -= 1
        return {date(2025, 1, 2), date(2025, 1, 3)}

    async def _fake_mark_symbol_error(symbol: str, error_message: str):
        raise AssertionError(f"Did not expect failure for {symbol}: {error_message}")

    monkeypatch.setattr(service, "_resolve_symbols_filter", _fake_resolve_symbols_filter)
    monkeypatch.setattr(service, "_get_local_history_dates", _fake_get_local_history_dates)
    monkeypatch.setattr(service, "_fetch_remote_history_dates", _fake_fetch_remote_history_dates)
    monkeypatch.setattr(service, "_mark_symbol_error", _fake_mark_symbol_error)

    result = await service.run_audit_sync(
        symbols=None,
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 3),
        auto_repair=False,
        index_symbol="VN30",
    )

    runtime = sync_status.history_audit
    assert runtime.is_running is False
    assert runtime.total_symbols == 4
    assert runtime.processed_symbols == 4
    assert runtime.success_symbols == 4
    assert runtime.failed_symbols == 0
    assert runtime.failed_tickers == []
    assert runtime.progress == 1.0

    assert result["started"] is True
    assert result["audited_symbols"] == 4
    assert result["processed_symbols"] == 4
    assert result["success_symbols"] == 4
    assert result["failed_symbols"] == 0
    assert len(result["results"]) == 4
    assert max_in_flight > 1
    assert max_in_flight <= service._sync_max_workers


@pytest.mark.asyncio
async def test_audit_tracks_failed_tickers(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    service._sync_max_workers = 1
    service._operation_worker_semaphore = asyncio.Semaphore(service._sync_max_workers)

    async def _fake_resolve_symbols_filter(symbols=None, index_symbol=None):
        return ["AAA", "BBB"]

    async def _fake_get_local_history_dates(_symbol: str, _start_date: date, _end_date: date):
        return {date(2025, 1, 2)}

    async def _fake_fetch_remote_history_dates(symbol: str, _start_date: date, _end_date: date):
        if symbol == "BBB":
            raise RuntimeError("upstream failed")
        return {date(2025, 1, 2), date(2025, 1, 3)}

    async def _fake_mark_symbol_error(_symbol: str, _error_message: str):
        return None

    monkeypatch.setattr(service, "_resolve_symbols_filter", _fake_resolve_symbols_filter)
    monkeypatch.setattr(service, "_get_local_history_dates", _fake_get_local_history_dates)
    monkeypatch.setattr(service, "_fetch_remote_history_dates", _fake_fetch_remote_history_dates)
    monkeypatch.setattr(service, "_mark_symbol_error", _fake_mark_symbol_error)

    result = await service.run_audit_sync(
        symbols=None,
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 3),
        auto_repair=False,
        index_symbol="VN30",
    )

    runtime = sync_status.history_audit
    assert runtime.failed_symbols == 1
    assert runtime.failed_tickers == ["BBB"]
    assert result["failed_symbols"] == 1


@pytest.mark.asyncio
async def test_repair_parallel_workers_update_counters(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    service._sync_max_workers = 3
    service._operation_worker_semaphore = asyncio.Semaphore(service._sync_max_workers)

    in_flight = 0
    max_in_flight = 0
    counter_lock = threading.Lock()

    def _fake_upsert(_symbol: str, _start_date: date, _end_date: date):
        nonlocal in_flight, max_in_flight
        with counter_lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.02)
        with counter_lock:
            in_flight -= 1

    async def _fake_mark_symbol_sync_result(_symbol: str):
        return None

    async def _fake_mark_symbol_error(symbol: str, error_message: str):
        raise AssertionError(f"Did not expect failure for {symbol}: {error_message}")

    monkeypatch.setattr(service._history, "_upsert_stock_daily_history", _fake_upsert)
    monkeypatch.setattr(service, "_mark_symbol_sync_result", _fake_mark_symbol_sync_result)
    monkeypatch.setattr(service, "_mark_symbol_error", _fake_mark_symbol_error)

    result = await service.run_repair_sync(
        symbols=["AAA", "AAB", "AAC", "AAD"],
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 3),
    )

    runtime = sync_status.history_repair
    assert runtime.is_running is False
    assert runtime.total_symbols == 4
    assert runtime.processed_symbols == 4
    assert runtime.success_symbols == 4
    assert runtime.failed_symbols == 0
    assert runtime.failed_tickers == []
    assert runtime.progress == 1.0

    assert result["started"] is True
    assert result["processed_symbols"] == 4
    assert result["success_symbols"] == 4
    assert result["failed_symbols"] == 0
    assert max_in_flight > 1
    assert max_in_flight <= service._sync_max_workers


@pytest.mark.asyncio
async def test_repair_tracks_failed_tickers(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    service._sync_max_workers = 1
    service._operation_worker_semaphore = asyncio.Semaphore(service._sync_max_workers)

    def _fake_upsert(symbol: str, _start_date: date, _end_date: date):
        if symbol == "BBB":
            raise RuntimeError("repair failed")

    async def _fake_mark_symbol_sync_result(_symbol: str):
        return None

    async def _fake_mark_symbol_error(_symbol: str, _error_message: str):
        return None

    monkeypatch.setattr(service._history, "_upsert_stock_daily_history", _fake_upsert)
    monkeypatch.setattr(service, "_mark_symbol_sync_result", _fake_mark_symbol_sync_result)
    monkeypatch.setattr(service, "_mark_symbol_error", _fake_mark_symbol_error)

    result = await service.run_repair_sync(
        symbols=["AAA", "BBB", "CCC"],
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 3),
    )

    runtime = sync_status.history_repair
    assert runtime.failed_symbols == 1
    assert runtime.failed_tickers == ["BBB"]
    assert result["failed_symbols"] == 1


@pytest.mark.asyncio
async def test_repair_uses_merged_manual_and_index_symbols(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    repaired_symbols = []

    async def _fake_fetch_symbols_for_index(_index_symbol: str):
        return ["FPT", "SSI", "HPG"]

    def _fake_upsert(symbol: str, _start_date: date, _end_date: date):
        repaired_symbols.append(symbol)

    async def _fake_mark_symbol_sync_result(_symbol: str):
        return None

    async def _fake_mark_symbol_error(symbol: str, error_message: str):
        raise AssertionError(f"Did not expect failure for {symbol}: {error_message}")

    monkeypatch.setattr(service, "_fetch_symbols_for_index", _fake_fetch_symbols_for_index)
    monkeypatch.setattr(service._history, "_upsert_stock_daily_history", _fake_upsert)
    monkeypatch.setattr(service, "_mark_symbol_sync_result", _fake_mark_symbol_sync_result)
    monkeypatch.setattr(service, "_mark_symbol_error", _fake_mark_symbol_error)

    result = await service.run_repair_sync(
        symbols=["FPT", "VCB"],
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 3),
        index_symbol="VN30",
    )

    assert result["started"] is True
    assert result["processed_symbols"] == 4
    assert repaired_symbols == ["FPT", "VCB", "SSI", "HPG"]


@pytest.mark.asyncio
async def test_repair_supports_index_only_scope(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    repaired_symbols = []

    async def _fake_fetch_symbols_for_index(_index_symbol: str):
        return ["AAA", "BBB"]

    def _fake_upsert(symbol: str, _start_date: date, _end_date: date):
        repaired_symbols.append(symbol)

    async def _fake_mark_symbol_sync_result(_symbol: str):
        return None

    async def _fake_mark_symbol_error(symbol: str, error_message: str):
        raise AssertionError(f"Did not expect failure for {symbol}: {error_message}")

    monkeypatch.setattr(service, "_fetch_symbols_for_index", _fake_fetch_symbols_for_index)
    monkeypatch.setattr(service._history, "_upsert_stock_daily_history", _fake_upsert)
    monkeypatch.setattr(service, "_mark_symbol_sync_result", _fake_mark_symbol_sync_result)
    monkeypatch.setattr(service, "_mark_symbol_error", _fake_mark_symbol_error)

    result = await service.run_repair_sync(
        symbols=None,
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 3),
        index_symbol="VN30",
    )

    assert result["started"] is True
    assert result["processed_symbols"] == 2
    assert repaired_symbols == ["AAA", "BBB"]


@pytest.mark.asyncio
async def test_repair_returns_not_started_without_symbols_or_index():
    service = HistorySyncService(history=HistoryService())

    result = await service.run_repair_sync(
        symbols=None,
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 3),
        index_symbol=None,
    )

    assert result == {
        "started": False,
        "message": "No symbols available for repair",
        "processed_symbols": 0,
        "success_symbols": 0,
        "failed_symbols": 0,
    }


@pytest.mark.asyncio
async def test_global_worker_cap_shared_across_sync_audit_repair(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    service._sync_max_workers = 2
    service._operation_worker_semaphore = asyncio.Semaphore(service._sync_max_workers)

    in_flight = 0
    max_in_flight = 0
    counter_lock = threading.Lock()

    def _enter_slot() -> None:
        nonlocal in_flight, max_in_flight
        with counter_lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)

    def _exit_slot() -> None:
        nonlocal in_flight
        with counter_lock:
            in_flight -= 1

    async def _fake_build_symbol_universe(_symbols=None):
        return [
            SymbolSyncMeta(symbol="S1", listing_date=date(2020, 1, 1)),
            SymbolSyncMeta(symbol="S2", listing_date=date(2020, 1, 1)),
            SymbolSyncMeta(symbol="S3", listing_date=date(2020, 1, 1)),
        ]

    async def _fake_ensure_sync_state_rows(_symbols_meta):
        return None

    async def _fake_sync_symbol(_meta: SymbolSyncMeta):
        _enter_slot()
        await asyncio.sleep(0.03)
        _exit_slot()

    async def _fake_mark_symbol_failed(symbol: str, error_message: str):
        raise AssertionError(f"Did not expect failure for {symbol}: {error_message}")

    async def _fake_resolve_symbols_filter(symbols=None, index_symbol=None):
        return ["A1", "A2", "A3"]

    async def _fake_get_local_history_dates(_symbol: str, _start_date: date, _end_date: date):
        return {date(2025, 1, 3)}

    async def _fake_fetch_remote_history_dates(_symbol: str, _start_date: date, _end_date: date):
        _enter_slot()
        await asyncio.sleep(0.03)
        _exit_slot()
        return {date(2025, 1, 3)}

    async def _fake_mark_symbol_sync_result(_symbol: str):
        return None

    async def _fake_mark_symbol_error(symbol: str, error_message: str):
        raise AssertionError(f"Did not expect failure for {symbol}: {error_message}")

    def _fake_upsert(_symbol: str, _start_date: date, _end_date: date):
        _enter_slot()
        time.sleep(0.03)
        _exit_slot()

    monkeypatch.setattr(service, "_build_symbol_universe", _fake_build_symbol_universe)
    monkeypatch.setattr(service, "_ensure_sync_state_rows", _fake_ensure_sync_state_rows)
    monkeypatch.setattr(service, "_sync_symbol", _fake_sync_symbol)
    monkeypatch.setattr(service, "_mark_symbol_failed", _fake_mark_symbol_failed)
    monkeypatch.setattr(service, "_resolve_symbols_filter", _fake_resolve_symbols_filter)
    monkeypatch.setattr(service, "_get_local_history_dates", _fake_get_local_history_dates)
    monkeypatch.setattr(service, "_fetch_remote_history_dates", _fake_fetch_remote_history_dates)
    monkeypatch.setattr(service, "_mark_symbol_sync_result", _fake_mark_symbol_sync_result)
    monkeypatch.setattr(service, "_mark_symbol_error", _fake_mark_symbol_error)
    monkeypatch.setattr(service._history, "_upsert_stock_daily_history", _fake_upsert)

    await asyncio.gather(
        service._run_sync(symbols=None),
        service.run_audit_sync(
            symbols=None,
            start_date=date(2025, 1, 3),
            end_date=date(2025, 1, 3),
            auto_repair=False,
            index_symbol="VN30",
        ),
        service.run_repair_sync(
            symbols=["R1", "R2", "R3"],
            start_date=date(2025, 1, 3),
            end_date=date(2025, 1, 3),
        ),
    )

    assert max_in_flight > 1
    assert max_in_flight <= service._sync_max_workers


@pytest.mark.asyncio
async def test_sync_symbol_uses_db_max_date_with_two_day_overlap(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    service._sync_chunk_days = 4000

    bounds_calls = {"count": 0}
    chunk_starts = []

    async def _fake_get_or_create_symbol_state(_symbol, listing_date=None):
        return SimpleNamespace(listing_date=date(2020, 1, 1))

    async def _fake_set_running(_symbol, _listing_date):
        return None

    async def _fake_get_bounds(_symbol):
        bounds_calls["count"] += 1
        if bounds_calls["count"] == 1:
            return date(2020, 1, 1), date(2025, 1, 10)
        return date(2020, 1, 1), date(2025, 1, 10)

    async def _fake_run_chunk(_symbol, start_date: date, _end_date: date):
        chunk_starts.append(start_date)
        return 0

    async def _fake_mark_completed(**_kwargs):
        return None

    monkeypatch.setattr(service, "_get_or_create_symbol_state", _fake_get_or_create_symbol_state)
    monkeypatch.setattr(service, "_set_symbol_sync_running", _fake_set_running)
    monkeypatch.setattr(service, "_get_symbol_bounds", _fake_get_bounds)
    monkeypatch.setattr(service, "_run_sync_chunk_with_retry", _fake_run_chunk)
    monkeypatch.setattr(service, "_mark_symbol_sync_completed", _fake_mark_completed)

    await service._sync_symbol(SymbolSyncMeta(symbol="AAA", listing_date=date(2020, 1, 1)))

    assert chunk_starts
    assert chunk_starts[0] == date(2025, 1, 9)


@pytest.mark.asyncio
async def test_sync_symbol_without_history_starts_from_listing_date(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    service._sync_chunk_days = 10000

    chunk_starts = []
    bounds_calls = {"count": 0}

    async def _fake_get_or_create_symbol_state(_symbol, _listing_date):
        return SimpleNamespace(listing_date=date(2021, 6, 15))

    async def _fake_set_running(_symbol, _listing_date):
        return None

    async def _fake_get_bounds(_symbol):
        bounds_calls["count"] += 1
        if bounds_calls["count"] == 1:
            return None, None
        return date(2021, 6, 15), date(2021, 6, 15)

    async def _fake_run_chunk(_symbol, start_date: date, _end_date: date):
        chunk_starts.append(start_date)
        return 0

    async def _fake_mark_completed(**_kwargs):
        return None

    monkeypatch.setattr(service, "_get_or_create_symbol_state", _fake_get_or_create_symbol_state)
    monkeypatch.setattr(service, "_set_symbol_sync_running", _fake_set_running)
    monkeypatch.setattr(service, "_get_symbol_bounds", _fake_get_bounds)
    monkeypatch.setattr(service, "_run_sync_chunk_with_retry", _fake_run_chunk)
    monkeypatch.setattr(service, "_mark_symbol_sync_completed", _fake_mark_completed)

    await service._sync_symbol(SymbolSyncMeta(symbol="AAA", listing_date=date(2021, 6, 15)))

    assert chunk_starts
    assert chunk_starts[0] == date(2021, 6, 15)


def test_fetch_remote_history_dates_sync_uses_history_service_quote_helper(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    calls = []

    def _fake_fetch(symbol: str, start_date: date, end_date: date, source: str = "VCI"):
        calls.append((symbol, start_date, end_date, source))
        return pd.DataFrame(
            [
                {"time": "2025-01-03"},
                {"time": "2025-01-05"},
                {"time": "2025-01-08"},
                {"time": None},
            ]
        )

    monkeypatch.setattr(service._history, "_fetch_ohlcv_history", _fake_fetch)

    result = service._fetch_remote_history_dates_sync(
        "AAA",
        date(2025, 1, 4),
        date(2025, 1, 7),
    )

    assert calls == [("AAA", date(2025, 1, 4), date(2025, 1, 7), "VCI")]
    assert result == {date(2025, 1, 5)}


def test_discover_oldest_history_date_uses_history_service_quote_helper(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    calls = []

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 4, 18)

    def _fake_fetch(symbol: str, start_date: date, end_date: date, source: str = "VCI"):
        calls.append((symbol, start_date, end_date, source))
        return pd.DataFrame(
            [
                {"time": "1999-01-04"},
                {"time": "2000-01-03"},
            ]
        )

    monkeypatch.setattr(history_sync_module, "date", FixedDate)
    monkeypatch.setattr(service._history, "_fetch_ohlcv_history", _fake_fetch)

    result = service._discover_oldest_history_date("AAA")

    assert calls == [("AAA", service.FALLBACK_DISCOVERY_START_DATE, FixedDate.today(), "VCI")]
    assert result == date(1999, 1, 4)


@pytest.mark.asyncio
async def test_mark_symbol_sync_result_sets_completed_status(monkeypatch, db_session):
    service = HistorySyncService(history=HistoryService())

    async def _fake_bounds(_symbol: str):
        return date(2020, 1, 2), date(2026, 2, 12)

    monkeypatch.setattr(service, "_get_symbol_bounds", _fake_bounds)

    db_session.add(StockHistorySyncState(symbol="AAA", sync_status="running"))
    await db_session.commit()

    await service._mark_symbol_sync_result("AAA")

    stmt = select(StockHistorySyncState).where(StockHistorySyncState.symbol == "AAA")
    row = (await db_session.execute(stmt)).scalar_one()

    assert row.sync_status == "completed"
    assert row.sync_completed_at is not None
    assert row.last_incremental_sync_at is not None
    assert row.earliest_synced_date == date(2020, 1, 2)
    assert row.latest_synced_date == date(2026, 2, 12)


@pytest.mark.asyncio
async def test_resolve_symbols_filter_merges_manual_and_index(monkeypatch):
    service = HistorySyncService(history=HistoryService())

    async def _fake_fetch_symbols_for_index(_index_symbol: str):
        return ["FPT", "SSI", "HPG"]

    monkeypatch.setattr(service, "_fetch_symbols_for_index", _fake_fetch_symbols_for_index)

    symbols = await service._resolve_symbols_filter(
        symbols=["FPT", "VCB"],
        index_symbol="VN30",
    )
    assert symbols == ["FPT", "VCB", "SSI", "HPG"]


@pytest.mark.asyncio
async def test_resolve_symbols_filter_returns_none_without_filters():
    service = HistorySyncService(history=HistoryService())
    symbols = await service._resolve_symbols_filter(
        symbols=None,
        index_symbol=None,
    )
    assert symbols is None


@pytest.mark.asyncio
async def test_request_sync_runs_incremental_overlap_and_repairs_requested_range(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    service._sync_chunk_days = 30
    chunk_calls = []
    repaired_calls = []

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 2, 11)

    async def _fake_get_or_create_symbol_state(_symbol, listing_date=None):
        return SimpleNamespace(listing_date=date(2020, 1, 1))

    async def _fake_get_symbol_bounds(_symbol):
        return date(2020, 1, 1), date(2026, 2, 8)

    async def _fake_set_symbol_sync_running(symbol, listing_date):
        return None

    async def _fake_run_sync_chunk_with_retry(_symbol, start_date: date, end_date: date):
        chunk_calls.append((start_date, end_date))
        return 0

    async def _fake_get_local_history_dates(_symbol, _start_date, _end_date):
        return {date(2026, 2, 5)}

    async def _fake_fetch_remote_history_dates(_symbol, _start_date, _end_date):
        return {date(2026, 2, 5), date(2026, 2, 6)}

    async def _fake_repair_missing_dates(_symbol, missing_dates):
        repaired_calls.append(list(missing_dates))
        return len(missing_dates)

    async def _fake_latest_iso(_symbol):
        return "2026-02-11"

    monkeypatch.setattr(history_sync_module, "date", FixedDate)
    monkeypatch.setattr(service, "_get_or_create_symbol_state", _fake_get_or_create_symbol_state)
    monkeypatch.setattr(service, "_get_symbol_bounds", _fake_get_symbol_bounds)
    monkeypatch.setattr(service, "_set_symbol_sync_running", _fake_set_symbol_sync_running)
    monkeypatch.setattr(service, "_run_sync_chunk_with_retry", _fake_run_sync_chunk_with_retry)
    monkeypatch.setattr(service, "_get_local_history_dates", _fake_get_local_history_dates)
    monkeypatch.setattr(service, "_fetch_remote_history_dates", _fake_fetch_remote_history_dates)
    monkeypatch.setattr(service, "_repair_missing_dates", _fake_repair_missing_dates)
    monkeypatch.setattr(service, "_get_symbol_latest_date_iso", _fake_latest_iso)

    result = await service.sync_symbol_history_for_request(
        symbol="aaax",
        start_date=date(2026, 2, 1),
        end_date=date(2026, 2, 11),
        timeout_seconds=5.0,
    )

    assert result["sync_performed"] is True
    assert result["sync_timed_out"] is False
    assert result["sync_error"] is None
    assert result["updated_through"] == "2026-02-11"
    assert result["repaired_missing_dates"] == 1
    assert chunk_calls == [(date(2026, 2, 7), date(2026, 2, 11))]
    assert repaired_calls == [[date(2026, 2, 6)]]


@pytest.mark.asyncio
async def test_repair_missing_dates_batches_sparse_gaps_into_fetch_windows(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    service._sync_chunk_days = 365
    chunk_calls = []
    marked = []

    async def _fake_run_sync_chunk_with_retry(_symbol, start_date: date, end_date: date):
        chunk_calls.append((start_date, end_date))
        return 0

    async def _fake_mark_symbol_sync_result(symbol: str):
        marked.append(symbol)
        return None

    monkeypatch.setattr(service, "_run_sync_chunk_with_retry", _fake_run_sync_chunk_with_retry)
    monkeypatch.setattr(service, "_mark_symbol_sync_result", _fake_mark_symbol_sync_result)

    repaired = await service._repair_missing_dates(
        "AAA",
        [
            date(2025, 1, 2),
            date(2025, 3, 15),
            date(2025, 12, 31),
            date(2026, 1, 2),
        ],
    )

    assert repaired == 4
    assert chunk_calls == [
        (date(2025, 1, 2), date(2025, 12, 31)),
        (date(2026, 1, 2), date(2026, 1, 2)),
    ]
    assert marked == ["AAA"]


@pytest.mark.asyncio
async def test_request_sync_deduplicates_concurrent_requests_for_same_symbol(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    executions = {"count": 0}

    async def _fake_request_task(symbol, start_date, end_date):
        executions["count"] += 1
        await asyncio.sleep(0.03)
        return RequestHistorySyncResult(
            sync_performed=True,
            updated_through="2026-02-11",
        )

    monkeypatch.setattr(service, "_sync_symbol_history_for_request_task", _fake_request_task)

    first, second = await asyncio.gather(
        service.sync_symbol_history_for_request(
            symbol="AAA",
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 11),
            timeout_seconds=1.0,
        ),
        service.sync_symbol_history_for_request(
            symbol="AAA",
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 11),
            timeout_seconds=1.0,
        ),
    )

    assert executions["count"] == 1
    assert first["sync_performed"] is True
    assert second["sync_performed"] is True


@pytest.mark.asyncio
async def test_request_sync_timeout_returns_fallback_and_task_continues(monkeypatch):
    service = HistorySyncService(history=HistoryService())
    started = asyncio.Event()
    finished = asyncio.Event()

    async def _fake_request_task(symbol, start_date, end_date):
        started.set()
        await asyncio.sleep(0.05)
        finished.set()
        return RequestHistorySyncResult(
            sync_performed=True,
            updated_through="2026-02-11",
        )

    async def _fake_latest_iso(_symbol):
        return "2026-02-10"

    monkeypatch.setattr(service, "_sync_symbol_history_for_request_task", _fake_request_task)
    monkeypatch.setattr(service, "_get_symbol_latest_date_iso", _fake_latest_iso)

    result = await service.sync_symbol_history_for_request(
        symbol="AAA",
        start_date=date(2026, 2, 1),
        end_date=date(2026, 2, 11),
        timeout_seconds=0.01,
    )

    assert started.is_set()
    assert result["sync_performed"] is False
    assert result["sync_timed_out"] is True
    assert result["sync_error"] is None
    assert result["updated_through"] == "2026-02-10"

    await asyncio.wait_for(finished.wait(), timeout=1.0)
    await asyncio.sleep(0)
    assert "AAA" not in service._request_sync_tasks
