from datetime import date
from types import SimpleNamespace
import asyncio
import threading
import time

import pytest

import app.services.vnstock_service.price_sync as price_sync_module
from app.services.sync_status import sync_status
from app.services.vnstock_service.history import HistoryService
from app.services.vnstock_service.price_sync import PriceSyncService, SymbolSyncMeta


@pytest.mark.asyncio
async def test_sync_chunk_rate_limit_then_success(monkeypatch):
    service = PriceSyncService(history=HistoryService())
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
        price_sync_module.shared_rate_limit_pause_controller,
        "wait_if_paused",
        _no_pause,
    )
    monkeypatch.setattr(
        price_sync_module.shared_rate_limit_pause_controller,
        "register_rate_limit_and_get_wait",
        _fixed_wait,
    )
    monkeypatch.setattr(price_sync_module.asyncio, "sleep", _fake_sleep)

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
    service = PriceSyncService(history=HistoryService())
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
        price_sync_module.shared_rate_limit_pause_controller,
        "wait_if_paused",
        _no_pause,
    )
    monkeypatch.setattr(
        price_sync_module.shared_rate_limit_pause_controller,
        "register_rate_limit_and_get_wait",
        _fixed_wait,
    )
    monkeypatch.setattr(price_sync_module.asyncio, "sleep", _fake_sleep)

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
    service = PriceSyncService(history=HistoryService())
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
    monkeypatch.setattr(price_sync_module.asyncio, "sleep", _fake_sleep)

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
    service = PriceSyncService(history=HistoryService())
    service._sync_target_rpm = 120  # 0.5s min interval

    fake_clock = {"now": 0.0}
    sleep_calls = []

    def _fake_monotonic() -> float:
        return fake_clock["now"]

    async def _fake_sleep(seconds: float):
        sleep_calls.append(seconds)
        fake_clock["now"] += seconds

    monkeypatch.setattr(price_sync_module.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(price_sync_module.asyncio, "sleep", _fake_sleep)

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
    service = PriceSyncService(history=HistoryService())
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

    runtime = sync_status.price_sync
    assert runtime.is_running is False
    assert runtime.total_symbols == 4
    assert runtime.processed_symbols == 4
    assert runtime.success_symbols == 4
    assert runtime.failed_symbols == 0
    assert runtime.progress == 1.0
    assert max_in_flight > 1


@pytest.mark.asyncio
async def test_audit_parallel_workers_update_counters(monkeypatch):
    service = PriceSyncService(history=HistoryService())
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

    runtime = sync_status.price_audit
    assert runtime.is_running is False
    assert runtime.total_symbols == 4
    assert runtime.processed_symbols == 4
    assert runtime.success_symbols == 4
    assert runtime.failed_symbols == 0
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
async def test_repair_parallel_workers_update_counters(monkeypatch):
    service = PriceSyncService(history=HistoryService())
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

    monkeypatch.setattr(service._history, "_upsert_stock_price_history", _fake_upsert)
    monkeypatch.setattr(service, "_mark_symbol_sync_result", _fake_mark_symbol_sync_result)
    monkeypatch.setattr(service, "_mark_symbol_error", _fake_mark_symbol_error)

    result = await service.run_repair_sync(
        symbols=["AAA", "AAB", "AAC", "AAD"],
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 3),
    )

    runtime = sync_status.price_repair
    assert runtime.is_running is False
    assert runtime.total_symbols == 4
    assert runtime.processed_symbols == 4
    assert runtime.success_symbols == 4
    assert runtime.failed_symbols == 0
    assert runtime.progress == 1.0

    assert result["started"] is True
    assert result["processed_symbols"] == 4
    assert result["success_symbols"] == 4
    assert result["failed_symbols"] == 0
    assert max_in_flight > 1
    assert max_in_flight <= service._sync_max_workers


@pytest.mark.asyncio
async def test_global_worker_cap_shared_across_sync_audit_repair(monkeypatch):
    service = PriceSyncService(history=HistoryService())
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
    monkeypatch.setattr(service._history, "_upsert_stock_price_history", _fake_upsert)

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
    service = PriceSyncService(history=HistoryService())
    service._sync_chunk_days = 4000

    bounds_calls = {"count": 0}
    chunk_starts = []

    async def _fake_get_or_create_symbol_state(_symbol, _listing_date):
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
    service = PriceSyncService(history=HistoryService())
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


@pytest.mark.asyncio
async def test_resolve_symbols_filter_merges_manual_and_index(monkeypatch):
    service = PriceSyncService(history=HistoryService())

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
    service = PriceSyncService(history=HistoryService())
    symbols = await service._resolve_symbols_filter(
        symbols=None,
        index_symbol=None,
    )
    assert symbols is None
