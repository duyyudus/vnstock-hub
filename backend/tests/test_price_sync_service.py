from datetime import date
import asyncio

import pytest

import app.services.vnstock_service.price_sync as price_sync_module
from app.services.sync_status import sync_status
from app.services.vnstock_service.history import HistoryService
from app.services.vnstock_service.price_sync import PriceSyncService, SymbolBootstrapMeta


@pytest.mark.asyncio
async def test_bootstrap_chunk_rate_limit_then_success(monkeypatch):
    service = PriceSyncService(history=HistoryService())
    service._bootstrap_rate_limit_max_retries = 5
    service._bootstrap_retry_base_delay_seconds = 0.1
    service._bootstrap_retry_max_delay_seconds = 0.1

    attempts = []
    sleeps = []

    async def _no_pace():
        return None

    async def _fake_execute(symbol: str, start_date: date, end_date: date):
        attempts.append((symbol, start_date, end_date))
        if len(attempts) <= 2:
            raise RuntimeError("Rate limit exceeded")
        return 500

    async def _fake_sleep(seconds: float):
        sleeps.append(seconds)

    monkeypatch.setattr(service, "_acquire_bootstrap_request_slot", _no_pace)
    monkeypatch.setattr(service, "_execute_bootstrap_chunk_upsert", _fake_execute)
    monkeypatch.setattr(price_sync_module.asyncio, "sleep", _fake_sleep)

    retries = await service._run_bootstrap_chunk_with_retry(
        symbol="AAA",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 12, 31),
    )

    assert retries == 2
    assert len(attempts) == 3
    assert attempts[0][1:] == attempts[1][1:] == attempts[2][1:]
    assert len(sleeps) == 2


@pytest.mark.asyncio
async def test_bootstrap_chunk_non_rate_limit_error_does_not_retry(monkeypatch):
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

    monkeypatch.setattr(service, "_acquire_bootstrap_request_slot", _no_pace)
    monkeypatch.setattr(service, "_execute_bootstrap_chunk_upsert", _fake_execute)
    monkeypatch.setattr(price_sync_module.asyncio, "sleep", _fake_sleep)

    with pytest.raises(RuntimeError, match="database exploded"):
        await service._run_bootstrap_chunk_with_retry(
            symbol="AAA",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
        )

    assert len(attempts) == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_bootstrap_pacer_respects_min_interval(monkeypatch):
    service = PriceSyncService(history=HistoryService())
    service._bootstrap_target_rpm = 120  # 0.5s min interval

    fake_clock = {"now": 0.0}
    sleep_calls = []

    def _fake_monotonic() -> float:
        return fake_clock["now"]

    async def _fake_sleep(seconds: float):
        sleep_calls.append(seconds)
        fake_clock["now"] += seconds

    monkeypatch.setattr(price_sync_module.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(price_sync_module.asyncio, "sleep", _fake_sleep)

    await service._reset_bootstrap_pacer()
    await service._acquire_bootstrap_request_slot()
    fake_clock["now"] += 0.1
    await service._acquire_bootstrap_request_slot()
    fake_clock["now"] += 0.49
    await service._acquire_bootstrap_request_slot()

    assert len(sleep_calls) == 2
    assert sleep_calls[0] == pytest.approx(0.4, abs=1e-9)
    assert sleep_calls[1] == pytest.approx(0.01, abs=1e-9)


@pytest.mark.asyncio
async def test_bootstrap_parallel_workers_update_counters(monkeypatch):
    service = PriceSyncService(history=HistoryService())
    service._bootstrap_max_concurrency = 3

    symbols_meta = [
        SymbolBootstrapMeta(symbol="AAA", listing_date=date(2020, 1, 1)),
        SymbolBootstrapMeta(symbol="AAB", listing_date=date(2020, 1, 1)),
        SymbolBootstrapMeta(symbol="AAC", listing_date=date(2020, 1, 1)),
        SymbolBootstrapMeta(symbol="AAD", listing_date=date(2020, 1, 1)),
    ]

    in_flight = 0
    max_in_flight = 0
    counter_lock = asyncio.Lock()

    async def _fake_build_symbol_universe():
        return symbols_meta

    async def _fake_ensure_sync_state_rows(_symbols_meta):
        return None

    async def _fake_mark_symbol_failed(symbol: str, error_message: str):
        raise AssertionError(f"Did not expect failure for {symbol}: {error_message}")

    async def _fake_bootstrap_symbol(meta: SymbolBootstrapMeta):
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
    monkeypatch.setattr(service, "_bootstrap_symbol", _fake_bootstrap_symbol)

    await service._run_bootstrap()

    runtime = sync_status.price_bootstrap
    assert runtime.state == "completed"
    assert runtime.total_symbols == 4
    assert runtime.processed_symbols == 4
    assert runtime.success_symbols == 4
    assert runtime.failed_symbols == 0
    assert runtime.progress == 1.0
    assert max_in_flight > 1
