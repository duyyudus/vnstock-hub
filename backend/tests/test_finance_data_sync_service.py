import asyncio

import pytest

from app.services.sync_status import sync_status
from app.services.vnstock_service.finance import FinanceService
from app.services.vnstock_service.finance_sync import FinanceDataSyncService


@pytest.mark.asyncio
async def test_finance_sync_parallel_workers_update_counters(monkeypatch):
    service = FinanceDataSyncService(finance=FinanceService())
    service._sync_max_workers = 3
    service._operation_worker_semaphore = asyncio.Semaphore(service._sync_max_workers)

    symbols = ["AAA", "AAB", "AAC", "AAD"]
    in_flight = 0
    max_in_flight = 0
    counter_lock = asyncio.Lock()

    async def _fake_build_symbol_universe(_symbols=None):
        return symbols

    async def _fake_sync_symbol(_symbol: str):
        nonlocal in_flight, max_in_flight
        async with counter_lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.02)
        async with counter_lock:
            in_flight -= 1

    monkeypatch.setattr(service, "_build_symbol_universe", _fake_build_symbol_universe)
    monkeypatch.setattr(service, "_sync_symbol", _fake_sync_symbol)

    await service._run_sync(symbols=None)

    runtime = sync_status.finance_sync
    assert runtime.is_running is False
    assert runtime.total_symbols == 4
    assert runtime.processed_symbols == 4
    assert runtime.success_symbols == 4
    assert runtime.failed_symbols == 0
    assert runtime.progress == 1.0
    assert max_in_flight > 1
    assert max_in_flight <= service._sync_max_workers


@pytest.mark.asyncio
async def test_resolve_symbols_filter_merges_manual_and_index(monkeypatch):
    service = FinanceDataSyncService(finance=FinanceService())

    async def _fake_fetch_symbols_for_index(_index_symbol: str):
        return ["FPT", "SSI", "HPG"]

    monkeypatch.setattr(service, "_fetch_symbols_for_index", _fake_fetch_symbols_for_index)

    symbols = await service._resolve_symbols_filter(
        symbols=["FPT", "VCB"],
        index_symbol="VN30",
    )

    assert symbols == ["FPT", "VCB", "SSI", "HPG"]


@pytest.mark.asyncio
async def test_run_sync_force_restart(monkeypatch):
    service = FinanceDataSyncService(finance=FinanceService())

    async def _fake_resolve_symbols_filter(symbols=None, index_symbol=None):
        return ["AAA"]

    async def _fake_run_sync(_symbols):
        await asyncio.sleep(0)

    monkeypatch.setattr(service, "_resolve_symbols_filter", _fake_resolve_symbols_filter)
    monkeypatch.setattr(service, "_run_sync", _fake_run_sync)

    existing_task = asyncio.create_task(asyncio.sleep(60))
    service._sync_task = existing_task

    not_restarted = await service.run_sync(force_restart=False, symbols=["AAA"], index_symbol=None)
    assert not_restarted["started"] is False

    restarted = await service.run_sync(force_restart=True, symbols=["AAA"], index_symbol=None)
    assert restarted["started"] is True
    assert restarted["state"] == "running"

    assert existing_task.cancelled()

    # Let newly created task run and clean up.
    await asyncio.sleep(0)
