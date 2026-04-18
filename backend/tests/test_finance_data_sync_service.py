import asyncio

import pytest

import app.services.vnstock_service.finance_sync as finance_sync_module
from app.services.sync_status import sync_status
from app.services.vnstock_service.core import CircuitOpenError, api_circuit_breaker
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

    async def _fake_sync_symbol(_symbol: str, force_refresh: bool = False):
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
    assert runtime.failed_tickers == []
    assert runtime.progress == 1.0
    assert max_in_flight > 1
    assert max_in_flight <= service._sync_max_workers


@pytest.mark.asyncio
async def test_finance_sync_tracks_failed_tickers_and_resets(monkeypatch):
    service = FinanceDataSyncService(finance=FinanceService())
    service._sync_max_workers = 1
    service._operation_worker_semaphore = asyncio.Semaphore(service._sync_max_workers)

    symbols = ["AAA", "BBB", "CCC"]

    async def _fake_build_symbol_universe(_symbols=None):
        return symbols

    async def _fail_on_bbb(symbol: str, force_refresh: bool = False):
        if symbol == "BBB":
            raise RuntimeError("sync failed")

    async def _all_success(_symbol: str, force_refresh: bool = False):
        return None

    monkeypatch.setattr(service, "_build_symbol_universe", _fake_build_symbol_universe)
    monkeypatch.setattr(service, "_sync_symbol", _fail_on_bbb)

    await service._run_sync(symbols=None)

    runtime = sync_status.finance_sync
    assert runtime.failed_symbols == 1
    assert runtime.failed_tickers == ["BBB"]

    monkeypatch.setattr(service, "_sync_symbol", _all_success)

    await service._run_sync(symbols=None)

    runtime = sync_status.finance_sync
    assert runtime.failed_symbols == 0
    assert runtime.failed_tickers == []


@pytest.mark.asyncio
async def test_finance_sync_quick_mode_skips_fully_synced_symbols(monkeypatch):
    service = FinanceDataSyncService(finance=FinanceService())

    async def _fake_fetch_fully_synced_symbols(symbols):
        assert symbols == ["AAA", "BBB", "CCC"]
        return {"AAA"}

    monkeypatch.setattr(service, "_fetch_fully_synced_symbols", _fake_fetch_fully_synced_symbols)

    filtered_symbols = await service._filter_quick_sync_symbols(["AAA", "BBB", "CCC"])

    assert filtered_symbols == ["BBB", "CCC"]


@pytest.mark.asyncio
async def test_finance_sync_quick_mode_keeps_partial_symbols(monkeypatch):
    service = FinanceDataSyncService(finance=FinanceService())

    async def _fake_fetch_fully_synced_symbols(symbols):
        assert symbols == ["AAA", "BBB"]
        return {"AAA"}

    monkeypatch.setattr(service, "_fetch_fully_synced_symbols", _fake_fetch_fully_synced_symbols)

    filtered_symbols = await service._filter_quick_sync_symbols(["AAA", "BBB"])

    assert filtered_symbols == ["BBB"]


@pytest.mark.asyncio
async def test_finance_sync_normal_mode_does_not_apply_quick_filter(monkeypatch):
    service = FinanceDataSyncService(finance=FinanceService())

    async def _fake_build_symbol_universe(_symbols=None):
        return ["AAA", "BBB"]

    async def _unexpected_filter(_symbols):
        raise AssertionError("quick filter should not be called when quick_sync is disabled")

    processed = []

    async def _fake_sync_symbol(symbol: str, force_refresh: bool = False):
        processed.append(symbol)

    monkeypatch.setattr(service, "_build_symbol_universe", _fake_build_symbol_universe)
    monkeypatch.setattr(service, "_filter_quick_sync_symbols", _unexpected_filter)
    monkeypatch.setattr(service, "_sync_symbol", _fake_sync_symbol)

    await service._run_sync(symbols=None, quick_sync=False)

    assert processed == ["AAA", "BBB"]


@pytest.mark.asyncio
async def test_finance_sync_force_refresh_flows_to_symbol_sync(monkeypatch):
    service = FinanceDataSyncService(finance=FinanceService())

    async def _fake_build_symbol_universe(_symbols=None):
        return ["AAA", "BBB"]

    processed = []

    async def _fake_sync_symbol(symbol: str, force_refresh: bool = False):
        processed.append((symbol, force_refresh))

    monkeypatch.setattr(service, "_build_symbol_universe", _fake_build_symbol_universe)
    monkeypatch.setattr(service, "_sync_symbol", _fake_sync_symbol)

    await service._run_sync(symbols=None, quick_sync=False, force_refresh=True)

    assert processed == [("AAA", True), ("BBB", True)]


@pytest.mark.asyncio
async def test_finance_fetch_rate_limit_then_success(monkeypatch):
    service = FinanceDataSyncService(finance=FinanceService())
    service._sync_rate_limit_fixed_wait_seconds = 0.1

    attempts = []
    sleeps = []
    sync_status.clear_rate_limit()
    api_circuit_breaker.reset()

    async def _no_pace():
        return None

    async def _no_pause():
        return None

    async def _fixed_wait(_fixed_wait_seconds: float):
        return 0.1

    async def _fake_refresh(
        symbol: str,
        data_type: str,
        lang: str = "en",
        raise_on_failure: bool = False,
        force_refresh: bool = False,
    ):
        attempts.append((symbol, data_type, lang, raise_on_failure, force_refresh))
        if len(attempts) <= 2:
            raise RuntimeError("Rate limit exceeded")
        return []

    async def _fake_sleep(seconds: float):
        sleeps.append(seconds)

    monkeypatch.setattr(service, "_acquire_sync_request_slot", _no_pace)
    monkeypatch.setattr(service._finance, "refresh_financial_dataset", _fake_refresh)
    monkeypatch.setattr(
        finance_sync_module.shared_rate_limit_pause_controller,
        "wait_if_paused",
        _no_pause,
    )
    monkeypatch.setattr(
        finance_sync_module.shared_rate_limit_pause_controller,
        "register_rate_limit_and_get_wait",
        _fixed_wait,
    )
    monkeypatch.setattr(finance_sync_module.asyncio, "sleep", _fake_sleep)

    retries = await service._run_finance_fetch_with_retry(
        symbol="AAA",
        data_type=FinanceService.DATA_TYPE_INCOME,
    )

    assert retries == 2
    assert len(attempts) == 3
    assert attempts[0][:2] == attempts[1][:2] == attempts[2][:2] == ("AAA", "income")
    assert all(entry[3] is True for entry in attempts)
    assert all(entry[4] is False for entry in attempts)
    assert len(sleeps) == 2


@pytest.mark.asyncio
async def test_finance_fetch_non_rate_limit_does_not_retry(monkeypatch):
    service = FinanceDataSyncService(finance=FinanceService())
    attempts = []
    sleeps = []

    async def _no_pace():
        return None

    async def _fake_refresh(
        symbol: str,
        data_type: str,
        lang: str = "en",
        raise_on_failure: bool = False,
        force_refresh: bool = False,
    ):
        attempts.append((symbol, data_type, lang, raise_on_failure, force_refresh))
        raise RuntimeError("database exploded")

    async def _fake_sleep(seconds: float):
        sleeps.append(seconds)

    monkeypatch.setattr(service, "_acquire_sync_request_slot", _no_pace)
    monkeypatch.setattr(service._finance, "refresh_financial_dataset", _fake_refresh)
    monkeypatch.setattr(finance_sync_module.asyncio, "sleep", _fake_sleep)

    with pytest.raises(RuntimeError, match="database exploded"):
        await service._run_finance_fetch_with_retry(
            symbol="AAA",
            data_type=FinanceService.DATA_TYPE_INCOME,
        )

    assert len(attempts) == 1
    assert sleeps == []

@pytest.mark.asyncio
async def test_finance_fetch_rate_limit_exceeds_max_wait_cap(monkeypatch):
    service = FinanceDataSyncService(finance=FinanceService())
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

    async def _fake_refresh(
        symbol: str,
        data_type: str,
        lang: str = "en",
        raise_on_failure: bool = False,
        force_refresh: bool = False,
    ):
        attempts.append((symbol, data_type, lang, raise_on_failure, force_refresh))
        raise CircuitOpenError("Rate limited")

    async def _fake_sleep(seconds: float):
        sleeps.append(seconds)

    monkeypatch.setattr(service, "_acquire_sync_request_slot", _no_pace)
    monkeypatch.setattr(service._finance, "refresh_financial_dataset", _fake_refresh)
    monkeypatch.setattr(
        finance_sync_module.shared_rate_limit_pause_controller,
        "wait_if_paused",
        _no_pause,
    )
    monkeypatch.setattr(
        finance_sync_module.shared_rate_limit_pause_controller,
        "register_rate_limit_and_get_wait",
        _fixed_wait,
    )
    monkeypatch.setattr(finance_sync_module.asyncio, "sleep", _fake_sleep)

    with pytest.raises(RuntimeError, match=r"Rate limit persisted for AAA .*cap=0.2s"):
        await service._run_finance_fetch_with_retry(
            symbol="AAA",
            data_type=FinanceService.DATA_TYPE_INCOME,
        )

    assert len(attempts) == 3
    assert sleeps == [0.1, 0.1]


@pytest.mark.asyncio
async def test_finance_sync_pacer_respects_min_interval(monkeypatch):
    service = FinanceDataSyncService(finance=FinanceService())
    service._sync_target_rpm = 120  # 0.5s min interval

    fake_clock = {"now": 0.0}
    sleep_calls = []

    def _fake_monotonic() -> float:
        return fake_clock["now"]

    async def _fake_sleep(seconds: float):
        sleep_calls.append(seconds)
        fake_clock["now"] += seconds

    monkeypatch.setattr(finance_sync_module.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(finance_sync_module.asyncio, "sleep", _fake_sleep)

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
async def test_sync_symbol_fetches_each_data_type_once(monkeypatch):
    service = FinanceDataSyncService(finance=FinanceService())
    calls = []

    async def _fake_run_finance_fetch_with_retry(
        symbol: str,
        data_type: str,
        lang: str = "en",
        force_refresh: bool = False,
    ) -> int:
        calls.append((symbol, data_type, lang, force_refresh))
        return 0

    monkeypatch.setattr(service, "_run_finance_fetch_with_retry", _fake_run_finance_fetch_with_retry)

    await service._sync_symbol("AAA")

    assert [entry[1] for entry in calls] == list(service.DATA_TYPES)
    assert all(entry[0] == "AAA" and entry[2] == "en" and entry[3] is False for entry in calls)


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

    async def _fake_run_sync(_symbols, quick_sync=False, force_refresh=False):
        assert quick_sync is False
        assert force_refresh is True
        await asyncio.sleep(0)

    monkeypatch.setattr(service, "_resolve_symbols_filter", _fake_resolve_symbols_filter)
    monkeypatch.setattr(service, "_run_sync", _fake_run_sync)

    existing_task = asyncio.create_task(asyncio.sleep(60))
    service._sync_task = existing_task

    not_restarted = await service.run_sync(force_restart=False, symbols=["AAA"], index_symbol=None)
    assert not_restarted["started"] is False

    restarted = await service.run_sync(
        force_restart=True,
        symbols=["AAA"],
        index_symbol=None,
        force_refresh=True,
    )
    assert restarted["started"] is True
    assert restarted["state"] == "running"

    assert existing_task.cancelled()

    # Let newly created task run and clean up.
    await asyncio.sleep(0)
