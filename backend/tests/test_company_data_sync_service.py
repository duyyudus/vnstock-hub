import asyncio
from concurrent.futures import Future

import pytest
from tenacity import RetryError

import app.services.vnstock_service.company_sync as company_sync_module
from app.services.sync_status import sync_status
from app.services.vnstock_service.company import CompanyService
from app.services.vnstock_service.company_sync import CompanyDataSyncService
from app.services.vnstock_service.core import CircuitOpenError, api_circuit_breaker


@pytest.mark.asyncio
async def test_company_sync_parallel_workers_update_counters(monkeypatch):
    service = CompanyDataSyncService(company=CompanyService())
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

    runtime = sync_status.company_sync
    assert runtime.is_running is False
    assert runtime.total_symbols == 4
    assert runtime.processed_symbols == 4
    assert runtime.success_symbols == 4
    assert runtime.failed_symbols == 0
    assert runtime.progress == 1.0
    assert max_in_flight > 1
    assert max_in_flight <= service._sync_max_workers


@pytest.mark.asyncio
async def test_company_sync_quick_mode_skips_fully_synced_symbols(monkeypatch):
    service = CompanyDataSyncService(company=CompanyService())

    async def _fake_fetch_fully_synced_symbols(symbols):
        assert symbols == ["AAA", "BBB", "CCC"]
        return {"AAA"}

    monkeypatch.setattr(service, "_fetch_fully_synced_symbols", _fake_fetch_fully_synced_symbols)

    filtered_symbols = await service._filter_quick_sync_symbols(["AAA", "BBB", "CCC"])

    assert filtered_symbols == ["BBB", "CCC"]


@pytest.mark.asyncio
async def test_company_sync_quick_mode_keeps_partial_symbols(monkeypatch):
    service = CompanyDataSyncService(company=CompanyService())

    async def _fake_fetch_fully_synced_symbols(symbols):
        assert symbols == ["AAA", "BBB"]
        return {"AAA"}

    monkeypatch.setattr(service, "_fetch_fully_synced_symbols", _fake_fetch_fully_synced_symbols)

    filtered_symbols = await service._filter_quick_sync_symbols(["AAA", "BBB"])

    assert filtered_symbols == ["BBB"]


@pytest.mark.asyncio
async def test_company_sync_normal_mode_does_not_apply_quick_filter(monkeypatch):
    service = CompanyDataSyncService(company=CompanyService())

    async def _fake_build_symbol_universe(_symbols=None):
        return ["AAA", "BBB"]

    async def _unexpected_filter(_symbols):
        raise AssertionError("quick filter should not be called when quick_sync is disabled")

    processed = []

    async def _fake_sync_symbol(symbol: str):
        processed.append(symbol)

    monkeypatch.setattr(service, "_build_symbol_universe", _fake_build_symbol_universe)
    monkeypatch.setattr(service, "_filter_quick_sync_symbols", _unexpected_filter)
    monkeypatch.setattr(service, "_sync_symbol", _fake_sync_symbol)

    await service._run_sync(symbols=None, quick_sync=False)

    assert processed == ["AAA", "BBB"]


@pytest.mark.asyncio
async def test_company_fetch_rate_limit_then_success(monkeypatch):
    service = CompanyDataSyncService(company=CompanyService())
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
        raise_on_failure: bool = False,
        force_refresh: bool = False,
    ):
        attempts.append((symbol, data_type, raise_on_failure, force_refresh))
        if len(attempts) <= 2:
            raise RuntimeError("Rate limit exceeded")
        return []

    async def _fake_sleep(seconds: float):
        sleeps.append(seconds)

    monkeypatch.setattr(service, "_acquire_sync_request_slot", _no_pace)
    monkeypatch.setattr(service._company, "refresh_company_dataset", _fake_refresh)
    monkeypatch.setattr(
        company_sync_module.shared_rate_limit_pause_controller,
        "wait_if_paused",
        _no_pause,
    )
    monkeypatch.setattr(
        company_sync_module.shared_rate_limit_pause_controller,
        "register_rate_limit_and_get_wait",
        _fixed_wait,
    )
    monkeypatch.setattr(company_sync_module.asyncio, "sleep", _fake_sleep)

    retries = await service._run_company_fetch_with_retry(
        symbol="AAA",
        data_type=CompanyService.DATA_TYPE_OVERVIEW,
    )

    assert retries == 2
    assert len(attempts) == 3
    assert attempts[0][:2] == attempts[1][:2] == attempts[2][:2] == ("AAA", "overview")
    assert all(entry[2] is True and entry[3] is True for entry in attempts)
    assert len(sleeps) == 2


@pytest.mark.asyncio
async def test_company_fetch_half_open_non_rate_limit_reopens_and_recovers(monkeypatch):
    company = CompanyService()
    service = CompanyDataSyncService(company=company)
    service._sync_rate_limit_fixed_wait_seconds = 0.06
    service._sync_rate_limit_max_wait_seconds = 1.0

    sync_status.clear_rate_limit()
    api_circuit_breaker.reset()

    original_failure_threshold = api_circuit_breaker.config.failure_threshold
    original_recovery_timeout = api_circuit_breaker.config.recovery_timeout
    original_half_open_max_calls = api_circuit_breaker.config.half_open_max_calls
    original_half_open_probe_timeout = api_circuit_breaker.config.half_open_probe_timeout

    api_circuit_breaker.config.failure_threshold = 1
    api_circuit_breaker.config.recovery_timeout = 0.05
    api_circuit_breaker.config.half_open_max_calls = 1
    api_circuit_breaker.config.half_open_probe_timeout = 0.02

    overview_calls = {"count": 0}

    class _Client:
        def overview(self):
            overview_calls["count"] += 1
            if overview_calls["count"] == 1:
                raise RuntimeError("Rate limit exceeded")
            if overview_calls["count"] == 2:
                raise RuntimeError("database exploded")
            return [{"symbol": "AAA"}]

    async def _fake_get_cache_entry(*_args, **_kwargs):
        return None

    async def _fake_upsert_cache_entry(*_args, **_kwargs):
        return None

    async def _no_pace():
        return None

    async def _no_pause():
        return None

    async def _fixed_wait(_fixed_wait_seconds: float):
        return 0.06

    def _fast_record_rate_limit(reset_seconds: float = 30.0):
        _ = reset_seconds
        api_circuit_breaker.record_failure(reset_timeout=0.05)
        sync_status.set_rate_limited(reset_in_seconds=0.05)

    monkeypatch.setattr(company, "_build_company_client", lambda _symbol: _Client())
    monkeypatch.setattr(company, "_get_cache_entry", _fake_get_cache_entry)
    monkeypatch.setattr(company, "_upsert_cache_entry", _fake_upsert_cache_entry)
    monkeypatch.setattr(service, "_acquire_sync_request_slot", _no_pace)
    monkeypatch.setattr(
        company_sync_module.shared_rate_limit_pause_controller,
        "wait_if_paused",
        _no_pause,
    )
    monkeypatch.setattr(
        company_sync_module.shared_rate_limit_pause_controller,
        "register_rate_limit_and_get_wait",
        _fixed_wait,
    )
    monkeypatch.setattr("app.services.vnstock_service.company._record_rate_limit", _fast_record_rate_limit)

    try:
        retries = await service._run_company_fetch_with_retry(
            symbol="AAA",
            data_type=CompanyService.DATA_TYPE_OVERVIEW,
        )
    finally:
        api_circuit_breaker.config.failure_threshold = original_failure_threshold
        api_circuit_breaker.config.recovery_timeout = original_recovery_timeout
        api_circuit_breaker.config.half_open_max_calls = original_half_open_max_calls
        api_circuit_breaker.config.half_open_probe_timeout = original_half_open_probe_timeout
        api_circuit_breaker.reset()
        sync_status.clear_rate_limit()

    assert retries == 2
    assert overview_calls["count"] == 3


@pytest.mark.asyncio
async def test_company_fetch_non_rate_limit_does_not_retry(monkeypatch):
    service = CompanyDataSyncService(company=CompanyService())
    attempts = []
    sleeps = []

    async def _no_pace():
        return None

    async def _fake_refresh(
        symbol: str,
        data_type: str,
        raise_on_failure: bool = False,
        force_refresh: bool = False,
    ):
        attempts.append((symbol, data_type, raise_on_failure, force_refresh))
        raise RuntimeError("database exploded")

    async def _fake_sleep(seconds: float):
        sleeps.append(seconds)

    monkeypatch.setattr(service, "_acquire_sync_request_slot", _no_pace)
    monkeypatch.setattr(service._company, "refresh_company_dataset", _fake_refresh)
    monkeypatch.setattr(company_sync_module.asyncio, "sleep", _fake_sleep)

    with pytest.raises(RuntimeError, match="database exploded"):
        await service._run_company_fetch_with_retry(
            symbol="AAA",
            data_type=CompanyService.DATA_TYPE_OVERVIEW,
        )

    assert len(attempts) == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_company_fetch_rate_limit_exceeds_max_wait_cap(monkeypatch):
    service = CompanyDataSyncService(company=CompanyService())
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
        raise_on_failure: bool = False,
        force_refresh: bool = False,
    ):
        attempts.append((symbol, data_type, raise_on_failure, force_refresh))
        raise CircuitOpenError("Rate limited")

    async def _fake_sleep(seconds: float):
        sleeps.append(seconds)

    monkeypatch.setattr(service, "_acquire_sync_request_slot", _no_pace)
    monkeypatch.setattr(service._company, "refresh_company_dataset", _fake_refresh)
    monkeypatch.setattr(
        company_sync_module.shared_rate_limit_pause_controller,
        "wait_if_paused",
        _no_pause,
    )
    monkeypatch.setattr(
        company_sync_module.shared_rate_limit_pause_controller,
        "register_rate_limit_and_get_wait",
        _fixed_wait,
    )
    monkeypatch.setattr(company_sync_module.asyncio, "sleep", _fake_sleep)

    with pytest.raises(RuntimeError, match=r"Rate limit persisted for AAA .*cap=0.2s"):
        await service._run_company_fetch_with_retry(
            symbol="AAA",
            data_type=CompanyService.DATA_TYPE_OVERVIEW,
        )

    assert len(attempts) == 3
    assert sleeps == [0.1, 0.1]


@pytest.mark.asyncio
async def test_company_sync_pacer_respects_min_interval(monkeypatch):
    service = CompanyDataSyncService(company=CompanyService())
    service._sync_target_rpm = 120  # 0.5s min interval

    fake_clock = {"now": 0.0}
    sleep_calls = []

    def _fake_monotonic() -> float:
        return fake_clock["now"]

    async def _fake_sleep(seconds: float):
        sleep_calls.append(seconds)
        fake_clock["now"] += seconds

    monkeypatch.setattr(company_sync_module.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(company_sync_module.asyncio, "sleep", _fake_sleep)

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
    service = CompanyDataSyncService(company=CompanyService())
    calls = []

    async def _fake_run_company_fetch_with_retry(symbol: str, data_type: str) -> int:
        calls.append((symbol, data_type))
        return 0

    monkeypatch.setattr(service, "_run_company_fetch_with_retry", _fake_run_company_fetch_with_retry)

    await service._sync_symbol("AAA")

    assert [entry[1] for entry in calls] == list(service.DATA_TYPES)
    assert all(entry[0] == "AAA" for entry in calls)


@pytest.mark.asyncio
async def test_resolve_symbols_filter_merges_manual_and_index(monkeypatch):
    service = CompanyDataSyncService(company=CompanyService())

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
    service = CompanyDataSyncService(company=CompanyService())

    async def _fake_resolve_symbols_filter(symbols=None, index_symbol=None):
        return ["AAA"]

    async def _fake_run_sync(_symbols, quick_sync=False):
        assert quick_sync is False
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


def _build_retry_error(inner_error: BaseException) -> RetryError:
    future: Future = Future()
    future.set_exception(inner_error)
    return RetryError(future)


def test_fetch_subsidiaries_known_organ_code_retry_error_returns_empty(monkeypatch):
    service = CompanyService()
    api_circuit_breaker.reset()

    class _Client:
        def subsidiaries(self):
            raise _build_retry_error(KeyError("['organ_code'] not found in axis"))

    monkeypatch.setattr(service, "_build_company_client", lambda _symbol: _Client())

    records = service._fetch_subsidiaries_sync("AAA")

    assert records == []


def test_fetch_subsidiaries_non_matching_retry_error_still_raises(monkeypatch):
    service = CompanyService()
    api_circuit_breaker.reset()

    class _Client:
        def subsidiaries(self):
            raise _build_retry_error(KeyError("['unexpected_col'] not found in axis"))

    monkeypatch.setattr(service, "_build_company_client", lambda _symbol: _Client())

    with pytest.raises(RetryError):
        service._fetch_subsidiaries_sync("AAA")


@pytest.mark.asyncio
async def test_sync_symbol_keeps_success_when_subsidiaries_known_retry_error(monkeypatch):
    company = CompanyService()
    service = CompanyDataSyncService(company=company)
    api_circuit_breaker.reset()
    sync_status.clear_rate_limit()

    class _Client:
        def overview(self):
            return [{"symbol": "AAA"}]

        def shareholders(self):
            return [{"name": "holder"}]

        def officers(self):
            return [{"name": "officer"}]

        def subsidiaries(self):
            raise _build_retry_error(KeyError("['organ_code'] not found in axis"))

    upsert_calls = []

    async def _fake_get_cache_entry(*_args, **_kwargs):
        return None

    async def _fake_upsert_cache_entry(symbol: str, data_type: str, data):
        upsert_calls.append((symbol, data_type, data))

    async def _no_pace():
        return None

    async def _no_pause():
        return None

    monkeypatch.setattr(company, "_build_company_client", lambda _symbol: _Client())
    monkeypatch.setattr(company, "_get_cache_entry", _fake_get_cache_entry)
    monkeypatch.setattr(company, "_upsert_cache_entry", _fake_upsert_cache_entry)
    monkeypatch.setattr(service, "_acquire_sync_request_slot", _no_pace)
    monkeypatch.setattr(
        company_sync_module.shared_rate_limit_pause_controller,
        "wait_if_paused",
        _no_pause,
    )

    await service._sync_symbol("AAA")

    assert [entry[1] for entry in upsert_calls] == list(service.DATA_TYPES)
    assert all(entry[0] == "AAA" for entry in upsert_calls)
    assert upsert_calls[-1][1] == CompanyService.DATA_TYPE_SUBSIDIARIES
    assert upsert_calls[-1][2] == []
