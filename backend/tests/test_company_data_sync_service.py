import asyncio
import sys
from types import SimpleNamespace

import pytest

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

    async def _fake_sync_symbol(_symbol: str, force_refresh: bool = False):
        nonlocal in_flight, max_in_flight
        assert force_refresh is False
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
    assert runtime.failed_tickers == []
    assert runtime.progress == 1.0
    assert max_in_flight > 1
    assert max_in_flight <= service._sync_max_workers


@pytest.mark.asyncio
async def test_company_sync_tracks_failed_tickers_and_resets(monkeypatch):
    service = CompanyDataSyncService(company=CompanyService())
    service._sync_max_workers = 1
    service._operation_worker_semaphore = asyncio.Semaphore(service._sync_max_workers)

    symbols = ["AAA", "BBB", "CCC"]

    async def _fake_build_symbol_universe(_symbols=None):
        return symbols

    async def _fail_on_bbb(symbol: str, force_refresh: bool = False):
        assert force_refresh is False
        if symbol == "BBB":
            raise RuntimeError("sync failed")

    async def _all_success(_symbol: str, force_refresh: bool = False):
        assert force_refresh is False
        return None

    monkeypatch.setattr(service, "_build_symbol_universe", _fake_build_symbol_universe)
    monkeypatch.setattr(service, "_sync_symbol", _fail_on_bbb)

    await service._run_sync(symbols=None)

    runtime = sync_status.company_sync
    assert runtime.failed_symbols == 1
    assert runtime.failed_tickers == ["BBB"]

    monkeypatch.setattr(service, "_sync_symbol", _all_success)

    await service._run_sync(symbols=None)

    runtime = sync_status.company_sync
    assert runtime.failed_symbols == 0
    assert runtime.failed_tickers == []


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

    async def _fake_sync_symbol(symbol: str, force_refresh: bool = False):
        assert force_refresh is False
        processed.append(symbol)

    monkeypatch.setattr(service, "_build_symbol_universe", _fake_build_symbol_universe)
    monkeypatch.setattr(service, "_filter_quick_sync_symbols", _unexpected_filter)
    monkeypatch.setattr(service, "_sync_symbol", _fake_sync_symbol)

    await service._run_sync(symbols=None, quick_sync=False)

    assert processed == ["AAA", "BBB"]


@pytest.mark.asyncio
async def test_company_sync_force_refresh_flows_to_symbol_sync(monkeypatch):
    service = CompanyDataSyncService(company=CompanyService())

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
        force_refresh=True,
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
            force_refresh=True,
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
            force_refresh=True,
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
            force_refresh=True,
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

    async def _fake_run_company_fetch_with_retry(
        symbol: str,
        data_type: str,
        force_refresh: bool = False,
    ) -> int:
        assert force_refresh is False
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


def test_company_fetch_symbols_for_vnall_uses_kbs_group(monkeypatch):
    service = CompanyDataSyncService(company=CompanyService())
    calls = []

    class FakeSeries:
        empty = False

        def tolist(self):
            return ["AAA", "ACB"]

    class FakeListing:
        def __init__(self, source: str):
            calls.append(("source", source))

        def symbols_by_group(self, group_code: str):
            calls.append(("group", group_code))
            return FakeSeries()

    monkeypatch.setitem(sys.modules, "vnstock", SimpleNamespace(Listing=FakeListing))

    symbols = service._fetch_symbols_for_index_sync("VNALL")

    assert symbols == ["AAA", "ACB"]
    assert calls == [("source", "KBS"), ("group", "VNALL")]


@pytest.mark.asyncio
async def test_run_sync_force_restart(monkeypatch):
    service = CompanyDataSyncService(company=CompanyService())

    async def _fake_resolve_symbols_filter(symbols=None, index_symbol=None):
        return ["AAA"]

    async def _fake_run_sync(_symbols, quick_sync=False, force_refresh=False):
        assert quick_sync is False
        assert force_refresh is False
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
def test_company_service_fetchers_preserve_kbs_native_payloads(monkeypatch):
    service = CompanyService()
    api_circuit_breaker.reset()

    class _Client:
        def overview(self):
            return [{"symbol": "AAA", "business_model": "Banking", "outstanding_shares": 123}]

        def shareholders(self):
            return [{"name": "State", "ownership_percentage": 74.8, "shares_owned": 100}]

        def officers(self):
            return [{"name": "Alice", "position": "CEO", "position_en": "Chief Executive Officer"}]

        def subsidiaries(self):
            return [{"name": "AAA Leasing", "ownership_percent": 60.0, "type": "công ty con"}]

        def ownership(self):
            return [{"owner_type": "State", "ownership_percentage": 74.8, "shares_owned": 100}]

        def capital_history(self):
            return [{"date": "2025-01-01", "charter_capital": 1000, "currency": "VND"}]

        def news(self):
            return [{"head": "corp", "article_id": 1, "title": "Headline", "publish_time": "2026-04-19", "url": "https://example.com/news"}]

        def events(self):
            return [{"event_name": "AGM", "event_date": "2026-05-01"}]

        def insider_trading(self):
            return [{"person_name": "Alice", "action": "Buy"}]

    monkeypatch.setattr(service, "_build_company_client", lambda _symbol: _Client())

    assert service._fetch_company_overview_sync("AAA") == [{"symbol": "AAA", "business_model": "Banking", "outstanding_shares": 123}]
    assert service._fetch_shareholders_sync("AAA") == [{"name": "State", "ownership_percentage": 74.8, "shares_owned": 100}]
    assert service._fetch_officers_sync("AAA") == [{"name": "Alice", "position": "CEO", "position_en": "Chief Executive Officer"}]
    assert service._fetch_subsidiaries_sync("AAA") == [{"name": "AAA Leasing", "ownership_percent": 60.0, "type": "công ty con"}]
    assert service._fetch_ownership_sync("AAA") == [{"owner_type": "State", "ownership_percentage": 74.8, "shares_owned": 100}]
    assert service._fetch_capital_history_sync("AAA") == [{"date": "2025-01-01", "charter_capital": 1000, "currency": "VND"}]
    assert service._fetch_news_sync("AAA") == [{"head": "corp", "article_id": 1, "title": "Headline", "publish_time": "2026-04-19", "url": "https://example.com/news"}]
    assert service._fetch_events_sync("AAA") == [{"event_name": "AGM", "event_date": "2026-05-01"}]
    assert service._fetch_insider_trading_sync("AAA") == [{"person_name": "Alice", "action": "Buy"}]


@pytest.mark.asyncio
async def test_sync_symbol_upserts_all_kbs_company_datasets(monkeypatch):
    company = CompanyService()
    service = CompanyDataSyncService(company=company)
    api_circuit_breaker.reset()
    sync_status.clear_rate_limit()

    class _Client:
        def overview(self):
            return [{"symbol": "AAA", "business_model": "Banking"}]

        def shareholders(self):
            return [{"name": "holder", "ownership_percentage": 60.0}]

        def officers(self):
            return [{"name": "officer", "position": "CEO"}]

        def subsidiaries(self):
            return [{"name": "AAA Leasing", "ownership_percent": 60.0, "type": "công ty con"}]

        def ownership(self):
            return [{"owner_type": "State", "ownership_percentage": 60.0}]

        def capital_history(self):
            return [{"date": "2025-01-01", "charter_capital": 1000, "currency": "VND"}]

        def news(self):
            return [{"head": "corp", "article_id": 1}]

        def events(self):
            return [{"event_name": "AGM"}]

        def insider_trading(self):
            return [{"person_name": "Alice", "action": "Buy"}]

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
    payloads = {data_type: data for _, data_type, data in upsert_calls}
    assert payloads[CompanyService.DATA_TYPE_OVERVIEW] == [{"symbol": "AAA", "business_model": "Banking"}]
    assert payloads[CompanyService.DATA_TYPE_SHAREHOLDERS] == [{"name": "holder", "ownership_percentage": 60.0}]
    assert payloads[CompanyService.DATA_TYPE_OFFICERS] == [{"name": "officer", "position": "CEO"}]
    assert payloads[CompanyService.DATA_TYPE_SUBSIDIARIES] == [{"name": "AAA Leasing", "ownership_percent": 60.0, "type": "công ty con"}]
    assert payloads[CompanyService.DATA_TYPE_OWNERSHIP] == [{"owner_type": "State", "ownership_percentage": 60.0}]
    assert payloads[CompanyService.DATA_TYPE_CAPITAL_HISTORY] == [{"date": "2025-01-01", "charter_capital": 1000, "currency": "VND"}]
    assert payloads[CompanyService.DATA_TYPE_NEWS] == [{"head": "corp", "article_id": 1}]
    assert payloads[CompanyService.DATA_TYPE_EVENTS] == [{"event_name": "AGM"}]
    assert payloads[CompanyService.DATA_TYPE_INSIDER_TRADING] == [{"person_name": "Alice", "action": "Buy"}]
