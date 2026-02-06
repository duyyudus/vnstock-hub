from datetime import date, datetime

import pytest

from app.services.vnstock_service.history import HistoryService


def test_check_prices_staleness_allows_late_first_point_if_latest_is_fresh():
    service = HistoryService()
    stocks_data = {
        "AAA": [
            {"date": "2025-01-10", "close": 10.0},
            {"date": "2026-02-06", "close": 12.0},
        ]
    }

    is_stale = service._check_prices_staleness(
        stocks_data=stocks_data,
        requested_symbols=["AAA"],
        end_date=date(2026, 2, 6),
    )

    assert is_stale is False


def test_check_prices_staleness_marks_missing_symbol_as_stale():
    service = HistoryService()
    stocks_data = {
        "AAA": [{"date": "2026-02-06", "close": 12.0}],
    }

    is_stale = service._check_prices_staleness(
        stocks_data=stocks_data,
        requested_symbols=["AAA", "BBB"],
        end_date=date(2026, 2, 6),
    )

    assert is_stale is True


def test_check_prices_staleness_marks_old_latest_date_as_stale():
    service = HistoryService()
    stocks_data = {
        "AAA": [{"date": "2026-01-20", "close": 12.0}],
    }

    is_stale = service._check_prices_staleness(
        stocks_data=stocks_data,
        requested_symbols=["AAA"],
        end_date=date(2026, 2, 6),
    )

    assert is_stale is True


def test_resolve_backfill_window_for_covered_symbol_returns_none():
    service = HistoryService()
    target_start = date(2010, 1, 1)
    oldest_date = date(2010, 1, 15)

    start_date, end_date = service._resolve_backfill_window(oldest_date, target_start)

    assert start_date is None
    assert end_date is None


def test_resolve_backfill_window_for_gap_returns_chunk():
    service = HistoryService()
    target_start = date(2010, 1, 1)
    oldest_date = date(2020, 1, 1)

    start_date, end_date = service._resolve_backfill_window(oldest_date, target_start)

    assert start_date is not None
    assert end_date is not None
    assert start_date <= end_date
    assert end_date == date(2019, 12, 31)


def test_check_prices_historical_coverage_marks_late_start_as_missing():
    service = HistoryService()
    stocks_data = {
        "AAA": [
            {"date": "2021-01-08", "close": 10.0},
            {"date": "2026-02-06", "close": 12.0},
        ]
    }

    has_gap = service._check_prices_historical_coverage(
        stocks_data=stocks_data,
        requested_symbols=["AAA"],
        start_date=date(2019, 1, 1),
    )

    assert has_gap is True


def test_check_prices_historical_coverage_allows_small_offset_from_start():
    service = HistoryService()
    stocks_data = {
        "AAA": [
            {"date": "2019-01-18", "close": 10.0},
            {"date": "2026-02-06", "close": 12.0},
        ]
    }

    has_gap = service._check_prices_historical_coverage(
        stocks_data=stocks_data,
        requested_symbols=["AAA"],
        start_date=date(2019, 1, 1),
    )

    assert has_gap is False


@pytest.mark.asyncio
async def test_get_stocks_weekly_prices_triggers_sync_for_historical_gap(monkeypatch):
    service = HistoryService()
    triggered = {}

    async def _noop_schedule(*args, **kwargs):
        return None

    async def _fake_load_weekly_prices(_symbols, _start_date, _end_date):
        return {
            "AAA": [
                {"date": "2021-01-08", "close": 10.0},
                {"date": "2026-02-06", "close": 12.0},
            ]
        }

    async def _fake_company_names(_symbols):
        return {"AAA": "AAA Corp"}

    async def _fake_benchmarks(_start_date, _end_date):
        return {}

    async def _fake_trigger(symbols, start_date, end_date, force=False):
        triggered["symbols"] = symbols
        triggered["start_date"] = start_date
        triggered["end_date"] = end_date
        triggered["force"] = force
        return True

    monkeypatch.setattr(service, "_schedule_completeness_backfill_safe", _noop_schedule)
    monkeypatch.setattr(service, "_load_weekly_prices_from_db", _fake_load_weekly_prices)
    monkeypatch.setattr(service, "_load_benchmark_prices", _fake_benchmarks)
    monkeypatch.setattr(service, "_get_company_names", _fake_company_names)
    monkeypatch.setattr(service, "_trigger_price_history_sync", _fake_trigger)

    result = await service.get_stocks_weekly_prices(
        symbols=["AAA"],
        start_year=2019,
        include_benchmarks=True
    )

    assert result["is_stale"] is True
    assert result["is_syncing"] is True
    assert triggered["symbols"] == ["AAA"]
    assert triggered["start_date"] == date(2019, 1, 1)
    assert triggered["force"] is False


@pytest.mark.asyncio
async def test_get_stocks_weekly_prices_forces_sync_on_missing_symbol_data(monkeypatch):
    service = HistoryService()
    triggered = {}

    async def _noop_schedule(*args, **kwargs):
        return None

    async def _fake_load_weekly_prices(_symbols, _start_date, _end_date):
        return {}

    async def _fake_company_names(_symbols):
        return {"AAA": "AAA Corp"}

    async def _fake_benchmarks(_start_date, _end_date):
        return {}

    async def _fake_trigger(symbols, start_date, end_date, force=False):
        triggered["symbols"] = symbols
        triggered["start_date"] = start_date
        triggered["end_date"] = end_date
        triggered["force"] = force
        return True

    monkeypatch.setattr(service, "_schedule_completeness_backfill_safe", _noop_schedule)
    monkeypatch.setattr(service, "_load_weekly_prices_from_db", _fake_load_weekly_prices)
    monkeypatch.setattr(service, "_load_benchmark_prices", _fake_benchmarks)
    monkeypatch.setattr(service, "_get_company_names", _fake_company_names)
    monkeypatch.setattr(service, "_trigger_price_history_sync", _fake_trigger)

    result = await service.get_stocks_weekly_prices(
        symbols=["AAA"],
        start_year=2019,
        include_benchmarks=True
    )

    assert result["is_stale"] is True
    assert result["is_syncing"] is True
    assert triggered["symbols"] == ["AAA"]
    assert triggered["start_date"] == date(2019, 1, 1)
    assert triggered["force"] is True


@pytest.mark.asyncio
async def test_trigger_price_history_sync_bypasses_cooldown_for_older_start_date(monkeypatch):
    service = HistoryService()
    recorded = {}

    async def _fake_get_state(_symbols):
        return {
            "AAA": {
                "last_attempt_at": datetime.utcnow(),
                "last_attempt_start_date": date(2023, 1, 1),
            }
        }

    async def _fake_set_state(symbols, attempted_at, attempted_start_date):
        recorded["symbols"] = symbols
        recorded["attempted_at"] = attempted_at
        recorded["attempted_start_date"] = attempted_start_date

    async def _fake_sync(symbols, start_date, end_date):
        for symbol in symbols:
            service._weekly_prices_syncing_symbols.discard(symbol)

    monkeypatch.setattr(service, "_get_weekly_sync_cooldown_state", _fake_get_state)
    monkeypatch.setattr(service, "_set_weekly_sync_cooldown_state", _fake_set_state)
    monkeypatch.setattr(service, "_sync_price_history_background", _fake_sync)

    triggered = await service._trigger_price_history_sync(
        symbols=["AAA"],
        start_date=date(2019, 1, 1),
        end_date=date(2026, 2, 6),
        force=False
    )

    if service._weekly_prices_sync_task is not None:
        await service._weekly_prices_sync_task

    assert triggered is True
    assert recorded["symbols"] == ["AAA"]
    assert recorded["attempted_start_date"] == date(2019, 1, 1)


@pytest.mark.asyncio
async def test_trigger_price_history_sync_respects_cooldown_for_newer_start_date(monkeypatch):
    service = HistoryService()

    async def _fake_get_state(_symbols):
        return {
            "AAA": {
                "last_attempt_at": datetime.utcnow(),
                "last_attempt_start_date": date(2019, 1, 1),
            }
        }

    async def _noop_set_state(symbols, attempted_at, attempted_start_date):
        return None

    monkeypatch.setattr(service, "_get_weekly_sync_cooldown_state", _fake_get_state)
    monkeypatch.setattr(service, "_set_weekly_sync_cooldown_state", _noop_set_state)

    triggered = await service._trigger_price_history_sync(
        symbols=["AAA"],
        start_date=date(2023, 1, 1),
        end_date=date(2026, 2, 6),
        force=False
    )

    assert triggered is False
