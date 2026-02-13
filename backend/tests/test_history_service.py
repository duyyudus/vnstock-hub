import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import app.services.vnstock_service.history as history_module
from app.core.config import settings
from app.db.models import StockDailyPrice, StockCompany, StockPriceSyncState
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
async def test_get_stocks_weekly_prices_does_not_trigger_request_path_sync_for_historical_gap(monkeypatch):
    service = HistoryService()

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

    monkeypatch.setattr(service, "_load_weekly_prices_from_db", _fake_load_weekly_prices)
    monkeypatch.setattr(service, "_load_benchmark_prices", _fake_benchmarks)
    monkeypatch.setattr(service, "_get_company_names", _fake_company_names)

    result = await service.get_stocks_weekly_prices(
        symbols=["AAA"],
        start_year=2019,
        include_benchmarks=True
    )

    assert result["is_stale"] is True
    assert result["is_syncing"] is False


@pytest.mark.asyncio
async def test_get_stocks_weekly_prices_missing_symbol_data_triggers_latest_sync(monkeypatch):
    service = HistoryService()
    trigger_calls = []

    async def _fake_load_weekly_prices(_symbols, _start_date, _end_date):
        return {}

    async def _fake_company_names(_symbols):
        return {"AAA": "AAA Corp"}

    async def _fake_benchmarks(_start_date, _end_date):
        return {}

    async def _fake_trigger(symbols, start_date, end_date, force=False):
        trigger_calls.append({
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "force": force,
        })
        return True

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
    assert len(trigger_calls) == 1
    assert trigger_calls[0]["symbols"] == ["AAA"]


@pytest.mark.asyncio
async def test_get_stocks_weekly_prices_triggers_request_path_sync_when_latest_is_stale(monkeypatch):
    service = HistoryService()
    trigger_calls = []

    async def _fake_load_weekly_prices(_symbols, _start_date, _end_date):
        return {
            "AAA": [
                {"date": "2025-01-03", "close": 10.0},
                {"date": "2025-01-10", "close": 10.5},
            ]
        }

    async def _fake_company_names(_symbols):
        return {"AAA": "AAA Corp"}

    async def _fake_benchmarks(_start_date, _end_date):
        return {}

    async def _fake_trigger(symbols, start_date, end_date, force=False):
        trigger_calls.append({
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "force": force,
        })
        return True

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
    assert len(trigger_calls) == 1
    assert trigger_calls[0]["symbols"] == ["AAA"]


@pytest.mark.asyncio
async def test_get_stocks_weekly_prices_does_not_trigger_request_path_sync_for_historical_gap_only(monkeypatch):
    service = HistoryService()

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

    async def _unexpected_trigger(*_args, **_kwargs):
        raise AssertionError("latest-stale sync should not trigger for historical-gap-only case")

    monkeypatch.setattr(service, "_load_weekly_prices_from_db", _fake_load_weekly_prices)
    monkeypatch.setattr(service, "_load_benchmark_prices", _fake_benchmarks)
    monkeypatch.setattr(service, "_get_company_names", _fake_company_names)
    monkeypatch.setattr(service, "_trigger_price_history_sync", _unexpected_trigger)

    result = await service.get_stocks_weekly_prices(
        symbols=["AAA"],
        start_year=2019,
        include_benchmarks=True
    )

    assert result["is_stale"] is True
    assert result["is_syncing"] is False


@pytest.mark.asyncio
async def test_trigger_price_history_sync_respects_db_cooldown(monkeypatch, db_session):
    service = HistoryService()
    now = datetime.utcnow()

    db_session.add(
        StockPriceSyncState(
            symbol="AAA",
            weekly_sync_last_attempt_at=now,
        )
    )
    await db_session.commit()

    async def _unexpected_set_state(*_args, **_kwargs):
        raise AssertionError("cooldown write should not happen when request is blocked")

    monkeypatch.setattr(service, "_set_weekly_sync_cooldown_state", _unexpected_set_state)

    triggered = await service._trigger_price_history_sync(
        symbols=["AAA"],
        start_date=date(2026, 1, 1),
        end_date=date(2026, 2, 6),
        force=False
    )

    assert triggered is False
    assert "AAA" not in service._weekly_prices_syncing_symbols


@pytest.mark.asyncio
async def test_weekly_sync_cooldown_state_persists_across_service_reinstantiation(db_session):
    service_a = HistoryService()
    attempted_at = datetime.utcnow()

    await service_a._set_weekly_sync_cooldown_state(
        symbols=["AAA"],
        attempted_at=attempted_at,
    )

    service_b = HistoryService()
    cooldown_state = await service_b._get_weekly_sync_cooldown_state(["AAA"])

    assert "AAA" in cooldown_state
    assert cooldown_state["AAA"] is not None


@pytest.mark.asyncio
async def test_stop_background_workers_cancels_weekly_sync_task():
    service = HistoryService()

    async def _never_ending_sync():
        await asyncio.sleep(60)

    service._weekly_prices_sync_task = asyncio.create_task(_never_ending_sync())
    service._weekly_prices_syncing_symbols.add("AAA")

    await service.stop_background_workers()

    assert service._weekly_prices_sync_task is None
    assert service._weekly_prices_syncing_symbols == set()


def test_normalize_price_history_payload_deduplicates_by_symbol_date():
    service = HistoryService()
    hist = pd.DataFrame(
        [
            {
                "time": "2021-10-29",
                "open": 9.1,
                "high": 9.3,
                "low": 9.0,
                "close": 9.0,
                "volume": 1000,
            },
            {
                "time": "2021-10-29",
                "open": 9.2,
                "high": 9.4,
                "low": 9.1,
                "close": 9.1,
                "volume": 1100,
            },
            {
                "time": "2021-11-01",
                "open": None,
                "high": None,
                "low": None,
                "close": 9.2,
                "volume": None,
            },
            {
                "time": "bad-date",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1,
            },
            {
                "time": "2021-11-02",
                "open": 9.2,
                "high": 9.3,
                "low": 9.1,
                "close": None,
                "volume": 1200,
            },
        ]
    )
    created_at = datetime(2026, 2, 8, 11, 17, 21)

    payload, min_date, max_date = service._normalize_price_history_payload(
        symbol="inc",
        hist=hist,
        created_at=created_at,
    )

    assert len(payload) == 2
    assert min_date == date(2021, 10, 29)
    assert max_date == date(2021, 11, 1)
    assert payload[0]["symbol"] == "INC"
    assert payload[0]["date"] == date(2021, 10, 29)
    assert payload[0]["close"] == 9.1
    assert payload[0]["volume"] == 1100
    assert payload[1]["date"] == date(2021, 11, 1)
    assert payload[1]["open"] is None
    assert payload[1]["high"] is None
    assert payload[1]["low"] is None
    assert payload[1]["volume"] is None
    assert payload[1]["created_at"] == created_at


def test_get_symbol_sync_lock_reuses_lock_for_same_symbol():
    service = HistoryService()

    lock_a = service._get_symbol_sync_lock("inc")
    lock_b = service._get_symbol_sync_lock("INC")
    lock_c = service._get_symbol_sync_lock("inc1")

    assert lock_a is lock_b
    assert lock_a is lock_c


def test_upsert_stock_price_history_rolls_back_session_on_failure(monkeypatch):
    service = HistoryService()
    hist = pd.DataFrame(
        [
            {
                "time": "2021-10-29",
                "open": 9.1,
                "high": 9.3,
                "low": 9.0,
                "close": 9.0,
                "volume": 1000,
            }
        ]
    )

    class FailingSession:
        def __init__(self):
            self.rollback_calls = 0

        def execute(self, _stmt):
            raise RuntimeError("db exploded")

        def commit(self):
            raise AssertionError("commit should not be called on failure")

        def rollback(self):
            self.rollback_calls += 1

    failing_session = FailingSession()
    monkeypatch.setattr(history_module, "retry_with_backoff", lambda _func, max_retries=2: hist)

    count = service._upsert_stock_price_history(
        symbol="INC",
        start_date=date(2021, 10, 1),
        end_date=date(2021, 11, 30),
        session=failing_session,
    )

    assert count == 0
    assert failing_session.rollback_calls == 1


def test_upsert_stock_price_history_raises_when_raise_on_error_enabled(monkeypatch):
    service = HistoryService()
    hist = pd.DataFrame(
        [
            {
                "time": "2021-10-29",
                "open": 9.1,
                "high": 9.3,
                "low": 9.0,
                "close": 9.0,
                "volume": 1000,
            }
        ]
    )

    class FailingSession:
        def __init__(self):
            self.rollback_calls = 0

        def execute(self, _stmt):
            raise RuntimeError("db exploded")

        def commit(self):
            raise AssertionError("commit should not be called on failure")

        def rollback(self):
            self.rollback_calls += 1

    failing_session = FailingSession()
    monkeypatch.setattr(history_module, "retry_with_backoff", lambda _func, max_retries=2: hist)

    with pytest.raises(RuntimeError, match="db exploded"):
        service._upsert_stock_price_history(
            symbol="INC",
            start_date=date(2021, 10, 1),
            end_date=date(2021, 11, 30),
            session=failing_session,
            raise_on_error=True,
        )

    assert failing_session.rollback_calls == 1


def test_upsert_stock_price_history_serializes_same_symbol_calls(monkeypatch):
    service = HistoryService()
    hist = pd.DataFrame(
        [
            {
                "time": "2021-10-29",
                "open": 9.1,
                "high": 9.3,
                "low": 9.0,
                "close": 9.0,
                "volume": 1000,
            }
        ]
    )

    class RecordingSession:
        def __init__(self):
            self.inflight = 0
            self.max_inflight = 0
            self.guard = threading.Lock()

        def execute(self, _stmt):
            with self.guard:
                self.inflight += 1
                self.max_inflight = max(self.max_inflight, self.inflight)
            time.sleep(0.05)
            with self.guard:
                self.inflight -= 1

        def commit(self):
            return None

        def rollback(self):
            return None

    session = RecordingSession()
    monkeypatch.setattr(history_module, "retry_with_backoff", lambda _func, max_retries=2: hist)

    def run_one_call():
        return service._upsert_stock_price_history(
            symbol="INC",
            start_date=date(2021, 10, 1),
            end_date=date(2021, 11, 30),
            session=session,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run_one_call)
        second = executor.submit(run_one_call)
        assert first.result() == 1
        assert second.result() == 1

    assert session.max_inflight == 1


def test_fetch_and_cache_history_sync_rolls_back_shared_session_on_symbol_error(monkeypatch):
    service = HistoryService()

    class SharedSession:
        def __init__(self):
            self.rollback_calls = 0

        def rollback(self):
            self.rollback_calls += 1

    shared_session = SharedSession()
    monkeypatch.setattr(history_module.api_circuit_breaker, "can_proceed", lambda: True)

    def _raise_upsert(*_args, **_kwargs):
        raise RuntimeError("upsert failed")

    monkeypatch.setattr(service, "_upsert_stock_price_history", _raise_upsert)

    service._fetch_and_cache_history_sync(shared_session, ["INC", "AAA"])

    assert shared_session.rollback_calls == 2


def test_fetch_volume_history_sync_uses_db_calendar_window_and_normalizes_symbol(monkeypatch):
    service = HistoryService()
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    StockCompany.__table__.create(bind=engine, checkfirst=True)
    StockDailyPrice.__table__.create(bind=engine, checkfirst=True)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 2, 11, 10, 0, 0)

    monkeypatch.setattr(history_module, "datetime", FixedDateTime)
    monkeypatch.setattr(history_module, "get_sync_engine", lambda: engine)

    with Session(engine) as session:
        session.add(StockCompany(symbol="TCB", company_name="Techcombank"))
        session.add(
            StockDailyPrice(
                symbol="TCB",
                date=date(2025, 11, 13),
                open=20.0,
                high=21.0,
                low=19.5,
                close=20.5,
                volume=100_000,
            )
        )  # Outside 90-day calendar window
        session.add(
            StockDailyPrice(
                symbol="TCB",
                date=date(2025, 11, 14),
                open=20.5,
                high=21.5,
                low=20.0,
                close=21.11,
                volume=1_000_000,
            )
        )
        session.add(
            StockDailyPrice(
                symbol="TCB",
                date=date(2026, 2, 10),
                open=22.0,
                high=22.2,
                low=21.7,
                close=22.0,
                volume=None,
            )
        )
        session.commit()

    result = service._fetch_volume_history_sync("tcbx", 90)

    assert result["symbol"] == "TCB"
    assert result["company_name"] == "Techcombank"
    assert [point["date"] for point in result["data"]] == ["2025-11-14", "2026-02-10"]
    assert result["data"][0]["volume"] == 1_000_000
    assert result["data"][0]["value"] == 21.11
    assert result["data"][1]["volume"] == 0
    assert result["data"][1]["value"] is None


def test_fetch_price_history_sync_uses_db_calendar_window_and_normalizes_symbol(monkeypatch):
    service = HistoryService()
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    StockCompany.__table__.create(bind=engine, checkfirst=True)
    StockDailyPrice.__table__.create(bind=engine, checkfirst=True)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 2, 11, 10, 0, 0)

    monkeypatch.setattr(history_module, "datetime", FixedDateTime)
    monkeypatch.setattr(history_module, "get_sync_engine", lambda: engine)

    with Session(engine) as session:
        session.add(StockCompany(symbol="TCB", company_name="Techcombank"))
        session.add(
            StockDailyPrice(
                symbol="TCB",
                date=date(2025, 11, 13),
                open=20.0,
                high=21.0,
                low=19.5,
                close=20.5,
                volume=100_000,
            )
        )  # Outside 90-day calendar window
        session.add(
            StockDailyPrice(
                symbol="TCB",
                date=date(2025, 11, 14),
                open=20.5,
                high=21.5,
                low=20.0,
                close=21.11,
                volume=1_000_000,
            )
        )
        session.add(
            StockDailyPrice(
                symbol="TCB",
                date=date(2026, 2, 10),
                open=22.0,
                high=22.2,
                low=21.7,
                close=22.0,
                volume=None,
            )
        )
        session.commit()

    result = service._fetch_price_history_sync("tcbx", 90)

    assert result["symbol"] == "TCB"
    assert result["company_name"] == "Techcombank"
    assert [point["date"] for point in result["data"]] == ["2025-11-14", "2026-02-10"]
    assert result["data"][0]["close"] == 21110.0
    assert result["data"][1]["close"] == 22000.0


@pytest.mark.asyncio
async def test_get_volume_history_runs_request_path_sync_and_returns_metadata(monkeypatch):
    service = HistoryService()
    sync_calls = []

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 2, 11)

    class InlineLoop:
        async def run_in_executor(self, _executor, func, *args):
            return func(*args)

    async def _fake_request_sync(symbol, start_date, end_date, timeout_seconds):
        sync_calls.append({
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "timeout_seconds": timeout_seconds,
        })
        return {
            "sync_performed": True,
            "sync_timed_out": False,
            "sync_error": None,
            "updated_through": "2026-02-11",
            "repaired_missing_dates": 3,
        }

    monkeypatch.setattr(history_module, "date", FixedDate)
    monkeypatch.setattr(history_module.asyncio, "get_event_loop", lambda: InlineLoop())
    monkeypatch.setattr(service, "_fetch_volume_history_sync", lambda _symbol, _days: {
        "symbol": "TCB",
        "company_name": "Techcombank",
        "data": [{"date": "2026-02-10", "volume": 1200, "value": 20.2}],
    })
    service.set_on_demand_history_sync_handler(_fake_request_sync)

    result = await service.get_volume_history("tcbx", 90)

    assert result["symbol"] == "TCB"
    assert len(sync_calls) == 1
    assert sync_calls[0]["symbol"] == "TCB"
    assert sync_calls[0]["start_date"] == date(2025, 11, 14)
    assert sync_calls[0]["end_date"] == date(2026, 2, 11)
    assert result["sync_performed"] is True
    assert result["sync_timed_out"] is False
    assert result["sync_error"] is None
    assert result["updated_through"] == "2026-02-11"
    assert result["repaired_missing_dates"] == 3


@pytest.mark.asyncio
async def test_get_price_history_runs_request_path_sync_and_returns_metadata(monkeypatch):
    service = HistoryService()
    sync_calls = []

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 2, 11)

    class InlineLoop:
        async def run_in_executor(self, _executor, func, *args):
            return func(*args)

    async def _fake_request_sync(symbol, start_date, end_date, timeout_seconds):
        sync_calls.append({
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "timeout_seconds": timeout_seconds,
        })
        return {
            "sync_performed": True,
            "sync_timed_out": False,
            "sync_error": None,
            "updated_through": "2026-02-11",
            "repaired_missing_dates": 1,
        }

    monkeypatch.setattr(history_module, "date", FixedDate)
    monkeypatch.setattr(history_module.asyncio, "get_event_loop", lambda: InlineLoop())
    monkeypatch.setattr(service, "_fetch_price_history_sync", lambda _symbol, _days: {
        "symbol": "TCB",
        "company_name": "Techcombank",
        "data": [{"date": "2026-02-10", "close": 22000.0}],
    })
    service.set_on_demand_history_sync_handler(_fake_request_sync)

    result = await service.get_price_history("tcbx", 90)

    assert result["symbol"] == "TCB"
    assert len(sync_calls) == 1
    assert sync_calls[0]["symbol"] == "TCB"
    assert sync_calls[0]["start_date"] == date(2025, 11, 14)
    assert sync_calls[0]["end_date"] == date(2026, 2, 11)
    assert result["sync_performed"] is True
    assert result["sync_timed_out"] is False
    assert result["sync_error"] is None
    assert result["updated_through"] == "2026-02-11"
    assert result["repaired_missing_dates"] == 1


@pytest.mark.asyncio
async def test_get_price_history_returns_timeout_metadata_when_request_sync_times_out(monkeypatch):
    service = HistoryService()

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 2, 11)

    class InlineLoop:
        async def run_in_executor(self, _executor, func, *args):
            return func(*args)

    async def _fake_request_sync(_symbol, _start_date, _end_date, _timeout_seconds):
        return {
            "sync_performed": False,
            "sync_timed_out": True,
            "sync_error": None,
            "updated_through": "2026-02-10",
            "repaired_missing_dates": 0,
        }

    monkeypatch.setattr(history_module, "date", FixedDate)
    monkeypatch.setattr(history_module.asyncio, "get_event_loop", lambda: InlineLoop())
    monkeypatch.setattr(service, "_fetch_price_history_sync", lambda _symbol, _days: {
        "symbol": "TCB",
        "company_name": "Techcombank",
        "data": [{"date": "2026-02-10", "close": 22000.0}],
    })
    service.set_on_demand_history_sync_handler(_fake_request_sync)

    result = await service.get_price_history("tcbx", 90)

    assert result["sync_performed"] is False
    assert result["sync_timed_out"] is True
    assert result["sync_error"] is None
    assert result["updated_through"] == "2026-02-10"
    assert result["repaired_missing_dates"] == 0


@pytest.mark.asyncio
async def test_get_volume_history_sets_sync_error_when_request_sync_fails(monkeypatch):
    service = HistoryService()

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 2, 11)

    class InlineLoop:
        async def run_in_executor(self, _executor, func, *args):
            return func(*args)

    async def _failing_request_sync(_symbol, _start_date, _end_date, _timeout_seconds):
        raise RuntimeError("request sync exploded")

    monkeypatch.setattr(history_module, "date", FixedDate)
    monkeypatch.setattr(history_module.asyncio, "get_event_loop", lambda: InlineLoop())
    monkeypatch.setattr(service, "_fetch_volume_history_sync", lambda _symbol, _days: {
        "symbol": "TCB",
        "company_name": "Techcombank",
        "data": [{"date": "2026-02-10", "volume": 1200, "value": 20.2}],
    })
    service.set_on_demand_history_sync_handler(_failing_request_sync)

    result = await service.get_volume_history("tcbx", 90)

    assert result["sync_performed"] is False
    assert result["sync_timed_out"] is False
    assert result["sync_error"] == "request sync exploded"
    assert result["updated_through"] is None
    assert result["repaired_missing_dates"] == 0


def test_upsert_stock_price_history_updates_conflicts_in_postgres(monkeypatch):
    database_url = settings.database_url
    if not database_url:
        pytest.skip("DATABASE_URL is not set")
    if "postgresql" not in database_url:
        pytest.skip("DATABASE_URL is not PostgreSQL")

    sync_url = database_url.replace("+asyncpg", "+psycopg2")
    engine = create_engine(sync_url)
    service = HistoryService()
    symbol = "INC"

    try:
        StockDailyPrice.__table__.create(bind=engine, checkfirst=True)
        with Session(engine) as session:
            session.query(StockDailyPrice).filter(StockDailyPrice.symbol == symbol).delete()
            session.commit()

            original_created_at = datetime(2020, 1, 1, 0, 0, 0)
            session.add(
                StockDailyPrice(
                    symbol=symbol,
                    date=date(2021, 10, 29),
                    open=8.0,
                    high=8.2,
                    low=7.9,
                    close=8.1,
                    volume=900,
                    created_at=original_created_at,
                )
            )
            session.commit()

            hist = pd.DataFrame(
                [
                    {
                        "time": "2021-10-29",
                        "open": 9.16,
                        "high": 9.32,
                        "low": 9.0,
                        "close": 9.0,
                        "volume": 1250,
                    },
                    {
                        "time": "2021-11-01",
                        "open": 9.3,
                        "high": 9.4,
                        "low": 9.2,
                        "close": 9.35,
                        "volume": 1400,
                    },
                ]
            )
            monkeypatch.setattr(history_module, "retry_with_backoff", lambda _func, max_retries=2: hist)

            synced = service._upsert_stock_price_history(
                symbol=symbol,
                start_date=date(2021, 10, 1),
                end_date=date(2021, 11, 30),
                session=session,
            )
            assert synced == 2

            rows = session.execute(
                select(StockDailyPrice)
                .where(StockDailyPrice.symbol == symbol)
                .order_by(StockDailyPrice.date.asc())
            ).scalars().all()

            assert len(rows) == 2
            assert rows[0].date == date(2021, 10, 29)
            assert rows[0].close == 9.0
            assert rows[0].volume == 1250
            assert rows[0].created_at == original_created_at
            assert rows[1].date == date(2021, 11, 1)
            assert rows[1].close == 9.35
            assert rows[1].volume == 1400
    finally:
        engine.dispose()
