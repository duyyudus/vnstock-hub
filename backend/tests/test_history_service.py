import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import app.services.vnstock_service.history as history_module
from app.db.models import StockDailyPrice
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
async def test_get_stocks_weekly_prices_missing_symbol_data_keeps_sync_disabled(monkeypatch):
    service = HistoryService()

    async def _fake_load_weekly_prices(_symbols, _start_date, _end_date):
        return {}

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


def test_upsert_stock_price_history_updates_conflicts_in_postgres(monkeypatch):
    test_database_url = os.getenv("TEST_DATABASE_URL")
    if not test_database_url:
        pytest.skip("TEST_DATABASE_URL is not set")
    if "postgresql" not in test_database_url:
        pytest.skip("TEST_DATABASE_URL is not PostgreSQL")

    sync_url = test_database_url.replace("+asyncpg", "+psycopg2")
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
