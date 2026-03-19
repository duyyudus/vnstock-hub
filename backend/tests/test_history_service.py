import asyncio
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from types import ModuleType
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import app.services.vnstock_service.history as history_module
from app.core.config import settings
from app.db.models import StockDailyHistory, StockCompany, StockHistorySyncState
from app.services.sync_status import sync_status
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


@pytest.mark.asyncio
async def test_get_stocks_weekly_prices_does_not_mark_stale_for_historical_gap_only(monkeypatch):
    service = HistoryService()

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 2, 6)

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
        raise AssertionError("latest-only staleness should not trigger for historical-gap-only case")

    monkeypatch.setattr(service, "_load_weekly_prices_from_db", _fake_load_weekly_prices)
    monkeypatch.setattr(service, "_load_benchmark_prices", _fake_benchmarks)
    monkeypatch.setattr(service, "_get_company_names", _fake_company_names)
    monkeypatch.setattr(service, "_trigger_price_history_sync", _unexpected_trigger)
    monkeypatch.setattr(history_module, "date", FixedDate)

    result = await service.get_stocks_weekly_prices(
        symbols=["AAA"],
        start_year=2019,
        include_benchmarks=True
    )

    assert result["is_stale"] is False
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
async def test_trigger_price_history_sync_respects_db_cooldown(monkeypatch, db_session):
    service = HistoryService()
    now = datetime.utcnow()

    db_session.add(
        StockHistorySyncState(
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
async def test_trigger_price_history_sync_delegates_to_history_sync_handler(monkeypatch):
    service = HistoryService()
    trigger_calls = []

    async def _fake_delegate(symbols):
        trigger_calls.append(symbols)
        return {
            "started": False,
            "state": "running",
        }

    service.set_weekly_history_sync_trigger_handler(_fake_delegate)

    triggered = await service._trigger_price_history_sync(
        symbols=["aaa", "AAA"],
        start_date=date(2026, 1, 1),
        end_date=date(2026, 2, 6),
        force=False,
    )

    assert triggered is True
    assert trigger_calls == [["AAA"]]


@pytest.mark.asyncio
async def test_get_stocks_weekly_prices_reports_db_syncing_status(monkeypatch, db_session):
    service = HistoryService()

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 2, 6)

    async def _fake_load_weekly_prices(_symbols, _start_date, _end_date):
        return {
            "AAA": [
                {"date": "2026-01-30", "close": 10.0},
                {"date": "2026-02-06", "close": 12.0},
            ]
        }

    async def _fake_company_names(_symbols):
        return {"AAA": "AAA Corp"}

    async def _fake_benchmarks(_start_date, _end_date):
        return {}

    db_session.add(StockHistorySyncState(symbol="AAA", sync_status="running"))
    await db_session.commit()
    sync_status.start_history_sync(total_symbols=1)

    try:
        monkeypatch.setattr(service, "_load_weekly_prices_from_db", _fake_load_weekly_prices)
        monkeypatch.setattr(service, "_load_benchmark_prices", _fake_benchmarks)
        monkeypatch.setattr(service, "_get_company_names", _fake_company_names)
        monkeypatch.setattr(history_module, "date", FixedDate)

        result = await service.get_stocks_weekly_prices(
            symbols=["AAA"],
            start_year=2026,
            include_benchmarks=True,
        )

        assert result["is_stale"] is False
        assert result["is_syncing"] is True
    finally:
        sync_status.complete_history_sync(success=True)


@pytest.mark.asyncio
async def test_get_stocks_volume_series_reports_db_syncing_status(monkeypatch, db_session):
    service = HistoryService()

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 2, 11)

    monkeypatch.setattr(history_module, "date", FixedDate)

    db_session.add(StockCompany(symbol="AAA", company_name="AAA Corp"))
    db_session.add(
        StockDailyHistory(
            symbol="AAA",
            date=date(2026, 2, 11),
            close=20.0,
            volume=1_000_000,
        )
    )
    db_session.add(StockHistorySyncState(symbol="AAA", sync_status="running"))
    await db_session.commit()
    sync_status.start_history_sync(total_symbols=1)

    try:
        result = await service.get_stocks_volume_series(
            symbols=["aaa"],
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 11),
        )

        assert result["is_stale"] is False
        assert result["is_syncing"] is True
    finally:
        sync_status.complete_history_sync(success=True)


@pytest.mark.asyncio
async def test_get_stocks_volume_series_ignores_stale_db_running_when_runtime_idle(monkeypatch, db_session):
    service = HistoryService()

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 2, 11)

    monkeypatch.setattr(history_module, "date", FixedDate)

    db_session.add(StockCompany(symbol="AAA", company_name="AAA Corp"))
    db_session.add(
        StockDailyHistory(
            symbol="AAA",
            date=date(2026, 2, 11),
            close=20.0,
            volume=1_000_000,
        )
    )
    db_session.add(StockHistorySyncState(symbol="AAA", sync_status="running"))
    await db_session.commit()

    sync_status.complete_history_sync(success=True)

    result = await service.get_stocks_volume_series(
        symbols=["aaa"],
        start_date=date(2026, 2, 1),
        end_date=date(2026, 2, 11),
    )

    assert result["is_stale"] is False
    assert result["is_syncing"] is False


@pytest.mark.asyncio
async def test_stop_background_workers_cancels_weekly_sync_task():
    service = HistoryService()

    await service.stop_background_workers()


def test_build_daily_history_payload_deduplicates_by_symbol_date():
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

    payload, min_date, max_date = service._build_daily_history_payload(
        symbol="inc",
        ohlcv_hist=hist,
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
    assert payload[1]["foreign_buy_volume"] is None
    assert payload[1]["prop_buy_volume"] is None


def test_build_daily_history_payload_merges_foreign_and_prop_metrics():
    service = HistoryService()
    ohlcv_hist = pd.DataFrame(
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
                "time": "2021-11-01",
                "open": 9.2,
                "high": 9.4,
                "low": 9.1,
                "close": 9.2,
                "volume": 1100,
            },
        ]
    )
    turnover_hist = pd.DataFrame(
        [
            {
                "time": "2021-10-29",
                "matched_volume": 1_000,
                "matched_value": 9_000_000_000.0,
                "deal_volume": 50,
                "deal_value": 450_000_000.0,
                "total_volume": 1_050,
                "total_value": 9_450_000_000.0,
            }
        ]
    )
    foreign_hist = pd.DataFrame(
        [
            {
                "time": "2021-10-29",
                "fr_buy_volume": 200,
                "fr_buy_value": 1000.5,
                "fr_sell_volume": 150,
                "fr_sell_value": 700.25,
                "fr_net_volume": 50,
                "fr_net_value": 300.25,
            }
        ]
    )
    prop_hist = pd.DataFrame(
        [
            {
                "time": "2021-11-01",
                "prop_buy_volume": 80,
                "prop_buy_value": 500.0,
                "prop_sell_volume": 60,
                "prop_sell_value": 420.0,
            }
        ]
    )

    payload, min_date, max_date = service._build_daily_history_payload(
        symbol="inc",
        ohlcv_hist=ohlcv_hist,
        turnover_hist=turnover_hist,
        foreign_hist=foreign_hist,
        prop_hist=prop_hist,
    )

    assert min_date == date(2021, 10, 29)
    assert max_date == date(2021, 11, 1)
    assert payload[0]["matched_volume"] == 1_000
    assert payload[0]["matched_value"] == 9_000_000_000.0
    assert payload[0]["deal_volume"] == 50
    assert payload[0]["deal_value"] == 450_000_000.0
    assert payload[0]["total_volume"] == 1_050
    assert payload[0]["total_value"] == 9_450_000_000.0
    assert payload[0]["foreign_buy_volume"] == 200
    assert payload[0]["foreign_net_value"] == 300.25
    assert payload[0]["prop_buy_volume"] is None
    assert payload[1]["prop_buy_volume"] == 80
    assert payload[1]["prop_sell_value"] == 420.0
    assert payload[1]["foreign_buy_volume"] is None


def test_normalize_foreign_trade_history_frame_to_vci_schema():
    service = HistoryService()
    raw_hist = pd.DataFrame(
        [
            {
                "Ngay": "2021-10-29",
                "KLMua": 200,
                "GtMua": 1000.5,
                "KLBan": 150,
                "GtBan": 700.25,
                "KLGDRong": 50,
                "GTDGRong": 300.25,
            }
        ]
    )

    normalized = service._normalize_foreign_trade_history_frame(raw_hist)

    assert normalized is not None
    assert normalized.loc[0, "time"] == "2021-10-29"
    assert normalized.loc[0, "fr_buy_volume"] == 200
    assert normalized.loc[0, "fr_buy_value"] == 1000.5
    assert normalized.loc[0, "fr_sell_volume"] == 150
    assert normalized.loc[0, "fr_sell_value"] == 700.25
    assert normalized.loc[0, "fr_net_volume"] == 50
    assert normalized.loc[0, "fr_net_value"] == 300.25


def test_normalize_foreign_trade_history_frame_from_live_total_columns():
    service = HistoryService()
    raw_hist = pd.DataFrame(
        [
            {
                "trading_date": "2021-10-29",
                "fr_buy_volume_total": 200,
                "fr_buy_value_total": 1000.5,
                "fr_sell_volume_total": 150,
                "fr_sell_value_total": 700.25,
                "fr_net_volume_matched": 20,
                "fr_net_volume_deal": 30,
                "fr_net_value_matched": 100.0,
                "fr_net_value_deal": 200.25,
            }
        ]
    )

    normalized = service._normalize_foreign_trade_history_frame(raw_hist)

    assert normalized is not None
    assert normalized.loc[0, "time"] == "2021-10-29"
    assert normalized.loc[0, "fr_buy_volume"] == 200
    assert normalized.loc[0, "fr_buy_value"] == 1000.5
    assert normalized.loc[0, "fr_sell_volume"] == 150
    assert normalized.loc[0, "fr_sell_value"] == 700.25
    assert normalized.loc[0, "fr_net_volume"] == 50
    assert normalized.loc[0, "fr_net_value"] == 300.25


def test_normalize_prop_trade_history_frame_to_vci_schema():
    service = HistoryService()
    raw_hist = pd.DataFrame(
        [
            {
                "Date": "2021-10-29",
                "KLcpMua": 80,
                "GtMua": 500.0,
                "KlcpBan": 60,
                "GtBan": 420.0,
            }
        ]
    )

    normalized = service._normalize_prop_trade_history_frame(raw_hist)

    assert normalized is not None
    assert normalized.loc[0, "time"] == "2021-10-29"
    assert normalized.loc[0, "prop_buy_volume"] == 80
    assert normalized.loc[0, "prop_buy_value"] == 500.0
    assert normalized.loc[0, "prop_sell_volume"] == 60
    assert normalized.loc[0, "prop_sell_value"] == 420.0


def test_normalize_prop_trade_history_frame_from_live_total_columns():
    service = HistoryService()
    raw_hist = pd.DataFrame(
        [
            {
                "trading_date": "2021-10-29",
                "total_buy_trade_volume": 80,
                "total_buy_trade_value": 500.0,
                "total_match_trade_sell_volume": 35,
                "total_deal_trade_sell_volume": 25,
                "total_match_trade_sell_value": 250.0,
                "total_deal_trade_sell_value": 170.0,
            }
        ]
    )

    normalized = service._normalize_prop_trade_history_frame(raw_hist)

    assert normalized is not None
    assert normalized.loc[0, "time"] == "2021-10-29"
    assert normalized.loc[0, "prop_buy_volume"] == 80
    assert normalized.loc[0, "prop_buy_value"] == 500.0
    assert normalized.loc[0, "prop_sell_volume"] == 60
    assert normalized.loc[0, "prop_sell_value"] == 420.0


def test_normalize_turnover_history_frame_derives_total_columns():
    service = HistoryService()
    raw_hist = pd.DataFrame(
        [
            {
                "trading_date": "2021-10-29",
                "matched_volume": 777_777,
                "matched_value": 7_777_777.0,
                "deal_volume": 666_666,
                "deal_value": 6_666_666.0,
            }
        ]
    )

    normalized = service._normalize_turnover_history_frame(raw_hist)

    assert normalized is not None
    assert normalized.loc[0, "time"] == "2021-10-29"
    assert normalized.loc[0, "matched_volume"] == 777_777
    assert normalized.loc[0, "matched_value"] == 7_777_777.0
    assert normalized.loc[0, "deal_volume"] == 666_666
    assert normalized.loc[0, "deal_value"] == 6_666_666.0
    assert normalized.loc[0, "total_volume"] == 1_444_443
    assert normalized.loc[0, "total_value"] == 14_444_443.0


def test_auxiliary_trade_fetches_use_vnstock_data_trading(monkeypatch):
    service = HistoryService()
    calls = []

    class FakeTrading:
        def __init__(self, source, symbol):
            calls.append(("init", source, symbol))

        def foreign_trade(self, **kwargs):
            calls.append(("foreign_trade", kwargs))
            return pd.DataFrame([{"time": "2021-10-29", "fr_buy_volume": 100}])

        def prop_trade(self, **kwargs):
            calls.append(("prop_trade", kwargs))
            return pd.DataFrame([{"time": "2021-10-29", "prop_buy_volume": 50}])

    fake_vnstock_data = ModuleType("vnstock_data")
    fake_vnstock_data.Trading = FakeTrading
    monkeypatch.setitem(sys.modules, "vnstock_data", fake_vnstock_data)

    foreign = service._fetch_foreign_trade_history(
        symbol="VCB",
        start_date=date(2021, 10, 1),
        end_date=date(2021, 10, 29),
    )
    prop = service._fetch_prop_trade_history(
        symbol="VCB",
        start_date=date(2021, 10, 1),
        end_date=date(2021, 10, 29),
    )

    assert foreign.loc[0, "fr_buy_volume"] == 100
    assert prop.loc[0, "prop_buy_volume"] == 50
    assert calls == [
        ("init", "VCI", "VCB"),
        (
            "foreign_trade",
            {
                "start": "2021-10-01",
                "end": "2021-10-29",
                "resolution": "1D",
                "limit": 100,
            },
        ),
        ("init", "VCI", "VCB"),
        (
            "prop_trade",
            {
                "start": "2021-10-01",
                "end": "2021-10-29",
                "resolution": "1D",
                "limit": 100,
            },
        ),
    ]


def test_turnover_fetch_uses_vendored_vnstock_data_alt_trading(monkeypatch):
    service = HistoryService()
    calls = []

    class FakeTrading:
        def __init__(self, source, symbol, show_log=False):
            calls.append(("init", source, symbol, show_log))

        def price_history(self, **kwargs):
            calls.append(("price_history", kwargs))
            return pd.DataFrame(
                [
                    {
                        "trading_date": "2025-03-07",
                        "matched_volume": 100,
                        "matched_value": 1_000.0,
                        "deal_volume": 50,
                        "deal_value": 700.0,
                    }
                ]
            )

    monkeypatch.setattr("app.lib.vnstock_data_alt.api.trading.Trading", FakeTrading)

    turnover = service._fetch_turnover_history(
        symbol="VCB",
        start_date=date(2025, 3, 1),
        end_date=date(2025, 3, 7),
    )

    assert turnover is not None
    assert turnover.loc[0, "matched_volume"] == 100
    assert turnover.loc[0, "matched_value"] == 1_000.0
    assert turnover.loc[0, "deal_volume"] == 50
    assert turnover.loc[0, "deal_value"] == 700.0
    assert turnover.loc[0, "total_volume"] == 150
    assert turnover.loc[0, "total_value"] == 1_700.0
    assert calls == [
        ("init", "vci", "VCB", False),
        (
            "price_history",
            {
                "start": "2025-03-01",
                "end": "2025-03-07",
                "limit": 100,
            },
        ),
    ]


def test_build_daily_history_payload_uses_provider_aggregate_flow_fields():
    service = HistoryService()
    ohlcv_hist = pd.DataFrame(
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
    foreign_hist = pd.DataFrame(
        [
            {
                "time": "2021-10-29",
                "matched_volume": 999_999,
                "matched_value": 9_999_999.0,
                "deal_volume": 888_888,
                "deal_value": 8_888_888.0,
                "fr_buy_volume": 200,
                "fr_buy_value": 1000.5,
                "fr_sell_volume": 150,
                "fr_sell_value": 700.25,
                "fr_net_volume": 50,
                "fr_net_value": 300.25,
            }
        ]
    )
    prop_hist = pd.DataFrame(
        [
            {
                "time": "2021-10-29",
                "matched_volume": 777_777,
                "matched_value": 7_777_777.0,
                "deal_volume": 666_666,
                "deal_value": 6_666_666.0,
                "prop_buy_volume": 80,
                "prop_buy_value": 500.0,
                "prop_sell_volume": 60,
                "prop_sell_value": 420.0,
            }
        ]
    )

    payload, _, _ = service._build_daily_history_payload(
        symbol="inc",
        ohlcv_hist=ohlcv_hist,
        turnover_hist=foreign_hist[["time", "matched_volume", "matched_value", "deal_volume", "deal_value"]].copy(),
        foreign_hist=foreign_hist,
        prop_hist=prop_hist,
    )

    assert len(payload) == 1
    assert payload[0]["matched_volume"] == 999_999
    assert payload[0]["matched_value"] == 9_999_999.0
    assert payload[0]["deal_volume"] == 888_888
    assert payload[0]["deal_value"] == 8_888_888.0
    assert payload[0]["total_volume"] == 1_888_887
    assert payload[0]["total_value"] == 18_888_887.0
    assert payload[0]["foreign_buy_value"] == 1000.5
    assert payload[0]["foreign_sell_value"] == 700.25
    assert payload[0]["foreign_net_value"] == 300.25
    assert payload[0]["prop_buy_value"] == 500.0
    assert payload[0]["prop_sell_value"] == 420.0


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

    count = service._upsert_stock_daily_history(
        symbol="INC",
        start_date=date(2021, 10, 1),
        end_date=date(2021, 11, 30),
        session=failing_session,
    )

    assert count == 0
    assert failing_session.rollback_calls == 1


def test_upsert_stock_daily_history_raises_when_raise_on_error_enabled(monkeypatch):
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
        service._upsert_stock_daily_history(
            symbol="INC",
            start_date=date(2021, 10, 1),
            end_date=date(2021, 11, 30),
            session=failing_session,
            raise_on_error=True,
        )

    assert failing_session.rollback_calls == 1


def test_upsert_stock_daily_history_serializes_same_symbol_calls(monkeypatch):
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
        return service._upsert_stock_daily_history(
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


def test_upsert_stock_daily_history_continues_when_auxiliary_fetch_fails(monkeypatch):
    service = HistoryService()
    ohlcv_hist = pd.DataFrame(
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

    class FakeQuote:
        def history(self, **_kwargs):
            return ohlcv_hist

    class FakeStock:
        quote = FakeQuote()

    class FakeVnstock:
        def stock(self, **_kwargs):
            return FakeStock()

    class FakeTrading:
        def __init__(self, **_kwargs):
            return None

        def foreign_trade(self, **_kwargs):
            raise RuntimeError("foreign unavailable")

        def prop_trade(self, **_kwargs):
            raise RuntimeError("prop unavailable")

    class RecordingSession:
        def __init__(self):
            self.executed = 0
            self.commits = 0

        def execute(self, _stmt):
            self.executed += 1

        def commit(self):
            self.commits += 1

        def rollback(self):
            raise AssertionError("rollback should not be called")

    monkeypatch.setitem(
        sys.modules,
        "vnstock",
        SimpleNamespace(Vnstock=FakeVnstock, Trading=FakeTrading),
    )
    monkeypatch.setattr(history_module, "retry_with_backoff", lambda func, max_retries=2: func())

    session = RecordingSession()
    count = service._upsert_stock_daily_history(
        symbol="INC",
        start_date=date(2021, 10, 1),
        end_date=date(2021, 11, 30),
        session=session,
    )

    assert count == 1
    assert session.executed == 1
    assert session.commits == 1


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

    monkeypatch.setattr(service, "_upsert_stock_daily_history", _raise_upsert)

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
    StockDailyHistory.__table__.create(bind=engine, checkfirst=True)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 2, 11, 10, 0, 0)

    monkeypatch.setattr(history_module, "datetime", FixedDateTime)
    monkeypatch.setattr(history_module, "get_sync_engine", lambda: engine)

    with Session(engine) as session:
        session.add(StockCompany(symbol="TCB", company_name="Techcombank"))
        session.add(
            StockDailyHistory(
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
            StockDailyHistory(
                symbol="TCB",
                date=date(2025, 11, 14),
                open=20.5,
                high=21.5,
                low=20.0,
                close=21.11,
                volume=1_000_000,
                matched_volume=1_010_000,
                matched_value=21_310_000_000,
                deal_volume=90_000,
                deal_value=1_890_000_000,
                total_volume=1_100_000,
                total_value=23_200_000_000,
                foreign_net_value=2_750_000_000,
                prop_buy_value=4_500_000_000,
                prop_sell_value=1_250_000_000,
            )
        )
        session.add(
            StockDailyHistory(
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
    assert result["data"][0]["matched_volume"] == 1_010_000
    assert result["data"][0]["matched_value"] == 21.31
    assert result["data"][0]["deal_volume"] == 90_000
    assert result["data"][0]["deal_value"] == 1.89
    assert result["data"][0]["total_volume"] == 1_100_000
    assert result["data"][0]["total_value"] == 23.2
    assert result["data"][0]["foreign_net_value"] == 2.75
    assert result["data"][0]["prop_buy_value"] == 4.5
    assert result["data"][0]["prop_sell_value"] == 1.25
    assert result["data"][0]["prop_net_value"] == 3.25
    assert result["data"][1]["volume"] == 0
    assert result["data"][1]["value"] is None
    assert result["data"][1]["matched_volume"] is None
    assert result["data"][1]["matched_value"] is None
    assert result["data"][1]["deal_volume"] is None
    assert result["data"][1]["deal_value"] is None
    assert result["data"][1]["total_volume"] is None
    assert result["data"][1]["total_value"] is None
    assert result["data"][1]["foreign_net_value"] is None
    assert result["data"][1]["prop_buy_value"] is None
    assert result["data"][1]["prop_sell_value"] is None
    assert result["data"][1]["prop_net_value"] is None


def test_fetch_volume_history_sync_derives_missing_net_values_only_when_buy_and_sell_exist(monkeypatch):
    service = HistoryService()
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    StockCompany.__table__.create(bind=engine, checkfirst=True)
    StockDailyHistory.__table__.create(bind=engine, checkfirst=True)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 2, 11, 10, 0, 0)

    monkeypatch.setattr(history_module, "datetime", FixedDateTime)
    monkeypatch.setattr(history_module, "get_sync_engine", lambda: engine)

    with Session(engine) as session:
        session.add(StockCompany(symbol="TCB", company_name="Techcombank"))
        session.add(
            StockDailyHistory(
                symbol="TCB",
                date=date(2026, 2, 10),
                open=20.5,
                high=21.0,
                low=20.0,
                close=20.5,
                volume=1_000_000,
                matched_value=10_000_000_000,
                deal_value=3_000_000_000,
                foreign_buy_value=12_500_000_000,
                foreign_sell_value=7_250_000_000,
                foreign_net_value=None,
                prop_buy_value=9_000_000_000,
                prop_sell_value=3_400_000_000,
            )
        )
        session.add(
            StockDailyHistory(
                symbol="TCB",
                date=date(2026, 2, 11),
                open=20.8,
                high=21.1,
                low=20.2,
                close=20.7,
                volume=900_000,
                foreign_buy_value=5_000_000_000,
                foreign_sell_value=None,
                foreign_net_value=None,
                prop_buy_value=4_200_000_000,
                prop_sell_value=None,
            )
        )
        session.commit()

    result = service._fetch_volume_history_sync("TCB", 2)

    assert result["data"][0]["total_value"] == 13.0
    assert result["data"][0]["foreign_net_value"] == 5.25
    assert result["data"][0]["prop_net_value"] == 5.6
    assert result["data"][1]["foreign_net_value"] is None
    assert result["data"][1]["prop_net_value"] is None


def test_fetch_price_history_sync_uses_db_calendar_window_and_normalizes_symbol(monkeypatch):
    service = HistoryService()
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    StockCompany.__table__.create(bind=engine, checkfirst=True)
    StockDailyHistory.__table__.create(bind=engine, checkfirst=True)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 2, 11, 10, 0, 0)

    monkeypatch.setattr(history_module, "datetime", FixedDateTime)
    monkeypatch.setattr(history_module, "get_sync_engine", lambda: engine)

    with Session(engine) as session:
        session.add(StockCompany(symbol="TCB", company_name="Techcombank"))
        session.add(
            StockDailyHistory(
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
            StockDailyHistory(
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
            StockDailyHistory(
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


def test_fetch_price_history_ohlcv_sync_returns_full_history_latest_first(monkeypatch):
    service = HistoryService()
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    StockCompany.__table__.create(bind=engine, checkfirst=True)
    StockDailyHistory.__table__.create(bind=engine, checkfirst=True)
    monkeypatch.setattr(history_module, "get_sync_engine", lambda: engine)

    with Session(engine) as session:
        session.add(StockCompany(symbol="TCB", company_name="Techcombank"))
        session.add(
            StockDailyHistory(
                symbol="TCB",
                date=date(2026, 2, 8),
                open=21.0,
                high=21.5,
                low=20.7,
                close=21.2,
                volume=1_200_000,
            )
        )
        session.add(
            StockDailyHistory(
                symbol="TCB",
                date=date(2026, 2, 10),
                open=22.0,
                high=22.4,
                low=21.8,
                close=22.1,
                volume=1_100_000,
            )
        )
        session.add(
            StockDailyHistory(
                symbol="TCB",
                date=date(2026, 2, 9),
                open=None,
                high=21.9,
                low=21.1,
                close=21.4,
                volume=None,
            )
        )
        session.commit()

    result = service._fetch_price_history_ohlcv_sync("tcbx")

    assert result["symbol"] == "TCB"
    assert result["company_name"] == "Techcombank"
    assert [point["date"] for point in result["data"]] == ["2026-02-10", "2026-02-09", "2026-02-08"]
    assert result["data"][0]["open"] == 22.0
    assert result["data"][0]["high"] == 22.4
    assert result["data"][0]["low"] == 21.8
    assert result["data"][0]["close"] == 22.1
    assert result["data"][0]["volume"] == 1_100_000
    assert result["data"][1]["open"] is None
    assert result["data"][1]["close"] == 21.4
    assert result["data"][1]["volume"] is None


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
async def test_get_volume_history_skips_request_path_sync_when_auto_sync_disabled(monkeypatch):
    service = HistoryService()
    sync_called = False

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 2, 11)

    class InlineLoop:
        async def run_in_executor(self, _executor, func, *args):
            return func(*args)

    async def _fake_request_sync(_symbol, _start_date, _end_date, _timeout_seconds):
        nonlocal sync_called
        sync_called = True
        return {
            "sync_performed": True,
            "sync_timed_out": False,
            "sync_error": None,
            "updated_through": "2026-02-11",
            "repaired_missing_dates": 1,
        }

    monkeypatch.setattr(history_module, "date", FixedDate)
    monkeypatch.setattr(history_module.asyncio, "get_event_loop", lambda: InlineLoop())
    monkeypatch.setattr(service, "_fetch_volume_history_sync", lambda _symbol, _days: {
        "symbol": "TCB",
        "company_name": "Techcombank",
        "data": [{"date": "2026-02-10", "volume": 1200, "value": 20.2}],
    })
    service.set_on_demand_history_sync_handler(_fake_request_sync)

    result = await service.get_volume_history("tcbx", 90, auto_sync=False)

    assert sync_called is False
    assert result["sync_performed"] is False
    assert result["sync_timed_out"] is False
    assert result["sync_error"] is None
    assert result["updated_through"] is None
    assert result["repaired_missing_dates"] == 0


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
async def test_get_price_history_skips_request_path_sync_when_auto_sync_disabled(monkeypatch):
    service = HistoryService()
    sync_called = False

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 2, 11)

    class InlineLoop:
        async def run_in_executor(self, _executor, func, *args):
            return func(*args)

    async def _fake_request_sync(_symbol, _start_date, _end_date, _timeout_seconds):
        nonlocal sync_called
        sync_called = True
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

    result = await service.get_price_history("tcbx", 90, auto_sync=False)

    assert sync_called is False
    assert result["sync_performed"] is False
    assert result["sync_timed_out"] is False
    assert result["sync_error"] is None
    assert result["updated_through"] is None
    assert result["repaired_missing_dates"] == 0


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


@pytest.mark.asyncio
async def test_get_stocks_volume_series_normalizes_inputs_and_computes_value(monkeypatch, db_session):
    service = HistoryService()
    trigger_calls = []

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 2, 11)

    async def _fake_trigger(symbols, start_date, end_date, force=False):
        trigger_calls.append({
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "force": force,
        })
        return False

    monkeypatch.setattr(history_module, "date", FixedDate)
    monkeypatch.setattr(service, "_trigger_price_history_sync", _fake_trigger)

    db_session.add(StockCompany(symbol="TCB", company_name="Techcombank"))
    db_session.add(
        StockDailyHistory(
            symbol="TCB",
            date=date(2026, 2, 10),
            open=20.0,
            high=20.5,
            low=19.8,
            close=20.0,
            volume=1_500_000,
        )
    )
    db_session.add(
        StockDailyHistory(
            symbol="TCB",
            date=date(2026, 2, 11),
            open=20.1,
            high=20.3,
            low=19.9,
            close=20.2,
            volume=None,
        )
    )
    await db_session.commit()

    result = await service.get_stocks_volume_series(
        symbols=["tcbx", "TCB", ""],
        start_date=date(2026, 2, 20),
        end_date=date(2026, 2, 1),
    )

    assert result["start_date"] == "2026-02-01"
    assert result["end_date"] == "2026-02-11"
    assert result["is_stale"] is False
    assert result["is_syncing"] is False
    assert trigger_calls == []

    assert len(result["stocks"]) == 1
    stock = result["stocks"][0]
    assert stock["symbol"] == "TCB"
    assert stock["ticker"] == "TCB"
    assert stock["company_name"] == "Techcombank"
    assert stock["data"][0]["date"] == "2026-02-10"
    assert stock["data"][0]["value"] == 30.0
    assert stock["data"][1]["date"] == "2026-02-11"
    assert stock["data"][1]["value"] is None


@pytest.mark.asyncio
async def test_get_stocks_volume_series_triggers_sync_for_stale_symbols(monkeypatch, db_session):
    service = HistoryService()
    trigger_calls = []

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 2, 11)

    async def _fake_trigger(symbols, start_date, end_date, force=False):
        trigger_calls.append({
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "force": force,
        })
        return True

    monkeypatch.setattr(history_module, "date", FixedDate)
    monkeypatch.setattr(service, "_trigger_price_history_sync", _fake_trigger)

    result = await service.get_stocks_volume_series(
        symbols=["VCB"],
        start_date=date(2026, 1, 1),
        end_date=date(2026, 2, 11),
    )

    assert result["is_stale"] is True
    assert result["is_syncing"] is True
    assert len(trigger_calls) == 1
    assert trigger_calls[0]["symbols"] == ["VCB"]
    assert trigger_calls[0]["start_date"] == date(2026, 1, 1)
    assert trigger_calls[0]["end_date"] == date(2026, 2, 11)
    assert trigger_calls[0]["force"] is False


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
        StockDailyHistory.__table__.create(bind=engine, checkfirst=True)
        with Session(engine) as session:
            session.query(StockDailyHistory).filter(StockDailyHistory.symbol == symbol).delete()
            session.commit()

            original_created_at = datetime(2020, 1, 1, 0, 0, 0)
            session.add(
                StockDailyHistory(
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
            monkeypatch.setattr(
                service,
                "_fetch_auxiliary_history_frames",
                lambda **_kwargs: (
                    pd.DataFrame(
                        [
                            {
                                "time": "2021-10-29",
                                "matched_volume": 900,
                                "matched_value": 9_000.0,
                                "deal_volume": 100,
                                "deal_value": 1_100.0,
                                "total_volume": 1_000,
                                "total_value": 10_100.0,
                            },
                            {
                                "time": "2021-11-01",
                                "matched_volume": 1_250,
                                "matched_value": 12_500.0,
                                "deal_volume": 150,
                                "deal_value": 1_600.0,
                                "total_volume": 1_400,
                                "total_value": 14_100.0,
                            },
                        ]
                    ),
                    pd.DataFrame(
                        [
                            {
                                "time": "2021-10-29",
                                "fr_buy_volume": 200,
                                "fr_buy_value": 1000.0,
                                "fr_sell_volume": 150,
                                "fr_sell_value": 800.0,
                                "fr_net_volume": 50,
                                "fr_net_value": 200.0,
                            },
                            {
                                "time": "2021-11-01",
                                "fr_buy_volume": 210,
                                "fr_buy_value": 1100.0,
                                "fr_sell_volume": 140,
                                "fr_sell_value": 750.0,
                                "fr_net_volume": 70,
                                "fr_net_value": 350.0,
                            },
                        ]
                    ),
                    pd.DataFrame(
                        [
                            {
                                "time": "2021-10-29",
                                "prop_buy_volume": 20,
                                "prop_buy_value": 110.0,
                                "prop_sell_volume": 10,
                                "prop_sell_value": 55.0,
                            },
                            {
                                "time": "2021-11-01",
                                "prop_buy_volume": 25,
                                "prop_buy_value": 130.0,
                                "prop_sell_volume": 12,
                                "prop_sell_value": 60.0,
                            },
                        ]
                    ),
                ),
            )

            synced = service._upsert_stock_daily_history(
                symbol=symbol,
                start_date=date(2021, 10, 1),
                end_date=date(2021, 11, 30),
                session=session,
            )
            assert synced == 2

            rows = session.execute(
                select(StockDailyHistory)
                .where(StockDailyHistory.symbol == symbol)
                .order_by(StockDailyHistory.date.asc())
            ).scalars().all()

            assert len(rows) == 2
            assert rows[0].date == date(2021, 10, 29)
            assert rows[0].close == 9.0
            assert rows[0].volume == 1250
            assert rows[0].matched_volume == 900
            assert rows[0].deal_value == 1_100.0
            assert rows[0].total_value == 10_100.0
            assert rows[0].foreign_buy_volume == 200
            assert rows[0].foreign_net_value == 200.0
            assert rows[0].prop_buy_volume == 20
            assert rows[0].created_at == original_created_at
            assert rows[1].date == date(2021, 11, 1)
            assert rows[1].close == 9.35
            assert rows[1].volume == 1400
            assert rows[1].total_volume == 1_400
            assert rows[1].foreign_buy_volume == 210
            assert rows[1].prop_sell_value == 60.0

            hist_without_aux = pd.DataFrame(
                [
                    {
                        "time": "2021-10-29",
                        "open": 9.2,
                        "high": 9.4,
                        "low": 9.1,
                        "close": 9.1,
                        "volume": 1300,
                    }
                ]
            )
            monkeypatch.setattr(history_module, "retry_with_backoff", lambda _func, max_retries=2: hist_without_aux)
            monkeypatch.setattr(
                service,
                "_fetch_auxiliary_history_frames",
                lambda **_kwargs: (None, None, None),
            )

            synced = service._upsert_stock_daily_history(
                symbol=symbol,
                start_date=date(2021, 10, 29),
                end_date=date(2021, 10, 29),
                session=session,
            )
            assert synced == 1

            preserved_row = session.execute(
                select(StockDailyHistory).where(
                    StockDailyHistory.symbol == symbol,
                    StockDailyHistory.date == date(2021, 10, 29),
                )
            ).scalar_one()

            assert preserved_row.close == 9.1
            assert preserved_row.volume == 1300
            assert preserved_row.matched_volume == 900
            assert preserved_row.total_value == 10_100.0
            assert preserved_row.foreign_buy_volume == 200
            assert preserved_row.foreign_net_value == 200.0
            assert preserved_row.prop_buy_volume == 20
            assert preserved_row.prop_sell_value == 55.0
    finally:
        engine.dispose()
