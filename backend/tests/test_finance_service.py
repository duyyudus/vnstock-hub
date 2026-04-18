from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from app.services.vnstock_service.finance import FinanceService


@pytest.mark.asyncio
async def test_get_income_statement_uses_fresh_cache(monkeypatch):
    service = FinanceService()
    cached = SimpleNamespace(
        updated_at=datetime.utcnow(),
        data=[{"metric": "Revenue", "value": 123}],
    )

    monkeypatch.setattr(service, "_get_cache_entry", AsyncMock(return_value=cached))
    refresh_mock = AsyncMock(return_value=[{"metric": "Revenue", "value": 999}])
    monkeypatch.setattr(service, "refresh_financial_dataset", refresh_mock)

    result = await service.get_income_statement("AAA", lang="en")

    assert result == [{"metric": "Revenue", "value": 123}]
    refresh_mock.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_financial_dataset_fetches_and_upserts_on_cache_miss(monkeypatch):
    service = FinanceService()

    monkeypatch.setattr(service, "_get_cache_entry", AsyncMock(return_value=None))
    monkeypatch.setattr(
        service,
        "_fetch_from_source",
        AsyncMock(return_value=[{"metric": "Revenue", "value": 200}]),
    )
    upsert_mock = AsyncMock()
    monkeypatch.setattr(service, "_upsert_cache_entry", upsert_mock)

    result = await service.refresh_financial_dataset(
        symbol="AAA",
        data_type=FinanceService.DATA_TYPE_INCOME,
        lang="en",
        raise_on_failure=True,
    )

    assert result == [{"metric": "Revenue", "value": 200}]
    upsert_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_refresh_financial_dataset_refreshes_when_stale(monkeypatch):
    service = FinanceService()
    stale_cache = SimpleNamespace(
        updated_at=datetime.utcnow() - timedelta(days=200),
        data=[{"metric": "Revenue", "value": 100}],
    )

    monkeypatch.setattr(service, "_get_cache_entry", AsyncMock(return_value=stale_cache))
    monkeypatch.setattr(
        service,
        "_fetch_from_source",
        AsyncMock(return_value=[{"metric": "Revenue", "value": 300}]),
    )
    upsert_mock = AsyncMock()
    monkeypatch.setattr(service, "_upsert_cache_entry", upsert_mock)

    result = await service.refresh_financial_dataset(
        symbol="AAA",
        data_type=FinanceService.DATA_TYPE_INCOME,
        lang="en",
        raise_on_failure=False,
    )

    assert result == [{"metric": "Revenue", "value": 300}]
    upsert_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_refresh_financial_dataset_force_refresh_bypasses_fresh_cache(monkeypatch):
    service = FinanceService()
    fresh_cache = SimpleNamespace(
        updated_at=datetime.utcnow(),
        data=[{"metric": "Revenue", "value": 100}],
    )

    monkeypatch.setattr(service, "_get_cache_entry", AsyncMock(return_value=fresh_cache))
    monkeypatch.setattr(
        service,
        "_fetch_from_source",
        AsyncMock(return_value=[{"metric": "Revenue", "value": 300}]),
    )
    upsert_mock = AsyncMock()
    monkeypatch.setattr(service, "_upsert_cache_entry", upsert_mock)

    result = await service.refresh_financial_dataset(
        symbol="AAA",
        data_type=FinanceService.DATA_TYPE_INCOME,
        lang="en",
        raise_on_failure=False,
        force_refresh=True,
    )

    assert result == [{"metric": "Revenue", "value": 300}]
    upsert_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_refresh_financial_dataset_fallbacks_to_stale_cache_on_refresh_error(monkeypatch):
    service = FinanceService()
    stale_cache = SimpleNamespace(
        updated_at=datetime.utcnow() - timedelta(days=200),
        data=[{"metric": "Revenue", "value": 100}],
    )

    monkeypatch.setattr(service, "_get_cache_entry", AsyncMock(return_value=stale_cache))
    monkeypatch.setattr(
        service,
        "_fetch_from_source",
        AsyncMock(side_effect=RuntimeError("boom")),
    )

    result = await service.refresh_financial_dataset(
        symbol="AAA",
        data_type=FinanceService.DATA_TYPE_INCOME,
        lang="en",
        raise_on_failure=False,
    )

    assert result == [{"metric": "Revenue", "value": 100}]


@pytest.mark.asyncio
async def test_refresh_financial_dataset_returns_empty_when_no_cache_and_refresh_fails(monkeypatch):
    service = FinanceService()

    monkeypatch.setattr(service, "_get_cache_entry", AsyncMock(return_value=None))
    monkeypatch.setattr(
        service,
        "_fetch_from_source",
        AsyncMock(side_effect=RuntimeError("boom")),
    )

    result = await service.refresh_financial_dataset(
        symbol="AAA",
        data_type=FinanceService.DATA_TYPE_INCOME,
        lang="en",
        raise_on_failure=False,
    )

    assert result == []


@pytest.mark.asyncio
async def test_refresh_financial_dataset_raises_when_configured(monkeypatch):
    service = FinanceService()

    monkeypatch.setattr(service, "_get_cache_entry", AsyncMock(return_value=None))
    monkeypatch.setattr(
        service,
        "_fetch_from_source",
        AsyncMock(side_effect=RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        await service.refresh_financial_dataset(
            symbol="AAA",
            data_type=FinanceService.DATA_TYPE_INCOME,
            lang="en",
            raise_on_failure=True,
        )


@pytest.mark.asyncio
async def test_refresh_financial_dataset_raises_even_with_stale_cache_when_configured(monkeypatch):
    service = FinanceService()
    stale_cache = SimpleNamespace(
        updated_at=datetime.utcnow() - timedelta(days=200),
        data=[{"metric": "Revenue", "value": 100}],
    )

    monkeypatch.setattr(service, "_get_cache_entry", AsyncMock(return_value=stale_cache))
    monkeypatch.setattr(
        service,
        "_fetch_from_source",
        AsyncMock(side_effect=RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        await service.refresh_financial_dataset(
            symbol="AAA",
            data_type=FinanceService.DATA_TYPE_INCOME,
            lang="en",
            raise_on_failure=True,
        )


def test_last_completed_quarter_start_handles_year_boundary():
    service = FinanceService()

    result = service._last_completed_quarter_start(datetime(2026, 2, 11, 8, 30, 0))

    assert result == datetime(2025, 10, 1)


def test_last_completed_quarter_start_handles_regular_boundary():
    service = FinanceService()

    result = service._last_completed_quarter_start(datetime(2026, 5, 10, 8, 30, 0))

    assert result == datetime(2026, 1, 1)


def test_is_stale_uses_last_completed_quarter_start(monkeypatch):
    service = FinanceService()
    boundary = datetime(2025, 10, 1)

    monkeypatch.setattr(service, "_last_completed_quarter_start", lambda reference=None: boundary)

    assert service._is_stale(datetime(2025, 9, 30, 23, 59, 59)) is True
    assert service._is_stale(datetime(2025, 10, 1, 0, 0, 0)) is False
    assert service._is_stale(datetime(2025, 12, 31, 23, 59, 59)) is False


def test_normalize_finance_dataframe_keeps_shape_and_json_safe_values():
    service = FinanceService()

    cols = pd.MultiIndex.from_tuples([
        ("Statement", "Revenue"),
        ("Statement", "Profit"),
        ("Meta", "as_of"),
    ])
    df = pd.DataFrame([
        [1000, float("nan"), pd.Timestamp("2025-03-31")],
    ], columns=cols)

    records = service._normalize_finance_dataframe(df)

    assert isinstance(records, list)
    assert len(records) == 1
    assert records[0]["Statement_Revenue"] == 1000
    assert records[0]["Statement_Profit"] is None
    assert records[0]["Meta_as_of"] == "2025-03-31T00:00:00"


def test_normalize_statement_dataset_merges_quarter_metadata_and_keeps_readable_columns():
    service = FinanceService()

    final_df = pd.DataFrame([
        {"report_period": "quarter", "ticker": "AAA", "Revenue": 250, "Profit": 25},
        {"report_period": "quarter", "ticker": "AAA", "Revenue": 100, "Profit": 10},
    ])
    raw_df = pd.DataFrame([
        {
            "organCode": "AAA",
            "ticker": "AAA",
            "createDate": "2024-08-01T00:00:00",
            "updateDate": "2024-08-01T00:00:00",
            "publicDate": "2024-08-02T00:00:00",
            "yearReport": 2024,
            "lengthReport": 2,
        },
        {
            "organCode": "AAA",
            "ticker": "AAA",
            "createDate": "2024-05-01T00:00:00",
            "updateDate": "2024-05-01T00:00:00",
            "publicDate": "2024-05-02T00:00:00",
            "yearReport": 2024,
            "lengthReport": 1,
        },
    ])

    records = service._normalize_statement_dataset(final_df, raw_df, dataset_name="income statement")

    assert records == [
        {
            "ticker": "AAA",
            "yearReport": 2024,
            "lengthReport": 2,
            "Revenue": 250,
            "Profit": 25,
        },
        {
            "ticker": "AAA",
            "yearReport": 2024,
            "lengthReport": 1,
            "Revenue": 100,
            "Profit": 10,
        },
    ]


def test_normalize_statement_dataset_raises_on_row_mismatch():
    service = FinanceService()

    final_df = pd.DataFrame([{"ticker": "AAA", "Revenue": 100}])
    raw_df = pd.DataFrame([
        {"ticker": "AAA", "yearReport": 2024, "lengthReport": 1},
        {"ticker": "AAA", "yearReport": 2024, "lengthReport": 2},
    ])

    with pytest.raises(ValueError, match="row count mismatch"):
        service._normalize_statement_dataset(final_df, raw_df, dataset_name="income statement")


def test_normalize_ratio_dataset_infers_meta_quarters_and_excludes_annual_rows():
    service = FinanceService()

    raw_df = pd.DataFrame([
        {
            "Ratio TTM Id": 101,
            "Ratio Type": "RATIO_TTM",
            "Ratio Year Id": None,
            "organCode": "AAA",
            "yearReport": 2024,
            "P/E": 11.0,
            "Debt to Equity": 0.5,
        },
        {
            "Ratio TTM Id": 102,
            "Ratio Type": "RATIO_TTM",
            "Ratio Year Id": None,
            "organCode": "AAA",
            "yearReport": 2024,
            "P/E": 12.0,
            "Debt to Equity": 0.6,
        },
        {
            "Ratio TTM Id": 999,
            "Ratio Type": "RATIO_YEAR",
            "Ratio Year Id": 999,
            "organCode": "AAA",
            "yearReport": 2024,
            "P/E": 99.0,
            "Debt to Equity": 9.9,
        },
        {
            "Ratio TTM Id": 201,
            "Ratio Type": "RATIO_TTM",
            "Ratio Year Id": None,
            "organCode": "AAA",
            "yearReport": 2025,
            "P/E": 21.0,
            "Debt to Equity": 0.7,
        },
    ])

    records = service._normalize_ratio_dataset(raw_df)

    assert records == [
        {
            "Meta_ticker": "AAA",
            "Meta_yearReport": 2025,
            "Meta_lengthReport": 1,
            "Meta_period": "2025-Q1",
            "P/E": 21.0,
            "Debt to Equity": 0.7,
        },
        {
            "Meta_ticker": "AAA",
            "Meta_yearReport": 2024,
            "Meta_lengthReport": 2,
            "Meta_period": "2024-Q2",
            "P/E": 12.0,
            "Debt to Equity": 0.6,
        },
        {
            "Meta_ticker": "AAA",
            "Meta_yearReport": 2024,
            "Meta_lengthReport": 1,
            "Meta_period": "2024-Q1",
            "P/E": 11.0,
            "Debt to Equity": 0.5,
        },
    ]
    assert service.extract_latest_pe_ratio(records) == 21.0


def test_fetch_income_statement_sync_uses_vnstock_data_alt_client(monkeypatch):
    service = FinanceService()
    calls = []

    class StubFinanceClient:
        def income_statement(self, *, period, lang, mode):
            calls.append((period, lang, mode))
            if mode == "final":
                return pd.DataFrame([
                    {"report_period": "quarter", "ticker": "AAA", "Revenue": 100},
                ])
            return pd.DataFrame([
                {"ticker": "AAA", "yearReport": 2024, "lengthReport": 1},
            ])

    monkeypatch.setattr(service, "_build_vnstock_data_alt_client", lambda symbol: StubFinanceClient())
    monkeypatch.setattr("app.services.vnstock_service.finance.api_circuit_breaker.can_proceed", lambda: True)
    monkeypatch.setattr("app.services.vnstock_service.finance.api_circuit_breaker.record_success", lambda: None)

    records = service._fetch_income_statement_sync("AAA", "en")

    assert calls == [("quarter", "en", "final"), ("quarter", "en", "raw")]
    assert records == [
        {
            "ticker": "AAA",
            "yearReport": 2024,
            "lengthReport": 1,
            "Revenue": 100,
        }
    ]


def test_fetch_financial_ratios_sync_uses_vnstock_data_alt_raw_ratios(monkeypatch):
    service = FinanceService()
    calls = []

    class StubFinanceClient:
        def ratio(self, *, period, lang, mode):
            calls.append((period, lang, mode))
            return pd.DataFrame([
                {
                    "Ratio TTM Id": 301,
                    "Ratio Type": "RATIO_TTM",
                    "organCode": "AAA",
                    "yearReport": 2024,
                    "P/E": 7.5,
                },
                {
                    "Ratio TTM Id": 302,
                    "Ratio Type": "RATIO_TTM",
                    "organCode": "AAA",
                    "yearReport": 2024,
                    "P/E": 8.5,
                },
            ])

    monkeypatch.setattr(service, "_build_vnstock_data_alt_client", lambda symbol: StubFinanceClient())
    monkeypatch.setattr("app.services.vnstock_service.finance.api_circuit_breaker.can_proceed", lambda: True)
    monkeypatch.setattr("app.services.vnstock_service.finance.api_circuit_breaker.record_success", lambda: None)

    records = service._fetch_financial_ratios_sync("AAA", "en")

    assert calls == [("quarter", "en", "raw")]
    assert records[0]["Meta_period"] == "2024-Q2"
    assert service.extract_latest_pe_ratio(records) == 8.5
