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

    result = await service.get_income_statement("AAA", period="quarter", lang="en")

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
        period="quarter",
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
        period="quarter",
        lang="en",
        raise_on_failure=False,
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
        period="quarter",
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
        period="quarter",
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
            period="quarter",
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
            period="quarter",
            lang="en",
            raise_on_failure=True,
        )


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
