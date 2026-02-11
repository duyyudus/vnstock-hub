import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
from app.services.vnstock_service.stock_metadata import StockMetadataService
from app.services.vnstock_service.models import StockInfo
from app.db.models import StockCompany
from app.services.vnstock_service.core import CircuitOpenError

@pytest.fixture
def metadata_service():
    return StockMetadataService()

@pytest.fixture
def mock_db_session():
    session = AsyncMock()
    # Mock the context manager behavior
    session.__aenter__.return_value = session
    return session

@pytest.mark.asyncio
async def test_apply_cache_to_stocks(metadata_service, mock_db_session):
    # Setup stocks
    stocks = [
        StockInfo(ticker="TCB", price=50000, market_cap=100000),
        StockInfo(ticker="VCB", price=90000, market_cap=200000)
    ]
    
    # Setup mock DB data
    mock_company = StockCompany(
        symbol="TCB", 
        company_name="Techcombank", 
        pe_ratio=5.5,
        charter_capital=1000000
    )
    
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [mock_company]
    mock_db_session.execute.return_value = mock_result

    with patch("app.services.vnstock_service.stock_metadata.async_session", return_value=mock_db_session):
        enriched = await metadata_service.apply_cache_to_stocks(stocks)
        
        assert enriched[0].ticker == "TCB"
        assert enriched[0].company_name == "Techcombank"
        assert enriched[0].pe_ratio == 5.5
        assert enriched[1].company_name == ""  # Default empty string

@pytest.mark.asyncio
async def test_enrich_stocks_everything_cached(metadata_service, mock_db_session):
    stocks = [StockInfo(ticker="TCB", price=50000, market_cap=100000)]
    now = datetime.utcnow()
    
    # Recently updated company
    mock_company = StockCompany(
        symbol="TCB", 
        company_name="Techcombank", 
        pe_ratio=5.5,
        updated_at=now
    )
    
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [mock_company]
    mock_db_session.execute.return_value = mock_result

    with patch("app.services.vnstock_service.stock_metadata.async_session", return_value=mock_db_session):
        with patch.object(metadata_service, "_fetch_all_symbols") as mock_fetch_symbols:
            with patch.object(metadata_service, "_fetch_stock_finance_sync") as mock_fetch_finance:
                enriched = await metadata_service.enrich_stocks_with_metadata(stocks)
                
                assert enriched[0].company_name == "Techcombank"
                mock_fetch_symbols.assert_not_called()
                mock_fetch_finance.assert_not_called()

@pytest.mark.asyncio
async def test_enrich_stocks_missing_data_triggers_api(metadata_service, mock_db_session):
    stocks = [StockInfo(ticker="TCB", price=50000, market_cap=100000)]
    
    # DB is empty
    mock_result_empty = MagicMock()
    mock_result_empty.scalars.return_value.all.return_value = []

    import pandas as pd
    mock_symbols_df = pd.DataFrame([{"symbol": "TCB", "organ_name": "Techcombank API"}])
    mock_finance_data = {"pe_ratio": 6.0}

    mock_company = StockCompany(
        symbol="TCB",
        company_name="Techcombank API",
        pe_ratio=6.0
    )
    mock_result_company = MagicMock()
    mock_result_company.scalars.return_value.all.return_value = [mock_company]

    # First select -> empty, insert -> placeholder, refresh select -> company
    mock_db_session.execute.side_effect = [mock_result_empty, MagicMock(), mock_result_company]

    with patch("app.services.vnstock_service.stock_metadata.async_session", return_value=mock_db_session):
        with patch.object(metadata_service, "_fetch_all_symbols", return_value=mock_symbols_df):
            with patch.object(metadata_service, "_fetch_stock_finance_sync", return_value=mock_finance_data):
                # Use a small sleep to avoid long waits in tests
                with patch("asyncio.sleep", return_value=None):
                    enriched = await metadata_service.enrich_stocks_with_metadata(stocks)
                    
                    assert enriched[0].company_name == "Techcombank API"
                    assert enriched[0].pe_ratio == 6.0
                    assert not mock_db_session.add.called
                    assert mock_db_session.commit.called

@pytest.mark.asyncio
async def test_enrich_stocks_missing_data_uses_upsert(metadata_service, mock_db_session):
    stocks = [StockInfo(ticker="TCB", price=50000, market_cap=100000)]

    # DB is empty
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db_session.execute.return_value = mock_result

    import pandas as pd
    mock_symbols_df = pd.DataFrame([{"symbol": "TCB", "organ_name": "Techcombank API"}])

    with patch("app.services.vnstock_service.stock_metadata.async_session", return_value=mock_db_session):
        with patch.object(metadata_service, "_fetch_all_symbols", return_value=mock_symbols_df):
            with patch.object(metadata_service, "_fetch_stock_finance_sync", return_value=None):
                with patch("asyncio.sleep", return_value=None):
                    await metadata_service.enrich_stocks_with_metadata(stocks)

                    # Verify we executed an INSERT with ON CONFLICT DO NOTHING
                    insert_calls = [
                        call for call in mock_db_session.execute.call_args_list
                        if "ON CONFLICT" in str(call.args[0])
                    ]
                    assert insert_calls

@pytest.mark.asyncio
async def test_enrich_stocks_rate_limited_skips_api(metadata_service, mock_db_session):
    stocks = [StockInfo(ticker="TCB", price=50000, market_cap=100000)]
    
    # DB is empty
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db_session.execute.return_value = mock_result

    with patch("app.services.vnstock_service.stock_metadata.async_session", return_value=mock_db_session):
        with patch("app.services.vnstock_service.stock_metadata.sync_status") as mock_sync:
            mock_sync.is_rate_limited = True
            
            with patch.object(metadata_service, "_fetch_all_symbols", side_effect=CircuitOpenError("Mock limit")):
                with patch.object(metadata_service, "_fetch_stock_finance_sync") as mock_fetch_finance:
                    await metadata_service.enrich_stocks_with_metadata(stocks)

                    # Finance calls should be skipped due to sync_status
                    mock_fetch_finance.assert_not_called()


@pytest.mark.asyncio
async def test_metadata_enrichment_logs_completed_on_success(metadata_service, mock_db_session):
    stocks = [StockInfo(ticker="TCB", price=50000, market_cap=100000)]
    stale_company = StockCompany(
        symbol="TCB",
        company_name="Techcombank",
        pe_ratio=None,
        updated_at=datetime.utcnow() - timedelta(days=8)
    )
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [stale_company]
    mock_db_session.execute.return_value = mock_result

    with patch("app.services.vnstock_service.stock_metadata.async_session", return_value=mock_db_session):
        with patch("app.services.vnstock_service.stock_metadata.api_circuit_breaker.can_proceed", return_value=True):
            with patch.object(metadata_service, "_fetch_stock_finance_sync", return_value={"pe_ratio": 6.0}):
                with patch("asyncio.sleep", return_value=None):
                    with patch("app.services.vnstock_service.stock_metadata.log_background_start") as mock_start:
                        with patch("app.services.vnstock_service.stock_metadata.log_background_complete") as mock_complete:
                            with patch("app.services.vnstock_service.stock_metadata.log_background_error") as mock_error:
                                enriched = await metadata_service.enrich_stocks_with_metadata(stocks)

    assert enriched[0].pe_ratio == 6.0
    mock_start.assert_called_once()
    mock_complete.assert_called_once()
    mock_error.assert_not_called()
    assert mock_complete.call_args.args[0] == "Metadata Enrichment"
    summary = mock_complete.call_args.args[1]
    assert "processed=1/1" in summary
    assert "pe_updated=1" in summary
    assert "stopped_early=false" in summary


@pytest.mark.asyncio
async def test_metadata_enrichment_logs_completed_partial_on_rate_limit_stop(metadata_service, mock_db_session):
    stocks = [
        StockInfo(ticker="AAA", price=50000, market_cap=100000),
        StockInfo(ticker="BBB", price=50000, market_cap=100000),
    ]
    stale_a = StockCompany(
        symbol="AAA",
        company_name="AAA Corp",
        pe_ratio=None,
        updated_at=datetime.utcnow() - timedelta(days=8)
    )
    stale_b = StockCompany(
        symbol="BBB",
        company_name="BBB Corp",
        pe_ratio=None,
        updated_at=datetime.utcnow() - timedelta(days=8)
    )
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [stale_a, stale_b]
    mock_db_session.execute.return_value = mock_result

    with patch("app.services.vnstock_service.stock_metadata.async_session", return_value=mock_db_session):
        with patch(
            "app.services.vnstock_service.stock_metadata.api_circuit_breaker.can_proceed",
            side_effect=[True, False]
        ):
            with patch.object(metadata_service, "_fetch_stock_finance_sync", return_value={"pe_ratio": 7.0}) as mock_fetch:
                with patch("asyncio.sleep", return_value=None):
                    with patch("app.services.vnstock_service.stock_metadata.log_background_start") as mock_start:
                        with patch("app.services.vnstock_service.stock_metadata.log_background_complete") as mock_complete:
                            with patch("app.services.vnstock_service.stock_metadata.log_background_error") as mock_error:
                                await metadata_service.enrich_stocks_with_metadata(stocks)

    mock_start.assert_called_once()
    mock_complete.assert_called_once()
    mock_error.assert_not_called()
    assert mock_fetch.call_count == 1
    assert mock_complete.call_args.args[0] == "Metadata Enrichment"
    summary = mock_complete.call_args.args[1]
    assert "processed=1/2" in summary
    assert "stopped_early=true" in summary
    assert "reason=circuit_open" in summary


@pytest.mark.asyncio
async def test_metadata_enrichment_logs_failed_on_unexpected_batch_exception(metadata_service, mock_db_session):
    stocks = [StockInfo(ticker="TCB", price=50000, market_cap=100000)]
    stale_company = StockCompany(
        symbol="TCB",
        company_name="Techcombank",
        pe_ratio=None,
        updated_at=datetime.utcnow() - timedelta(days=8)
    )
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [stale_company]
    mock_db_session.execute.return_value = mock_result
    mock_db_session.commit = AsyncMock(side_effect=[None, RuntimeError("commit failed")])

    with patch("app.services.vnstock_service.stock_metadata.async_session", return_value=mock_db_session):
        with patch("app.services.vnstock_service.stock_metadata.api_circuit_breaker.can_proceed", return_value=True):
            with patch.object(metadata_service, "_fetch_stock_finance_sync", return_value={"pe_ratio": 6.0}):
                with patch("asyncio.sleep", return_value=None):
                    with patch("app.services.vnstock_service.stock_metadata.log_background_start") as mock_start:
                        with patch("app.services.vnstock_service.stock_metadata.log_background_complete") as mock_complete:
                            with patch("app.services.vnstock_service.stock_metadata.log_background_error") as mock_error:
                                with pytest.raises(RuntimeError, match="commit failed"):
                                    await metadata_service.enrich_stocks_with_metadata(stocks)

    mock_start.assert_called_once()
    mock_complete.assert_not_called()
    mock_error.assert_called_once()
    assert mock_error.call_args.args[0] == "Metadata Enrichment"


@pytest.mark.asyncio
async def test_metadata_enrichment_no_terminal_log_when_not_started(metadata_service, mock_db_session):
    stocks = [StockInfo(ticker="TCB", price=50000, market_cap=100000)]
    fresh_company = StockCompany(
        symbol="TCB",
        company_name="Techcombank",
        pe_ratio=5.5,
        updated_at=datetime.utcnow()
    )
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [fresh_company]
    mock_db_session.execute.return_value = mock_result

    with patch("app.services.vnstock_service.stock_metadata.async_session", return_value=mock_db_session):
        with patch("app.services.vnstock_service.stock_metadata.log_background_start") as mock_start:
            with patch("app.services.vnstock_service.stock_metadata.log_background_complete") as mock_complete:
                with patch("app.services.vnstock_service.stock_metadata.log_background_error") as mock_error:
                    await metadata_service.enrich_stocks_with_metadata(stocks)

    mock_start.assert_not_called()
    mock_complete.assert_not_called()
    mock_error.assert_not_called()


def test_fetch_stock_finance_sync_uses_finance_service_not_direct_vnstock():
    finance_service = MagicMock()
    finance_service.get_financial_ratios = AsyncMock(return_value=[{"P/E": 8.2}])
    finance_service.extract_latest_pe_ratio.return_value = 8.2
    metadata_service = StockMetadataService(finance_service=finance_service)

    with patch("app.services.vnstock_service.stock_metadata.asyncio.run", return_value=[{"P/E": 8.2}]) as mock_run:
        result = metadata_service._fetch_stock_finance_sync("TCB")

    assert result == {"pe_ratio": 8.2}
    mock_run.assert_called_once()
    finance_service.extract_latest_pe_ratio.assert_called_once_with([{"P/E": 8.2}])
