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
                    enriched = await metadata_service.enrich_stocks_with_metadata(stocks)
                    
                    # Finance calls should be skipped due to sync_status
                    mock_fetch_finance.assert_not_called()
                    # Symbol call is attempted but catches CircuitOpenError
