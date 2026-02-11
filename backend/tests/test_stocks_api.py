import pytest
from unittest.mock import patch
from app.db.models import StockIndex
from app.services.vnstock_service import vnstock_service, StockInfo

@pytest.mark.asyncio
async def test_get_indices(client):
    # We use client fixture which sets up the DB
    # We need to populate data using the db_session implicitly used by client
    # But client fixture doesn't expose db_session directly.
    # We can request db_session fixture as well.
    pass

@pytest.mark.asyncio
async def test_get_indices_with_data(client, db_session):
    # Setup
    idx1 = StockIndex(symbol='VN30', name='VN30 Index', group='HOSE')
    idx2 = StockIndex(symbol='VN100', name='VN100 Index', group='HOSE')
    db_session.add_all([idx1, idx2])
    await db_session.commit()

    # We mock vnstock_service.get_indices to return DB data? 
    # Actually the endpoint calls vnstock_service.get_indices() which queries DB.
    # So we don't need to patch vnstock_service if we populated DB.
    
    response = await client.get("/api/v1/stocks/indices")
    assert response.status_code == 200
    data = response.json()
    assert "indices" in data
    assert len(data["indices"]) >= 2
    symbols = [idx["symbol"] for idx in data["indices"]]
    assert "VN30" in symbols

@pytest.mark.asyncio
async def test_get_stocks_by_index(client):
    mock_stocks = [
        StockInfo(ticker="TCB", price=50000, market_cap=100000, company_name="Techcombank"),
        StockInfo(ticker="VCB", price=90000, market_cap=200000, company_name="Vietcombank")
    ]

    with patch.object(vnstock_service, 'get_index_stocks', return_value=mock_stocks):
        response = await client.get("/api/v1/stocks/index/VN30")
        assert response.status_code == 200
        data = response.json()
        assert "stocks" in data
        assert data["count"] == 2
        assert data["index_symbol"] == "VN30"
        assert data["stocks"][0]["ticker"] == "TCB"

@pytest.mark.asyncio
async def test_get_industries(client):
    mock_industries = [
        {"icb_name": "Banks", "en_icb_name": "Banks", "icb_code": "8300"},
        {"icb_name": "Real Estate", "en_icb_name": "Real Estate", "icb_code": "8600"}
    ]
    with patch.object(vnstock_service, 'get_industry_list', return_value=mock_industries):
        response = await client.get("/api/v1/stocks/industries")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert data["industries"][0]["name"] == "Banks"

@pytest.mark.asyncio
async def test_get_stocks_by_industry(client):
    mock_stocks = [
        StockInfo(ticker="TCB", price=50000, market_cap=100000, company_name="Techcombank")
    ]
    with patch.object(vnstock_service, 'get_industry_stocks', return_value=mock_stocks):
        response = await client.get("/api/v1/stocks/industry/Banks")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["industry_name"] == "Banks"
        assert data["stocks"][0]["ticker"] == "TCB"

@pytest.mark.asyncio
async def test_get_volume_history(client):
    mock_volume = {
        "symbol": "TCB",
        "company_name": "Techcombank",
        "data": [{"date": "2023-01-01", "volume": 1000, "value": 50.0}]
    }
    with patch.object(vnstock_service, 'get_volume_history', return_value=mock_volume):
        response = await client.get("/api/v1/stocks/history/TCB/volume")
        assert response.status_code == 200
        data = response.json()
    assert data["symbol"] == "TCB"
    assert data["data"][0]["volume"] == 1000


@pytest.mark.asyncio
async def test_get_price_history(client):
    mock_prices = {
        "symbol": "TCB",
        "company_name": "Techcombank",
        "data": [{"date": "2023-01-01", "close": 50100.0}]
    }
    with patch.object(vnstock_service, 'get_price_history', return_value=mock_prices):
        response = await client.get("/api/v1/stocks/history/TCB/price")
        assert response.status_code == 200
        data = response.json()
    assert data["symbol"] == "TCB"
    assert data["data"][0]["close"] == 50100.0

@pytest.mark.asyncio
async def test_get_stock_quotes(client):
    mock_stocks = [
        StockInfo(ticker="TCB", price=50000, market_cap=100000, company_name="Techcombank"),
        StockInfo(ticker="VCB", price=90000, market_cap=200000, company_name="Vietcombank")
    ]

    with patch.object(vnstock_service, 'get_symbol_stocks', return_value=mock_stocks):
        response = await client.post(
            "/api/v1/stocks/quotes",
            json={"symbols": ["TCB", "VCB"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        tickers = [stock["ticker"] for stock in data["stocks"]]
        assert "TCB" in tickers
        assert "VCB" in tickers
