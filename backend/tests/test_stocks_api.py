import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.db.models import StockIndex
from app.db.database import async_session
from sqlalchemy import delete

@pytest.mark.asyncio
async def test_get_indices():
    # Setup: Ensure some data exists
    async with async_session() as session:
        await session.execute(delete(StockIndex))
        idx1 = StockIndex(symbol='VN30', name='VN30 Index', group='HOSE')
        idx2 = StockIndex(symbol='VN100', name='VN100 Index', group='HOSE')
        session.add_all([idx1, idx2])
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/v1/stocks/indices")
    
    assert response.status_code == 200
    data = response.json()
    assert "indices" in data
    assert len(data["indices"]) >= 2
    
    symbols = [idx["symbol"] for idx in data["indices"]]
    assert "VN30" in symbols
    assert "VN100" in symbols

@pytest.mark.asyncio
async def test_get_stocks_by_index():
    # Mock the vnstock service response to avoid external calls
    from app.services.vnstock_service import vnstock_service
    from app.services.vnstock_service import StockInfo
    from unittest.mock import patch

    mock_stocks = [
        StockInfo(ticker="TCB", price=50000, market_cap=100000, company_name="Techcombank"),
        StockInfo(ticker="VCB", price=90000, market_cap=200000, company_name="Vietcombank")
    ]

    with patch.object(vnstock_service, 'get_index_stocks', return_value=mock_stocks):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/api/v1/stocks/index/VN30")
        
        assert response.status_code == 200
        data = response.json()
        assert "stocks" in data
        assert data["count"] == 2
        assert data["index_symbol"] == "VN30"
        assert data["stocks"][0]["ticker"] == "TCB"

