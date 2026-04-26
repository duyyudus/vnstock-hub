import pytest
from unittest.mock import patch, MagicMock
from httpx import AsyncClient
from app.services.vnstock_service import vnstock_service
from app.services.vnstock_service.funds import FundsService

@pytest.mark.asyncio
async def test_get_fund_listing(client: AsyncClient):
    mock_data = [
        {"symbol": "FUND1", "name": "Fund 1", "fund_type": "ETF"},
        {"symbol": "FUND2", "name": "Fund 2", "fund_type": "OPEN"}
    ]
    
    # We patch the instance method on the vnstock_service object
    with patch.object(vnstock_service, 'get_fund_listing', return_value=mock_data) as mock_method:
        response = await client.get("/api/v1/funds/listing")
        assert response.status_code == 200
        data = response.json()
        
        assert data["count"] == 2
        assert len(data["data"]) == 2
        assert data["data"][0]["symbol"] == "FUND1"
        
        # Verify the service method was called
        mock_method.assert_called_once()

@pytest.mark.asyncio
async def test_get_fund_listing_filtered(client: AsyncClient):
    mock_data = [{"symbol": "FUND1", "name": "Fund 1", "fund_type": "ETF"}]
    
    with patch.object(vnstock_service, 'get_fund_listing', return_value=mock_data) as mock_method:
        response = await client.get("/api/v1/funds/listing?fund_type=ETF")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        
        # Verify service was called with correct arg
        mock_method.assert_called_once_with(fund_type="ETF")

@pytest.mark.asyncio
async def test_get_fund_nav_report(client: AsyncClient):
    mock_data = [
        {"date": "2023-01-01", "nav": 10000},
        {"date": "2023-01-08", "nav": 10100}
    ]
    
    with patch.object(vnstock_service, 'get_fund_nav_report', return_value=mock_data):
        response = await client.get("/api/v1/funds/FUND1/nav-report")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "FUND1"
        assert len(data["data"]) == 2
        assert data["data"][0]["nav"] == 10000

@pytest.mark.asyncio
async def test_get_fund_performance(client: AsyncClient):
    mock_data = {
        "funds": [{"symbol": "FUND1", "name": "Fund 1"}],
        "benchmarks": {"VNINDEX": []},
        "common_start_date": "2023-01-01",
        "is_stale": False,
        "is_syncing": False
    }
    
    with patch.object(vnstock_service, 'get_fund_performance_data', return_value=mock_data):
        response = await client.get("/api/v1/funds/performance")
        assert response.status_code == 200
        data = response.json()
        assert len(data["funds"]) == 1
        assert "benchmarks" in data

@pytest.mark.asyncio
async def test_get_fund_overview(client: AsyncClient):
    mock_data = {
        "sectors": [
            {
                "sector": "Banks",
                "total_allocation": 18.0,
                "stock_count": 1,
                "stocks": [
                    {
                        "ticker": "TCB",
                        "company_name": "Techcombank",
                        "sector": "Banks",
                        "total_allocation": 18.0,
                        "fund_count": 2,
                        "funds": [
                            {
                                "symbol": "FUND1",
                                "name": "Fund 1",
                                "allocation": 10.0,
                                "holding_updated_at": "2026-02-01",
                            },
                            {
                                "symbol": "FUND2",
                                "name": "Fund 2",
                                "allocation": 8.0,
                                "holding_updated_at": "2026-02-02",
                            },
                        ],
                    }
                ],
            }
        ],
        "fund_count": 2,
        "stock_count": 1,
        "last_updated": "2026-02-02T00:00:00",
    }

    with patch.object(vnstock_service, 'get_fund_overview', return_value=mock_data) as mock_method:
        response = await client.get("/api/v1/funds/overview")
        assert response.status_code == 200
        data = response.json()
        assert data["fund_count"] == 2
        assert data["stock_count"] == 1
        assert data["sectors"][0]["sector"] == "Banks"
        assert data["sectors"][0]["stocks"][0]["ticker"] == "TCB"
        mock_method.assert_called_once()


def test_build_fund_overview_sums_stock_holdings_by_sector():
    service = FundsService()
    rows = [
        {
            "symbol": "FUND1",
            "updated_at": "2026-02-01T09:00:00",
            "data": [
                {
                    "ticker": "TCB",
                    "industry": "Banks",
                    "allocation": 10.0,
                    "type_asset": "STOCK",
                    "update_at": "2026-02-01",
                },
                {
                    "stock_code": "HPG",
                    "sector": "Materials",
                    "net_asset_percent": 4.0,
                    "type_asset": "STOCK",
                    "update_at": "2026-02-01",
                },
                {
                    "ticker": "BOND1",
                    "industry": "Banks",
                    "allocation": 30.0,
                    "type_asset": "BOND",
                    "update_at": "2026-02-01",
                },
            ],
        },
        {
            "symbol": "FUND2",
            "updated_at": "2026-02-02T09:00:00",
            "data": [
                {
                    "symbol": "TCB",
                    "industry_name": "Banks",
                    "weight": 8.0,
                    "type_asset": "STOCK",
                    "update_at": "2026-02-02",
                },
                {
                    "ticker": "FPT",
                    "industry": "Technology",
                    "percentage": 12.0,
                    "type_asset": "STOCK",
                    "update_at": "2026-02-02",
                },
            ],
        },
    ]

    overview = service._build_fund_overview(
        rows,
        fund_names={"FUND1": "Fund 1", "FUND2": "Fund 2"},
        company_names={"TCB": "Techcombank", "FPT": "FPT Corp"},
    )

    assert overview["fund_count"] == 2
    assert overview["stock_count"] == 3
    assert overview["last_updated"] == "2026-02-02T09:00:00"
    assert [sector["sector"] for sector in overview["sectors"]] == ["Banks", "Technology", "Materials"]

    banks = overview["sectors"][0]
    assert banks["total_allocation"] == 18.0
    assert banks["stock_count"] == 1
    assert banks["stocks"][0]["ticker"] == "TCB"
    assert banks["stocks"][0]["company_name"] == "Techcombank"
    assert banks["stocks"][0]["total_allocation"] == 18.0
    assert banks["stocks"][0]["fund_count"] == 2
    assert banks["stocks"][0]["funds"] == [
        {
            "symbol": "FUND1",
            "name": "Fund 1",
            "allocation": 10.0,
            "holding_updated_at": "2026-02-01",
        },
        {
            "symbol": "FUND2",
            "name": "Fund 2",
            "allocation": 8.0,
            "holding_updated_at": "2026-02-02",
        },
    ]

@pytest.mark.asyncio
async def test_get_fund_top_holding(client: AsyncClient):
    mock_data = [{"ticker": "TCB", "allocation": 10.5}]
    with patch.object(vnstock_service, 'get_fund_top_holding', return_value=mock_data):
        response = await client.get("/api/v1/funds/FUND1/top-holding")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "FUND1"
        assert data["data"][0]["ticker"] == "TCB"

@pytest.mark.asyncio
async def test_get_fund_industry_holding(client: AsyncClient):
    mock_data = [{"industry": "Banks", "allocation": 45.0}]
    with patch.object(vnstock_service, 'get_fund_industry_holding', return_value=mock_data):
        response = await client.get("/api/v1/funds/FUND1/industry-holding")
        assert response.status_code == 200
        data = response.json()
        assert data["data"][0]["industry"] == "Banks"

@pytest.mark.asyncio
async def test_get_fund_asset_holding(client: AsyncClient):
    mock_data = [{"asset_type": "Stocks", "allocation": 90.0}]
    with patch.object(vnstock_service, 'get_fund_asset_holding', return_value=mock_data):
        response = await client.get("/api/v1/funds/FUND1/asset-holding")
        assert response.status_code == 200
        data = response.json()
        assert data["data"][0]["asset_type"] == "Stocks"
