from datetime import date
import pytest
from unittest.mock import patch
from app.db.models import StockIndex
from app.services.vnstock_service import vnstock_service, StockInfo
from app.services.vnstock_service.models import IndexContribution, IndexContributionTotals
from app.services.vnstock_service.stocks import StocksService

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
        StockInfo(
            ticker="TCB",
            price=50000,
            market_cap=100000,
            company_name="Techcombank",
            foreign_buy_value=65.01,
            foreign_sell_value=34.01,
            current_room=1234567,
            total_room=8765432,
            atl_price=45000.0,
            atl_date="2026-01-15",
            atl_diff_pct=11.11,
            ath_price=52000.0,
            ath_date="2026-03-01",
            ath_diff_pct=-3.85,
        ),
        StockInfo(
            ticker="VCB",
            price=90000,
            market_cap=200000,
            company_name="Vietcombank",
            foreign_buy_value=None,
            foreign_sell_value=None,
            current_room=None,
            total_room=None,
        )
    ]

    with patch.object(vnstock_service, 'get_index_stocks', return_value=mock_stocks) as mock_get_index_stocks:
        response = await client.get("/api/v1/stocks/index/VN30?range_start=2026-01-01&range_end=2026-03-22")
        assert response.status_code == 200
        data = response.json()
    mock_get_index_stocks.assert_called_once_with(
        "VN30",
        1000,
        range_start=date(2026, 1, 1),
        range_end=date(2026, 3, 22),
    )
    assert "stocks" in data
    assert data["count"] == 2
    assert data["index_symbol"] == "VN30"
    assert data["stocks"][0]["ticker"] == "TCB"
    assert data["stocks"][0]["foreign_buy_value"] == 65.01
    assert data["stocks"][0]["foreign_sell_value"] == 34.01
    assert data["stocks"][0]["current_room"] == 1234567
    assert data["stocks"][0]["total_room"] == 8765432
    assert data["stocks"][0]["atl_price"] == 45000.0
    assert data["stocks"][0]["atl_date"] == "2026-01-15"
    assert data["stocks"][0]["atl_diff_pct"] == 11.11
    assert data["stocks"][0]["ath_price"] == 52000.0
    assert data["stocks"][0]["ath_date"] == "2026-03-01"
    assert data["stocks"][0]["ath_diff_pct"] == -3.85
    assert data["stocks"][1]["foreign_buy_value"] is None
    assert data["stocks"][1]["foreign_sell_value"] is None
    assert data["stocks"][1]["current_room"] is None
    assert data["stocks"][1]["total_room"] is None


def test_build_index_contribution_rows_uses_adjusted_weights_and_points():
    stocks = [
        StockInfo(ticker="AAA", price=110, market_cap=1000, company_name="AAA Corp", price_change_24h=10),
        StockInfo(ticker="BBB", price=90, market_cap=1000, company_name="BBB Corp", price_change_24h=-10),
    ]
    overview_by_ticker = {
        "AAA": {"outstanding_shares": 10_000_000, "free_float": 50},
        "BBB": {"outstanding_shares": 20_000_000, "free_float": 25},
    }

    rows, excluded = StocksService.build_index_contribution_rows(
        stocks=stocks,
        overview_by_ticker=overview_by_ticker,
        index_change_value=-5.0,
        apply_single_stock_cap=False,
    )

    assert excluded == 0
    rows_by_ticker = {row.ticker: row for row in rows}
    assert rows_by_ticker["AAA"].free_float_ratio == 0.5
    assert rows_by_ticker["BBB"].free_float_ratio == 0.25
    assert rows_by_ticker["AAA"].effective_weight == pytest.approx(0.5)
    assert rows_by_ticker["BBB"].effective_weight == pytest.approx(0.5)
    assert rows_by_ticker["AAA"].percent_contribution == pytest.approx(5.0)
    assert rows_by_ticker["BBB"].percent_contribution == pytest.approx(-5.0)
    assert rows_by_ticker["AAA"].point_contribution is None
    assert rows_by_ticker["BBB"].point_contribution is None


def test_build_index_contribution_rows_applies_single_stock_cap():
    stocks = [
        StockInfo(ticker="BIG", price=100, market_cap=9000, company_name="Big", price_change_24h=1),
    ]
    stocks.extend(
        StockInfo(ticker=f"S{i}", price=100, market_cap=100, company_name=f"Small {i}", price_change_24h=1)
        for i in range(1, 10)
    )
    overview_by_ticker = {
        "BIG": {"outstanding_shares": 90_000_000, "free_float": 100},
    }
    overview_by_ticker.update({
        f"S{i}": {"outstanding_shares": 1_000_000, "free_float": 100}
        for i in range(1, 10)
    })

    rows, excluded = StocksService.build_index_contribution_rows(
        stocks=stocks,
        overview_by_ticker=overview_by_ticker,
        index_change_value=3.0,
        apply_single_stock_cap=True,
    )

    assert excluded == 0
    rows_by_ticker = {row.ticker: row for row in rows}
    assert rows_by_ticker["BIG"].effective_weight == pytest.approx(0.1)
    assert rows_by_ticker["S1"].effective_weight == pytest.approx(0.1)
    assert sum(row.effective_weight for row in rows) == pytest.approx(1.0)


def test_build_index_contribution_rows_flags_missing_company_data_fallbacks():
    stocks = [
        StockInfo(ticker="AAA", price=100, market_cap=1000, company_name="AAA Corp", price_change_24h=2),
    ]

    rows, excluded = StocksService.build_index_contribution_rows(
        stocks=stocks,
        overview_by_ticker={},
        index_change_value=2.5,
        apply_single_stock_cap=False,
    )

    assert excluded == 0
    assert rows[0].missing_outstanding_shares is True
    assert rows[0].missing_free_float is True
    assert rows[0].used_market_cap_shares_fallback is True


@pytest.mark.asyncio
async def test_get_index_contribution(client):
    contribution = IndexContribution(
        symbol="VN30",
        name="VN30 Index",
        value=1234.5,
        change=-0.4,
        change_value=-5.0,
        rows=[],
        totals=IndexContributionTotals(
            positive_percent=0.1,
            negative_percent=-0.5,
            net_percent=-0.4,
            positive_points=1.25,
            negative_points=-6.25,
            net_points=-5.0,
            excluded_count=2,
            missing_outstanding_shares_count=1,
            missing_free_float_count=1,
        ),
    )

    with patch.object(vnstock_service, 'get_index_contribution', return_value=contribution) as mock_get_contribution:
        response = await client.get("/api/v1/stocks/index/VN30/contribution")
        assert response.status_code == 200
        data = response.json()

    mock_get_contribution.assert_called_once_with("VN30")
    assert data["symbol"] == "VN30"
    assert data["name"] == "VN30 Index"
    assert data["change_value"] == -5.0
    assert data["totals"]["net_points"] == -5.0
    assert data["totals"]["excluded_count"] == 2

@pytest.mark.asyncio
async def test_get_industries(client):
    mock_industries = [
        {
            "icb_name": "Banks",
            "en_icb_name": "Banks",
            "icb_code": "8300",
            "icb_family_code": "8301",
            "icb_family_name": "Banks",
            "icb_family_en_name": "Banks",
        },
        {
            "icb_name": "Real Estate",
            "en_icb_name": "Real Estate",
            "icb_code": "8600",
            "icb_family_code": "8000",
            "icb_family_name": "Financials",
            "icb_family_en_name": "Financials",
        },
    ]
    with patch.object(vnstock_service, 'get_industry_list', return_value=mock_industries):
        response = await client.get("/api/v1/stocks/industries")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert data["industries"][0]["name"] == "Banks"
        assert data["industries"][0]["family_code"] == "8301"
        assert data["industries"][1]["family_name"] == "Financials"

@pytest.mark.asyncio
async def test_get_stocks_by_industry(client):
    mock_stocks = [
        StockInfo(ticker="TCB", price=50000, market_cap=100000, company_name="Techcombank")
    ]
    with patch.object(vnstock_service, 'get_industry_stocks', return_value=mock_stocks) as mock_get_industry_stocks:
        response = await client.get("/api/v1/stocks/industry/Banks?range_start=2026-02-01&range_end=2026-03-01")
        assert response.status_code == 200
        data = response.json()
    mock_get_industry_stocks.assert_called_once_with(
        "Banks",
        1000,
        range_start=date(2026, 2, 1),
        range_end=date(2026, 3, 1),
    )
    assert data["count"] == 1
    assert data["industry_name"] == "Banks"
    assert data["stocks"][0]["ticker"] == "TCB"

@pytest.mark.asyncio
async def test_get_volume_history(client):
    mock_volume = {
        "symbol": "TCB",
        "company_name": "Techcombank",
        "data": [{
            "date": "2023-01-01",
            "volume": 1000,
            "value": 50.0,
            "matched_volume": 1100,
            "matched_value": 52.0,
            "deal_volume": 100,
            "deal_value": 8.0,
            "total_volume": 1200,
            "total_value": 60.0,
            "foreign_net_value": 12.5,
            "prop_buy_value": 4.0,
            "prop_sell_value": 1.5,
            "prop_net_value": 2.5,
        }],
        "sync_performed": True,
        "sync_timed_out": False,
        "sync_error": None,
        "updated_through": "2023-01-01",
        "repaired_missing_dates": 2,
    }
    with patch.object(vnstock_service, 'get_volume_history', return_value=mock_volume) as mock_get_volume_history:
        response = await client.get("/api/v1/stocks/history/TCB/volume")
        assert response.status_code == 200
        data = response.json()
    mock_get_volume_history.assert_called_once_with("TCB", days=30, auto_sync=True)
    assert data["symbol"] == "TCB"
    assert data["data"][0]["volume"] == 1000
    assert data["data"][0]["matched_volume"] == 1100
    assert data["data"][0]["matched_value"] == 52.0
    assert data["data"][0]["deal_volume"] == 100
    assert data["data"][0]["deal_value"] == 8.0
    assert data["data"][0]["total_volume"] == 1200
    assert data["data"][0]["total_value"] == 60.0
    assert data["data"][0]["foreign_net_value"] == 12.5
    assert data["data"][0]["prop_buy_value"] == 4.0
    assert data["data"][0]["prop_sell_value"] == 1.5
    assert data["data"][0]["prop_net_value"] == 2.5
    assert data["sync_performed"] is True
    assert data["sync_timed_out"] is False
    assert data["sync_error"] is None
    assert data["updated_through"] == "2023-01-01"
    assert data["repaired_missing_dates"] == 2


@pytest.mark.asyncio
async def test_get_stocks_volume_series(client):
    mock_series = {
        "stocks": [
            {
                "symbol": "TCB",
                "ticker": "TCB",
                "company_name": "Techcombank",
                "data": [
                    {"date": "2026-02-10", "value": 25.1, "foreign_net_value": 1.4, "prop_net_value": -0.8},
                    {"date": "2026-02-11", "value": 27.3, "foreign_net_value": None, "prop_net_value": 0.3},
                ],
            },
            {
                "symbol": "VCB",
                "ticker": "VCB",
                "company_name": "Vietcombank",
                "data": [],
            },
        ],
        "start_date": "2025-11-14",
        "end_date": "2026-02-11",
        "is_stale": True,
        "is_syncing": True,
    }
    with patch.object(vnstock_service, 'get_stocks_volume_series', return_value=mock_series):
        response = await client.post(
            "/api/v1/stocks/volume-series",
            json={
                "symbols": ["TCB", "VCB"],
                "start_date": "2025-11-14",
                "end_date": "2026-02-11",
            }
        )
        assert response.status_code == 200
        data = response.json()

    assert data["start_date"] == "2025-11-14"
    assert data["end_date"] == "2026-02-11"
    assert data["is_stale"] is True
    assert data["is_syncing"] is True
    assert data["stocks"][0]["symbol"] == "TCB"
    assert data["stocks"][0]["data"][0]["value"] == 25.1
    assert data["stocks"][0]["data"][0]["foreign_net_value"] == 1.4
    assert data["stocks"][0]["data"][0]["prop_net_value"] == -0.8
    assert data["stocks"][0]["data"][1]["foreign_net_value"] is None
    assert data["stocks"][0]["data"][1]["prop_net_value"] == 0.3
    assert data["stocks"][1]["symbol"] == "VCB"
    assert data["stocks"][1]["data"] == []


@pytest.mark.asyncio
async def test_get_stocks_weekly_prices(client):
    mock_series = {
        "stocks": [
            {
                "symbol": "TCB",
                "ticker": "TCB",
                "company_name": "Techcombank",
                "prices": [
                    {"date": "2025-11-14", "close": 25.1},
                    {"date": "2026-02-11", "close": 27.3},
                ],
            },
        ],
        "benchmarks": {
            "VNINDEX": [
                {"date": "2025-11-14", "close": 1200.0},
                {"date": "2026-02-11", "close": 1245.0},
            ],
        },
        "start_date": "2025-11-14",
        "end_date": "2026-02-11",
        "is_stale": False,
        "is_syncing": True,
    }
    with patch.object(vnstock_service, 'get_stocks_weekly_prices', return_value=mock_series) as mock_get_stocks_weekly_prices:
        response = await client.post(
            "/api/v1/stocks/weekly-prices",
            json={
                "symbols": ["TCB"],
                "start_date": "2025-11-14",
                "end_date": "2026-02-11",
                "include_benchmarks": True,
            }
        )
        assert response.status_code == 200
        data = response.json()

    mock_get_stocks_weekly_prices.assert_called_once_with(
        symbols=["TCB"],
        start_date=date(2025, 11, 14),
        end_date=date(2026, 2, 11),
        include_benchmarks=True,
    )
    assert data["start_date"] == "2025-11-14"
    assert data["end_date"] == "2026-02-11"
    assert data["is_stale"] is False
    assert data["is_syncing"] is True
    assert data["stocks"][0]["symbol"] == "TCB"
    assert data["stocks"][0]["prices"][0]["close"] == 25.1
    assert data["benchmarks"]["VNINDEX"][0]["close"] == 1200.0


@pytest.mark.asyncio
async def test_get_price_history(client):
    mock_prices = {
        "symbol": "TCB",
        "company_name": "Techcombank",
        "data": [{"date": "2023-01-01", "close": 50100.0}],
        "sync_performed": False,
        "sync_timed_out": True,
        "sync_error": None,
        "updated_through": "2022-12-30",
        "repaired_missing_dates": 0,
    }
    with patch.object(vnstock_service, 'get_price_history', return_value=mock_prices) as mock_get_price_history:
        response = await client.get("/api/v1/stocks/history/TCB/price")
        assert response.status_code == 200
        data = response.json()
    mock_get_price_history.assert_called_once_with("TCB", days=30, auto_sync=True)
    assert data["symbol"] == "TCB"
    assert data["data"][0]["close"] == 50100.0
    assert data["sync_performed"] is False
    assert data["sync_timed_out"] is True
    assert data["sync_error"] is None
    assert data["updated_through"] == "2022-12-30"
    assert data["repaired_missing_dates"] == 0


@pytest.mark.asyncio
async def test_get_volume_history_allows_disabling_auto_sync(client):
    mock_volume = {
        "symbol": "TCB",
        "company_name": "Techcombank",
        "data": [{"date": "2023-01-01", "volume": 1000, "value": 50.0}],
        "sync_performed": False,
        "sync_timed_out": False,
        "sync_error": None,
        "updated_through": None,
        "repaired_missing_dates": 0,
    }
    with patch.object(vnstock_service, 'get_volume_history', return_value=mock_volume) as mock_get_volume_history:
        response = await client.get("/api/v1/stocks/history/TCB/volume?auto_sync=false")
        assert response.status_code == 200
        data = response.json()

    mock_get_volume_history.assert_called_once_with("TCB", days=30, auto_sync=False)
    assert data["sync_performed"] is False
    assert data["sync_timed_out"] is False


@pytest.mark.asyncio
async def test_get_price_history_allows_disabling_auto_sync(client):
    mock_prices = {
        "symbol": "TCB",
        "company_name": "Techcombank",
        "data": [{"date": "2023-01-01", "close": 50100.0}],
        "sync_performed": False,
        "sync_timed_out": False,
        "sync_error": None,
        "updated_through": None,
        "repaired_missing_dates": 0,
    }
    with patch.object(vnstock_service, 'get_price_history', return_value=mock_prices) as mock_get_price_history:
        response = await client.get("/api/v1/stocks/history/TCB/price?auto_sync=false")
        assert response.status_code == 200
        data = response.json()

    mock_get_price_history.assert_called_once_with("TCB", days=30, auto_sync=False)
    assert data["sync_performed"] is False
    assert data["sync_timed_out"] is False


@pytest.mark.asyncio
async def test_get_price_history_ohlcv(client):
    mock_ohlcv = {
        "symbol": "TCB",
        "company_name": "Techcombank",
        "data": [
            {
                "date": "2023-01-02",
                "open": 20.1,
                "high": 20.6,
                "low": 19.9,
                "close": 20.4,
                "volume": 1005000,
            },
            {
                "date": "2023-01-01",
                "open": None,
                "high": 20.0,
                "low": 19.5,
                "close": 19.8,
                "volume": None,
            },
        ],
    }
    with patch.object(vnstock_service, 'get_price_history_ohlcv', return_value=mock_ohlcv):
        response = await client.get("/api/v1/stocks/history/TCB/ohlcv")
        assert response.status_code == 200
        data = response.json()
    assert data["symbol"] == "TCB"
    assert data["company_name"] == "Techcombank"
    assert data["count"] == 2
    assert data["data"][0]["date"] == "2023-01-02"
    assert data["data"][0]["open"] == 20.1
    assert data["data"][0]["close"] == 20.4
    assert data["data"][0]["volume"] == 1005000
    assert data["data"][1]["open"] is None
    assert data["data"][1]["volume"] is None

@pytest.mark.asyncio
async def test_get_stock_quotes(client):
    mock_stocks = [
        StockInfo(
            ticker="TCB",
            price=50000,
            market_cap=100000,
            company_name="Techcombank",
            foreign_buy_value=10.0,
            foreign_sell_value=8.0,
            current_room=300000,
            total_room=600000,
        ),
        StockInfo(
            ticker="VCB",
            price=90000,
            market_cap=200000,
            company_name="Vietcombank",
            foreign_buy_value=None,
            foreign_sell_value=None,
            current_room=None,
            total_room=None,
        )
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
        first_stock = next(stock for stock in data["stocks"] if stock["ticker"] == "TCB")
        second_stock = next(stock for stock in data["stocks"] if stock["ticker"] == "VCB")
        assert first_stock["foreign_buy_value"] == 10.0
        assert first_stock["foreign_sell_value"] == 8.0
        assert first_stock["current_room"] == 300000
        assert first_stock["total_room"] == 600000
        assert second_stock["foreign_buy_value"] is None
        assert second_stock["foreign_sell_value"] is None
        assert second_stock["current_room"] is None
        assert second_stock["total_room"] is None
