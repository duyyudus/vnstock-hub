"""
Stock-related API endpoints.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from app.services.vnstock_service import vnstock_service

router = APIRouter(prefix="/stocks", tags=["stocks"])


class StockResponse(BaseModel):
    """Response model for a single stock."""
    ticker: str
    price: float
    market_cap: float  # In billion VND
    company_name: str
    exchange: str = ""
    charter_capital: float = 0.0  # In billion VND
    pe_ratio: Optional[float] = None
    accumulated_value: Optional[float] = None  # In billion VND
    foreign_buy_value: Optional[float] = None  # In billion VND
    foreign_sell_value: Optional[float] = None  # In billion VND
    current_room: Optional[int] = None
    total_room: Optional[int] = None
    price_change_24h: Optional[float] = None  # Percentage
    price_change_1w: Optional[float] = None  # Percentage
    price_change_1m: Optional[float] = None  # Percentage
    price_change_6m: Optional[float] = None  # Percentage
    price_change_1y: Optional[float] = None  # Percentage
    price_change_2y: Optional[float] = None  # Percentage
    price_change_3y: Optional[float] = None  # Percentage
    industry: str = ""  # ICB Level 2 industry classification


class IndexStocksResponse(BaseModel):
    """Response model for stocks list of a specific index."""
    stocks: List[StockResponse]
    count: int
    index_symbol: str


class IndexInfo(BaseModel):
    """Information about a stock index."""
    symbol: str
    name: str
    group: Optional[str] = None
    description: Optional[str] = None


class IndexListResponse(BaseModel):
    """Response model for indices list."""
    indices: List[IndexInfo]
    count: int


class IndustryInfo(BaseModel):
    """Information about an ICB industry."""
    name: str
    en_name: str
    code: str


class IndustryListResponse(BaseModel):
    """Response model for industries list."""
    industries: List[IndustryInfo]
    count: int


class IndustryStocksResponse(BaseModel):
    """Response model for stocks list of a specific industry."""
    stocks: List[StockResponse]
    count: int
    industry_name: str


class IndexValueInfo(BaseModel):
    """Information about an index value."""
    symbol: str
    name: str
    value: float
    change: float
    change_value: float


class IndexValuesResponse(BaseModel):
    """Response model for index values."""
    indices: List[IndexValueInfo]
    count: int


class VolumeDataPoint(BaseModel):
    """A single data point for volume history."""
    date: str
    volume: int
    value: Optional[float] = None


class VolumeHistoryResponse(BaseModel):
    """Response model for volume history."""
    symbol: str
    company_name: str
    data: List[VolumeDataPoint]
    count: int
    sync_performed: bool = False
    sync_timed_out: bool = False
    sync_error: Optional[str] = None
    updated_through: Optional[str] = None
    repaired_missing_dates: int = 0


class PriceDataPoint(BaseModel):
    """A single data point for price history."""
    date: str
    close: float


class PriceHistoryResponse(BaseModel):
    """Response model for price history."""
    symbol: str
    company_name: str
    data: List[PriceDataPoint]
    count: int
    sync_performed: bool = False
    sync_timed_out: bool = False
    sync_error: Optional[str] = None
    updated_through: Optional[str] = None
    repaired_missing_dates: int = 0


class OhlcvDataPoint(BaseModel):
    """A single OHLCV data point."""
    date: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: float
    volume: Optional[int] = None


class OhlcvHistoryResponse(BaseModel):
    """Response model for full OHLCV history."""
    symbol: str
    company_name: str
    data: List[OhlcvDataPoint]
    count: int


class WeeklyPricePoint(BaseModel):
    """A single weekly price data point."""
    date: str
    close: float


class StockWeeklyPriceData(BaseModel):
    """Weekly price data for a single stock."""
    symbol: str
    ticker: str
    company_name: str
    prices: List[WeeklyPricePoint]


class StocksWeeklyPricesRequest(BaseModel):
    """Request model for fetching weekly prices for multiple stocks."""
    symbols: List[str]
    start_year: int
    include_benchmarks: bool = True


class StocksWeeklyPricesResponse(BaseModel):
    """Response model for weekly prices of multiple stocks."""
    stocks: List[StockWeeklyPriceData]
    benchmarks: dict
    start_date: str
    end_date: str
    is_stale: bool = False
    is_syncing: bool = False


class StockQuotesRequest(BaseModel):
    """Request model for fetching quotes for multiple stocks."""
    symbols: List[str]


class StockQuotesResponse(BaseModel):
    """Response model for stock quotes."""
    stocks: List[StockResponse]
    count: int


@router.get("/index-values", response_model=IndexValuesResponse)
async def get_index_values():
    """
    Get latest values for major market indices (VNINDEX, HNXINDEX, UPCOMINDEX, VN30, HNX30).
    """
    indices = await vnstock_service.get_index_values()
    return IndexValuesResponse(
        indices=[
            IndexValueInfo(
                symbol=idx.symbol,
                name=idx.name,
                value=idx.value,
                change=idx.change,
                change_value=idx.change_value
            )
            for idx in indices
        ],
        count=len(indices)
    )


@router.get("/indices", response_model=IndexListResponse)
async def get_indices():
    """
    Get all available stock indices.
    """
    indices = await vnstock_service.get_indices()
    return IndexListResponse(
        indices=[
            IndexInfo(
                symbol=idx.symbol,
                name=idx.name,
                group=idx.group,
                description=idx.description
            )
            for idx in indices
        ],
        count=len(indices)
    )


@router.get("/index/{index_symbol}", response_model=IndexStocksResponse)
async def get_stocks_by_index(index_symbol: str, limit: int = 1000):
    """
    Get stocks for a specific index.
    """
    stocks = await vnstock_service.get_index_stocks(index_symbol, limit)
    
    return IndexStocksResponse(
        stocks=[
            StockResponse(
                ticker=stock.ticker,
                price=stock.price,
                market_cap=stock.market_cap,
                company_name=stock.company_name,
                exchange=stock.exchange,
                charter_capital=stock.charter_capital,
                pe_ratio=stock.pe_ratio,
                accumulated_value=stock.accumulated_value,
                foreign_buy_value=stock.foreign_buy_value,
                foreign_sell_value=stock.foreign_sell_value,
                current_room=stock.current_room,
                total_room=stock.total_room,
                price_change_24h=stock.price_change_24h,
                price_change_1w=stock.price_change_1w,
                price_change_1m=stock.price_change_1m,
                price_change_6m=stock.price_change_6m,
                price_change_1y=stock.price_change_1y,
                price_change_2y=stock.price_change_2y,
                price_change_3y=stock.price_change_3y,
                industry=stock.industry
            )
            for stock in stocks
        ],
        count=len(stocks),
        index_symbol=index_symbol
    )


@router.get("/industries", response_model=IndustryListResponse)
async def get_industries():
    """
    Get all available ICB level 2 industries.
    """
    industries = await vnstock_service.get_industry_list()
    return IndustryListResponse(
        industries=[
            IndustryInfo(
                name=ind['icb_name'],
                en_name=ind['en_icb_name'],
                code=ind['icb_code']
            )
            for ind in industries
        ],
        count=len(industries)
    )


@router.get("/industry/{industry_name}", response_model=IndustryStocksResponse)
async def get_stocks_by_industry(industry_name: str, limit: int = 1000):
    """
    Get stocks for a specific industry.
    """
    stocks = await vnstock_service.get_industry_stocks(industry_name, limit)
    
    return IndustryStocksResponse(
        stocks=[
            StockResponse(
                ticker=stock.ticker,
                price=stock.price,
                market_cap=stock.market_cap,
                company_name=stock.company_name,
                exchange=stock.exchange,
                charter_capital=stock.charter_capital,
                pe_ratio=stock.pe_ratio,
                accumulated_value=stock.accumulated_value,
                foreign_buy_value=stock.foreign_buy_value,
                foreign_sell_value=stock.foreign_sell_value,
                current_room=stock.current_room,
                total_room=stock.total_room,
                price_change_24h=stock.price_change_24h,
                price_change_1w=stock.price_change_1w,
                price_change_1m=stock.price_change_1m,
                price_change_6m=stock.price_change_6m,
                price_change_1y=stock.price_change_1y,
                price_change_2y=stock.price_change_2y,
                price_change_3y=stock.price_change_3y,
                industry=stock.industry
            )
            for stock in stocks
        ],
        count=len(stocks),
        industry_name=industry_name
    )


@router.post("/quotes", response_model=StockQuotesResponse)
async def get_stock_quotes(request: StockQuotesRequest):
    """
    Get latest quotes for multiple stock symbols.
    """
    symbols = [symbol.strip().upper() for symbol in request.symbols if symbol and symbol.strip()]
    if not symbols:
        return StockQuotesResponse(stocks=[], count=0)
    unique_symbols = list(dict.fromkeys(symbols))
    stocks = await vnstock_service.get_symbol_stocks(unique_symbols)
    return StockQuotesResponse(
        stocks=[
            StockResponse(
                ticker=stock.ticker,
                price=stock.price,
                market_cap=stock.market_cap,
                company_name=stock.company_name,
                exchange=stock.exchange,
                charter_capital=stock.charter_capital,
                pe_ratio=stock.pe_ratio,
                accumulated_value=stock.accumulated_value,
                foreign_buy_value=stock.foreign_buy_value,
                foreign_sell_value=stock.foreign_sell_value,
                current_room=stock.current_room,
                total_room=stock.total_room,
                price_change_24h=stock.price_change_24h,
                price_change_1w=stock.price_change_1w,
                price_change_1m=stock.price_change_1m,
                price_change_6m=stock.price_change_6m,
                price_change_1y=stock.price_change_1y,
                price_change_2y=stock.price_change_2y,
                price_change_3y=stock.price_change_3y,
                industry=stock.industry
            )
            for stock in stocks
        ],
        count=len(stocks)
    )


class FinancialDataResponse(BaseModel):
    """Response model for financial data."""
    symbol: str
    data: List[dict]
    count: int


@router.get("/finance/{symbol}/income-statement", response_model=FinancialDataResponse)
async def get_income_statement(symbol: str):
    """
    Get income statement data for a specific stock.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'VIC', 'VNM')
    """
    data = await vnstock_service.get_income_statement(symbol, lang='en')
    return FinancialDataResponse(
        symbol=symbol,
        data=data,
        count=len(data)
    )


@router.get("/finance/{symbol}/balance-sheet", response_model=FinancialDataResponse)
async def get_balance_sheet(symbol: str):
    """
    Get balance sheet data for a specific stock.
    
    Args:
        symbol: Stock ticker symbol
    """
    data = await vnstock_service.get_balance_sheet(symbol, lang='en')
    return FinancialDataResponse(
        symbol=symbol,
        data=data,
        count=len(data)
    )


@router.get("/finance/{symbol}/cash-flow", response_model=FinancialDataResponse)
async def get_cash_flow(symbol: str):
    """
    Get cash flow statement data for a specific stock.
    
    Args:
        symbol: Stock ticker symbol
    """
    data = await vnstock_service.get_cash_flow(symbol, lang='en')
    return FinancialDataResponse(
        symbol=symbol,
        data=data,
        count=len(data)
    )


@router.get("/finance/{symbol}/ratios", response_model=FinancialDataResponse)
async def get_financial_ratios(symbol: str):
    """
    Get financial ratios for a specific stock.
    Includes P/E, P/B, P/S, ROE, ROA, etc.
    
    Args:
        symbol: Stock ticker symbol
    """
    data = await vnstock_service.get_financial_ratios(symbol, lang='en')
    return FinancialDataResponse(
        symbol=symbol,
        data=data,
        count=len(data)
    )


@router.get("/company/{symbol}/overview", response_model=FinancialDataResponse)
async def get_company_overview(symbol: str):
    """
    Get company overview for a specific stock.
    """
    data = await vnstock_service.get_company_overview(symbol)
    return FinancialDataResponse(
        symbol=symbol,
        data=data,
        count=len(data)
    )


@router.get("/company/{symbol}/shareholders", response_model=FinancialDataResponse)
async def get_shareholders(symbol: str):
    """
    Get major shareholders for a specific stock.
    """
    data = await vnstock_service.get_shareholders(symbol)
    return FinancialDataResponse(
        symbol=symbol,
        data=data,
        count=len(data)
    )


@router.get("/company/{symbol}/officers", response_model=FinancialDataResponse)
async def get_officers(symbol: str):
    """
    Get company officers for a specific stock.
    """
    data = await vnstock_service.get_officers(symbol)
    return FinancialDataResponse(
        symbol=symbol,
        data=data,
        count=len(data)
    )


@router.get("/company/{symbol}/subsidiaries", response_model=FinancialDataResponse)
async def get_subsidiaries(symbol: str):
    """
    Get subsidiaries and relevant companies for a specific stock.
    """
    data = await vnstock_service.get_subsidiaries(symbol)
    return FinancialDataResponse(
        symbol=symbol,
        data=data,
        count=len(data)
    )


@router.get("/history/{symbol}/volume", response_model=VolumeHistoryResponse)
async def get_volume_history(symbol: str, days: int = 30):
    """
    Get volume history for a specific stock.

    Args:
        symbol: Stock ticker symbol
        days: Number of days to fetch (default: 30)

    Returns:
        Volume history data points with date, volume, and accumulated value
    """
    result = await vnstock_service.get_volume_history(symbol, days=days)
    return VolumeHistoryResponse(
        symbol=result["symbol"],
        company_name=result["company_name"],
        data=[VolumeDataPoint(**point) for point in result["data"]],
        count=len(result["data"]),
        sync_performed=bool(result.get("sync_performed", False)),
        sync_timed_out=bool(result.get("sync_timed_out", False)),
        sync_error=result.get("sync_error"),
        updated_through=result.get("updated_through"),
        repaired_missing_dates=int(result.get("repaired_missing_dates", 0)),
    )


@router.get("/history/{symbol}/price", response_model=PriceHistoryResponse)
async def get_price_history(symbol: str, days: int = 30):
    """
    Get price history for a specific stock.

    Args:
        symbol: Stock ticker symbol
        days: Number of days to fetch (default: 30)

    Returns:
        Price history data points with date and close price in VND
    """
    result = await vnstock_service.get_price_history(symbol, days=days)
    return PriceHistoryResponse(
        symbol=result["symbol"],
        company_name=result["company_name"],
        data=[PriceDataPoint(**point) for point in result["data"]],
        count=len(result["data"]),
        sync_performed=bool(result.get("sync_performed", False)),
        sync_timed_out=bool(result.get("sync_timed_out", False)),
        sync_error=result.get("sync_error"),
        updated_through=result.get("updated_through"),
        repaired_missing_dates=int(result.get("repaired_missing_dates", 0)),
    )


@router.get("/history/{symbol}/ohlcv", response_model=OhlcvHistoryResponse)
async def get_price_history_ohlcv(symbol: str):
    """
    Get full OHLCV history for a specific stock from DB cache.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Full OHLCV data points sorted from latest to earliest
    """
    result = await vnstock_service.get_price_history_ohlcv(symbol)
    return OhlcvHistoryResponse(
        symbol=result["symbol"],
        company_name=result["company_name"],
        data=[OhlcvDataPoint(**point) for point in result["data"]],
        count=len(result["data"]),
    )


@router.post("/weekly-prices", response_model=StocksWeeklyPricesResponse)
async def get_stocks_weekly_prices(request: StocksWeeklyPricesRequest):
    """
    Get weekly price data for multiple stocks.
    Used for growth chart visualization.

    Args:
        request: Request with list of symbols, start year, and benchmark options

    Returns:
        Weekly prices for each stock normalized for chart display
    """
    result = await vnstock_service.get_stocks_weekly_prices(
        symbols=request.symbols,
        start_year=request.start_year,
        include_benchmarks=request.include_benchmarks
    )
    return StocksWeeklyPricesResponse(**result)
