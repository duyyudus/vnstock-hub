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
    charter_capital: float = 0.0  # In billion VND
    pe_ratio: Optional[float] = None
    price_change_24h: Optional[float] = None  # Percentage
    price_change_1w: Optional[float] = None  # Percentage
    price_change_1m: Optional[float] = None  # Percentage
    price_change_1y: Optional[float] = None  # Percentage


class VN100Response(BaseModel):
    """Response model for VN-100 stocks list."""
    stocks: List[StockResponse]
    count: int


class VN30Response(BaseModel):
    """Response model for VN-30 stocks list."""
    stocks: List[StockResponse]
    count: int


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
                charter_capital=stock.charter_capital,
                pe_ratio=stock.pe_ratio,
                price_change_24h=stock.price_change_24h,
                price_change_1w=stock.price_change_1w,
                price_change_1m=stock.price_change_1m,
                price_change_1y=stock.price_change_1y
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
                charter_capital=stock.charter_capital,
                pe_ratio=stock.pe_ratio,
                price_change_24h=stock.price_change_24h,
                price_change_1w=stock.price_change_1w,
                price_change_1m=stock.price_change_1m,
                price_change_1y=stock.price_change_1y
            )
            for stock in stocks
        ],
        count=len(stocks),
        industry_name=industry_name
    )


@router.get("/vn100", response_model=VN100Response)
async def get_vn100_stocks():
    """
    Get VN-100 stocks (top 100 stocks by market cap).
    
    Returns:
        List of stocks with ticker, price, and market cap data.
    """
    stocks = await vnstock_service.get_vn100_stocks()
    
    return VN100Response(
        stocks=[
            StockResponse(
                ticker=stock.ticker,
                price=stock.price,
                market_cap=stock.market_cap,
                company_name=stock.company_name,
                charter_capital=stock.charter_capital,
                pe_ratio=stock.pe_ratio,
                price_change_24h=stock.price_change_24h,
                price_change_1w=stock.price_change_1w,
                price_change_1m=stock.price_change_1m,
                price_change_1y=stock.price_change_1y
            )
            for stock in stocks
        ],
        count=len(stocks)
    )


@router.get("/vn30", response_model=VN30Response)
async def get_vn30_stocks():
    """
    Get VN-30 stocks (top 30 stocks by market cap and liquidity).
    
    Returns:
        List of stocks with ticker, price, and market cap data.
    """
    stocks = await vnstock_service.get_vn30_stocks()
    
    return VN30Response(
        stocks=[
            StockResponse(
                ticker=stock.ticker,
                price=stock.price,
                market_cap=stock.market_cap,
                company_name=stock.company_name,
                charter_capital=stock.charter_capital,
                pe_ratio=stock.pe_ratio,
                price_change_24h=stock.price_change_24h,
                price_change_1w=stock.price_change_1w,
                price_change_1m=stock.price_change_1m,
                price_change_1y=stock.price_change_1y
            )
            for stock in stocks
        ],
        count=len(stocks)
    )

