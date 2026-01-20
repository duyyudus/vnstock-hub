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

