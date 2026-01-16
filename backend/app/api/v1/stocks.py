"""
Stock-related API endpoints.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from app.services.vnstock_service import vnstock_service

router = APIRouter(prefix="/stocks", tags=["stocks"])


class StockResponse(BaseModel):
    """Response model for a single stock."""
    ticker: str
    price: float
    market_cap: float  # In billion VND
    company_name: str


class VN100Response(BaseModel):
    """Response model for VN-100 stocks list."""
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
                company_name=stock.company_name
            )
            for stock in stocks
        ],
        count=len(stocks)
    )
