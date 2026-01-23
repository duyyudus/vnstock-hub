"""
Fund-related API endpoints.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from app.services.vnstock_service import vnstock_service

router = APIRouter(prefix="/funds", tags=["funds"])


class FundDataResponse(BaseModel):
    """Generic response model for fund data."""
    symbol: Optional[str] = None
    data: List[dict]
    count: int


@router.get("/listing", response_model=FundDataResponse)
async def get_fund_listing(fund_type: str = ""):
    """
    Get list of all available funds.

    Args:
        fund_type: Optional filter by fund type (e.g., "STOCK", "BOND", "BALANCED")

    Returns:
        List of funds with metadata (symbol, name, type, owner, etc.)
    """
    data = await vnstock_service.get_fund_listing(fund_type=fund_type)
    return FundDataResponse(
        data=data,
        count=len(data)
    )


@router.get("/{symbol}/nav-report", response_model=FundDataResponse)
async def get_fund_nav_report(symbol: str):
    """
    Get NAV (Net Asset Value) history for a specific fund.

    Args:
        symbol: Fund symbol (e.g., "SSISCA")

    Returns:
        NAV history data points with date and NAV value
    """
    data = await vnstock_service.get_fund_nav_report(symbol)
    return FundDataResponse(
        symbol=symbol,
        data=data,
        count=len(data)
    )


@router.get("/{symbol}/top-holding", response_model=FundDataResponse)
async def get_fund_top_holding(symbol: str):
    """
    Get top stock holdings for a specific fund.

    Args:
        symbol: Fund symbol

    Returns:
        List of top holdings with ticker, allocation percentage, etc.
    """
    data = await vnstock_service.get_fund_top_holding(symbol)
    return FundDataResponse(
        symbol=symbol,
        data=data,
        count=len(data)
    )


@router.get("/{symbol}/industry-holding", response_model=FundDataResponse)
async def get_fund_industry_holding(symbol: str):
    """
    Get industry allocation for a specific fund.

    Args:
        symbol: Fund symbol

    Returns:
        List of industry allocations with industry name and percentage
    """
    data = await vnstock_service.get_fund_industry_holding(symbol)
    return FundDataResponse(
        symbol=symbol,
        data=data,
        count=len(data)
    )


@router.get("/{symbol}/asset-holding", response_model=FundDataResponse)
async def get_fund_asset_holding(symbol: str):
    """
    Get asset type allocation for a specific fund.

    Args:
        symbol: Fund symbol

    Returns:
        List of asset type allocations with type and percentage
    """
    data = await vnstock_service.get_fund_asset_holding(symbol)
    return FundDataResponse(
        symbol=symbol,
        data=data,
        count=len(data)
    )
