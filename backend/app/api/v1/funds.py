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


class FundPerformanceResponse(BaseModel):
    """Response model for fund performance comparison data."""
    funds: List[dict]
    benchmarks: dict
    common_start_date: Optional[str] = None
    last_updated: Optional[str] = None
    is_stale: bool = False
    is_syncing: bool = False


class FundOverviewHoldingFund(BaseModel):
    """Fund-level contribution to an aggregate stock holding."""
    symbol: str
    name: Optional[str] = None
    allocation: float
    holding_updated_at: Optional[str] = None


class FundOverviewStock(BaseModel):
    """Aggregate stock holding across funds."""
    ticker: str
    company_name: Optional[str] = None
    sector: str
    total_allocation: float
    fund_count: int
    funds: List[FundOverviewHoldingFund]


class FundOverviewSector(BaseModel):
    """Sector group for aggregate fund stock holdings."""
    sector: str
    total_allocation: float
    stock_count: int
    stocks: List[FundOverviewStock]


class FundOverviewResponse(BaseModel):
    """Response model for aggregate latest fund holdings overview."""
    sectors: List[FundOverviewSector]
    fund_count: int
    stock_count: int
    last_updated: Optional[str] = None


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


@router.get("/performance", response_model=FundPerformanceResponse)
async def get_fund_performance():
    """
    Get aggregated fund performance data for comparison charts.
    Includes normalized NAV, periodic returns (YTD, 1Y, 3Y, 5Y), and risk metrics.
    Data is cached daily.

    Returns:
        FundPerformanceResponse with funds, benchmarks (VN-Index & VN30),
        common start date, and cache timestamp.
    """
    data = await vnstock_service.get_fund_performance_data()
    return FundPerformanceResponse(**data)


@router.get("/overview", response_model=FundOverviewResponse)
async def get_fund_overview():
    """
    Get aggregated top stock holdings across all funds from cached DB data only.

    Returns:
        Sector-grouped stock holdings sorted by summed allocation across funds.
    """
    data = await vnstock_service.get_fund_overview()
    return FundOverviewResponse(**data)


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
