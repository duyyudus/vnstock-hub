"""Vnstock service package facade."""
from __future__ import annotations

from typing import List, Dict, Any

from app.core.config import settings

from .core import logger, retry_with_backoff, async_retry_with_backoff, RateLimitError
from .models import IndexValue, StockInfo
from .indices import IndicesService
from .stocks import StocksService
from .stock_metadata import StockMetadataService
from .history import HistoryService
from .finance import FinanceService
from .company import CompanyService
from .funds import FundsService
from .symbols import VALID_GROUPS


class VnstockService:
    """Facade service that composes vnstock sub-services."""

    VALID_GROUPS = VALID_GROUPS

    def __init__(self) -> None:
        # Initialize vnstock API key if provided
        if settings.vnstock_api_key:
            try:
                import vnstock
                vnstock.change_api_key(settings.vnstock_api_key)
                logger.info("vnstock API key configured")
            except Exception as e:
                logger.error(f"Error configuring vnstock API key: {e}")

        self.metadata = StockMetadataService()
        self.history = HistoryService()
        self.indices = IndicesService()
        self.funds = FundsService()
        self.finance = FinanceService()
        self.company = CompanyService()
        self.stocks = StocksService(metadata=self.metadata, history=self.history)

    # Indices
    async def sync_indices(self) -> None:
        return await self.indices.sync_indices()

    async def get_indices(self):
        return await self.indices.get_indices()

    async def get_index_values(self, symbols: List[str] | None = None) -> List[IndexValue]:
        return await self.indices.get_index_values(symbols)

    # Stocks
    async def get_index_stocks(self, index_symbol: str, limit: int = 100) -> List[StockInfo]:
        return await self.stocks.get_index_stocks(index_symbol, limit)

    async def get_industry_list(self) -> List[Dict[str, str]]:
        return await self.stocks.get_industry_list()

    async def get_industry_stocks(self, industry_name: str, limit: int = 100) -> List[StockInfo]:
        return await self.stocks.get_industry_stocks(industry_name, limit)

    # Finance
    async def get_income_statement(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        return await self.finance.get_income_statement(symbol, period=period, lang=lang)

    async def get_balance_sheet(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        return await self.finance.get_balance_sheet(symbol, period=period, lang=lang)

    async def get_cash_flow(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        return await self.finance.get_cash_flow(symbol, period=period, lang=lang)

    async def get_financial_ratios(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        return await self.finance.get_financial_ratios(symbol, period=period, lang=lang)

    # Company
    async def get_company_overview(self, symbol: str, source: str = "auto") -> List[Dict[str, Any]]:
        return await self.company.get_company_overview(symbol, source=source)

    async def get_shareholders(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.company.get_shareholders(symbol)

    async def get_officers(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.company.get_officers(symbol)

    async def get_subsidiaries(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.company.get_subsidiaries(symbol)

    # History
    async def get_volume_history(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        return await self.history.get_volume_history(symbol, days=days)

    async def get_stocks_weekly_prices(
        self,
        symbols: List[str],
        start_year: int,
        include_benchmarks: bool = True
    ) -> Dict[str, Any]:
        return await self.history.get_stocks_weekly_prices(
            symbols=symbols,
            start_year=start_year,
            include_benchmarks=include_benchmarks
        )

    # Funds
    async def get_fund_listing(self, fund_type: str = "") -> List[Dict[str, Any]]:
        return await self.funds.get_fund_listing(fund_type=fund_type)

    async def get_fund_performance_data(self) -> Dict[str, Any]:
        return await self.funds.get_fund_performance_data()

    async def get_fund_nav_report(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.funds.get_fund_nav_report(symbol)

    async def get_fund_top_holding(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.funds.get_fund_top_holding(symbol)

    async def get_fund_industry_holding(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.funds.get_fund_industry_holding(symbol)

    async def get_fund_asset_holding(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.funds.get_fund_asset_holding(symbol)


# Singleton instance
vnstock_service = VnstockService()

__all__ = [
    "IndexValue",
    "StockInfo",
    "VnstockService",
    "vnstock_service",
    "retry_with_backoff",
    "async_retry_with_backoff",
    "RateLimitError",
]
