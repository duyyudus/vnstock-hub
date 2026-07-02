"""Vnstock service package facade."""
from __future__ import annotations

from datetime import date
from typing import List, Dict, Any

from app.lib.vnstock_runtime import install_vnstock_aliases

install_vnstock_aliases()

from .core import logger, retry_with_backoff, async_retry_with_backoff, RateLimitError
from .models import IndexContribution, IndexValue, StockInfo
from .indices import IndicesService
from .stocks import StocksService
from .stock_metadata import StockMetadataService
from .history import HistoryService
from .history_sync import HistorySyncService
from .finance_sync import FinanceDataSyncService
from .company_sync import CompanyDataSyncService
from .scheduler import ScheduledSyncService
from .finance import FinanceService
from .company import CompanyService
from .funds import FundsService
from .symbols import VALID_GROUPS


class VnstockService:
    """Facade service that composes vnstock sub-services."""

    VALID_GROUPS = VALID_GROUPS

    def __init__(self) -> None:
        self.history = HistoryService()
        self.finance = FinanceService()
        self.company = CompanyService()
        self.metadata = StockMetadataService(finance_service=self.finance)
        self.history_sync = HistorySyncService(history=self.history)
        self.history.set_on_demand_history_sync_handler(
            self.history_sync.sync_symbol_history_for_request
        )
        self.history.set_weekly_history_sync_trigger_handler(
            self._run_history_sync_for_symbols
        )
        self.finance_data_sync = FinanceDataSyncService(finance=self.finance)
        self.company_data_sync = CompanyDataSyncService(company=self.company)
        self.scheduler = ScheduledSyncService(
            run_history_sync=self.run_history_sync,
            run_history_audit_sync=self.run_history_audit_sync,
            run_history_repair_sync=self.run_history_repair_sync,
            run_finance_sync=self.run_finance_sync,
            run_company_sync=self.run_company_sync,
        )
        self.indices = IndicesService()
        self.funds = FundsService()
        self.stocks = StocksService(metadata=self.metadata, history=self.history)

    async def _run_history_sync_for_symbols(self, symbols: List[str]) -> Dict[str, Any]:
        return await self.history_sync.run_sync(symbols=symbols)

    async def start_background_tasks(self) -> None:
        """Start long-running background workers."""
        await self.history.start_background_workers()
        await self.history_sync.start_background_tasks()
        await self.finance_data_sync.start_background_tasks()
        await self.company_data_sync.start_background_tasks()
        await self.scheduler.start_background_tasks()

    async def stop_background_tasks(self) -> None:
        """Stop long-running background workers."""
        await self.scheduler.stop_background_tasks()
        await self.company_data_sync.stop_background_tasks()
        await self.finance_data_sync.stop_background_tasks()
        await self.history_sync.stop_background_tasks()
        await self.history.stop_background_workers()

    # Indices
    async def sync_indices(self) -> None:
        return await self.indices.sync_indices()

    async def get_indices(self):
        return await self.indices.get_indices()

    async def get_index_values(self, symbols: List[str] | None = None) -> List[IndexValue]:
        return await self.indices.get_index_values(symbols)

    # Stocks
    async def get_index_stocks(
        self,
        index_symbol: str,
        limit: int = 100,
        range_start: date | None = None,
        range_end: date | None = None,
    ) -> List[StockInfo]:
        return await self.stocks.get_index_stocks(index_symbol, limit, range_start, range_end)

    async def get_index_contribution(self, index_symbol: str) -> IndexContribution:
        index_symbol_key = index_symbol.upper()
        index_values = await self.indices.get_index_values([index_symbol_key])
        index_value = index_values[0] if index_values else None
        return await self.stocks.get_index_contribution(
            index_symbol=index_symbol_key,
            index_value=index_value,
            company_overview_fetcher=self.company.get_company_overview,
        )

    async def get_industry_list(self) -> List[Dict[str, str]]:
        return await self.stocks.get_industry_list()

    async def get_industry_stocks(
        self,
        industry_name: str,
        limit: int = 100,
        range_start: date | None = None,
        range_end: date | None = None,
    ) -> List[StockInfo]:
        return await self.stocks.get_industry_stocks(industry_name, limit, range_start, range_end)

    async def get_symbol_stocks(
        self,
        symbols: List[str],
        limit: int = 100,
        range_start: date | None = None,
        range_end: date | None = None,
    ) -> List[StockInfo]:
        return await self.stocks.get_symbol_stocks(symbols, limit, range_start, range_end)

    # Finance
    async def get_income_statement(self, symbol: str, lang: str = 'en') -> List[Dict[str, Any]]:
        return await self.finance.get_income_statement(symbol, lang=lang)

    async def get_balance_sheet(self, symbol: str, lang: str = 'en') -> List[Dict[str, Any]]:
        return await self.finance.get_balance_sheet(symbol, lang=lang)

    async def get_cash_flow(self, symbol: str, lang: str = 'en') -> List[Dict[str, Any]]:
        return await self.finance.get_cash_flow(symbol, lang=lang)

    async def get_financial_ratios(self, symbol: str, lang: str = 'en') -> List[Dict[str, Any]]:
        return await self.finance.get_financial_ratios(symbol, lang=lang)

    # Company
    async def get_company_overview(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.company.get_company_overview(symbol)

    async def get_shareholders(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.company.get_shareholders(symbol)

    async def get_officers(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.company.get_officers(symbol)

    async def get_subsidiaries(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.company.get_subsidiaries(symbol)

    async def get_ownership(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.company.get_ownership(symbol)

    async def get_capital_history(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.company.get_capital_history(symbol)

    async def get_news(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.company.get_news(symbol)

    async def get_events(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.company.get_events(symbol)

    async def get_insider_trading(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.company.get_insider_trading(symbol)

    # History
    async def get_volume_history(self, symbol: str, days: int = 30, auto_sync: bool = True) -> Dict[str, Any]:
        return await self.history.get_volume_history(symbol, days=days, auto_sync=auto_sync)

    async def get_price_history(self, symbol: str, days: int = 30, auto_sync: bool = True) -> Dict[str, Any]:
        return await self.history.get_price_history(symbol, days=days, auto_sync=auto_sync)

    async def get_price_history_ohlcv(self, symbol: str) -> Dict[str, Any]:
        return await self.history.get_price_history_ohlcv(symbol)

    async def get_stocks_weekly_prices(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        include_benchmarks: bool = True
    ) -> Dict[str, Any]:
        return await self.history.get_stocks_weekly_prices(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            include_benchmarks=include_benchmarks
        )

    async def get_stocks_volume_series(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> Dict[str, Any]:
        return await self.history.get_stocks_volume_series(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )

    # History Sync
    async def run_history_sync(
        self,
        force_restart: bool = False,
        symbols: List[str] | None = None,
        index_symbol: str | None = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        return await self.history_sync.run_sync(
            force_restart=force_restart,
            symbols=symbols,
            index_symbol=index_symbol,
            force_refresh=force_refresh,
        )

    async def cancel_history_sync(self) -> Dict[str, Any]:
        return await self.history_sync.cancel_running_sync()

    async def run_history_audit_sync(
        self,
        symbols: List[str] | None,
        start_date: str,
        end_date: str,
        auto_repair: bool = False,
        index_symbol: str | None = None,
    ) -> Dict[str, Any]:
        from datetime import datetime
        parsed_start = datetime.strptime(start_date, "%Y-%m-%d").date()
        parsed_end = datetime.strptime(end_date, "%Y-%m-%d").date()
        return await self.history_sync.run_audit_sync(
            symbols=symbols,
            start_date=parsed_start,
            end_date=parsed_end,
            auto_repair=auto_repair,
            index_symbol=index_symbol,
        )

    async def cancel_history_audit_sync(self) -> Dict[str, Any]:
        return await self.history_sync.cancel_running_audit()

    async def run_history_repair_sync(
        self,
        symbols: List[str] | None,
        start_date: str,
        end_date: str,
        index_symbol: str | None = None,
    ) -> Dict[str, Any]:
        from datetime import datetime
        parsed_start = datetime.strptime(start_date, "%Y-%m-%d").date()
        parsed_end = datetime.strptime(end_date, "%Y-%m-%d").date()
        return await self.history_sync.run_repair_sync(
            symbols=symbols,
            start_date=parsed_start,
            end_date=parsed_end,
            index_symbol=index_symbol,
        )

    async def cancel_history_repair_sync(self) -> Dict[str, Any]:
        return await self.history_sync.cancel_running_repair()

    # Finance Sync
    async def run_finance_sync(
        self,
        force_restart: bool = False,
        symbols: List[str] | None = None,
        index_symbol: str | None = None,
        quick_sync: bool = False,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        return await self.finance_data_sync.run_sync(
            force_restart=force_restart,
            symbols=symbols,
            index_symbol=index_symbol,
            quick_sync=quick_sync,
            force_refresh=force_refresh,
        )

    async def cancel_finance_sync(self) -> Dict[str, Any]:
        return await self.finance_data_sync.cancel_running_sync()

    # Company Sync
    async def run_company_sync(
        self,
        force_restart: bool = False,
        symbols: List[str] | None = None,
        index_symbol: str | None = None,
        quick_sync: bool = False,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        return await self.company_data_sync.run_sync(
            force_restart=force_restart,
            symbols=symbols,
            index_symbol=index_symbol,
            quick_sync=quick_sync,
            force_refresh=force_refresh,
        )

    async def cancel_company_sync(self) -> Dict[str, Any]:
        return await self.company_data_sync.cancel_running_sync()

    # Funds
    async def get_fund_listing(self, fund_type: str = "") -> List[Dict[str, Any]]:
        return await self.funds.get_fund_listing(fund_type=fund_type)

    async def get_fund_performance_data(self) -> Dict[str, Any]:
        return await self.funds.get_fund_performance_data()

    async def run_fund_sync(self, category: str = "ALL") -> Dict[str, Any]:
        return await self.funds.run_fund_sync(category=category)

    async def get_fund_overview(self) -> Dict[str, Any]:
        return await self.funds.get_fund_overview()

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
