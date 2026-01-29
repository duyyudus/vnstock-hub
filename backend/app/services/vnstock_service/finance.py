from __future__ import annotations

from typing import List, Dict, Any
import asyncio
import pandas as pd

from .core import (
    frontend_executor,
    logger,
    api_circuit_breaker,
    CircuitOpenError,
    _record_rate_limit,
    _is_rate_limit_error,
    _flatten_columns,
)


class FinanceService:
    """Financial statements and ratios."""

    async def get_income_statement(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        """Fetch income statement data for a given stock symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(frontend_executor, self._fetch_income_statement_sync, symbol, period, lang)

    def _fetch_income_statement_sync(self, symbol: str, period: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch income statement synchronously."""
        from vnstock import Vnstock

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch income statement for {symbol}")

        try:
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            df = s.finance.income_statement(period=period, lang=lang)
            api_circuit_breaker.record_success()

            if df is not None and not df.empty:
                df = _flatten_columns(df)
                # Convert to list of dicts, handling NaN values
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching income statement for {symbol}: {e}")
            logger.warning(f"Error fetching income statement for {symbol}: {e}")
            return []

    async def get_balance_sheet(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        """Fetch balance sheet data for a given stock symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(frontend_executor, self._fetch_balance_sheet_sync, symbol, period, lang)

    def _fetch_balance_sheet_sync(self, symbol: str, period: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch balance sheet synchronously."""
        from vnstock import Vnstock

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch balance sheet for {symbol}")

        try:
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            df = s.finance.balance_sheet(period=period, lang=lang)
            api_circuit_breaker.record_success()

            if df is not None and not df.empty:
                df = _flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching balance sheet for {symbol}: {e}")
            logger.warning(f"Error fetching balance sheet for {symbol}: {e}")
            return []

    async def get_cash_flow(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        """Fetch cash flow statement data for a given stock symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(frontend_executor, self._fetch_cash_flow_sync, symbol, period, lang)

    def _fetch_cash_flow_sync(self, symbol: str, period: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch cash flow synchronously."""
        from vnstock import Vnstock

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch cash flow for {symbol}")

        try:
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            df = s.finance.cash_flow(period=period, lang=lang)
            api_circuit_breaker.record_success()

            if df is not None and not df.empty:
                df = _flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching cash flow for {symbol}: {e}")
            logger.warning(f"Error fetching cash flow for {symbol}: {e}")
            return []

    async def get_financial_ratios(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        """Fetch financial ratios for a given stock symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(frontend_executor, self._fetch_financial_ratios_sync, symbol, period, lang)

    def _fetch_financial_ratios_sync(self, symbol: str, period: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch financial ratios synchronously."""
        from vnstock import Vnstock

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch financial ratios for {symbol}")

        try:
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            df = s.finance.ratio(period=period, lang=lang)
            api_circuit_breaker.record_success()

            if df is not None and not df.empty:
                df = _flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching financial ratios for {symbol}: {e}")
            logger.warning(f"Error fetching financial ratios for {symbol}: {e}")
            return []
