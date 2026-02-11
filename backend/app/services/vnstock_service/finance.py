from __future__ import annotations

from typing import List, Dict, Any, Callable
from datetime import datetime, date
import asyncio
import pandas as pd

from sqlalchemy import select

from app.db.database import async_session
from app.db.models import StockFinancialDataCache

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
    """Financial statements and ratios with DB-first caching."""

    DATA_TYPE_INCOME = "income"
    DATA_TYPE_BALANCE_SHEET = "balance_sheet"
    DATA_TYPE_CASHFLOW = "cashflow"
    DATA_TYPE_RATIOS = "ratios"

    SUPPORTED_PERIODS = {"quarter", "year"}

    def __init__(self) -> None:
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._refresh_locks_guard = asyncio.Lock()
        self._fetchers: Dict[str, Callable[[str, str, str], List[Dict[str, Any]]]] = {
            self.DATA_TYPE_INCOME: self._fetch_income_statement_sync,
            self.DATA_TYPE_BALANCE_SHEET: self._fetch_balance_sheet_sync,
            self.DATA_TYPE_CASHFLOW: self._fetch_cash_flow_sync,
            self.DATA_TYPE_RATIOS: self._fetch_financial_ratios_sync,
        }

    async def get_income_statement(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        """Fetch income statement data for a given stock symbol."""
        return await self._get_financial_dataset(
            symbol=symbol,
            data_type=self.DATA_TYPE_INCOME,
            period=period,
            lang=lang,
        )

    async def get_balance_sheet(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        """Fetch balance sheet data for a given stock symbol."""
        return await self._get_financial_dataset(
            symbol=symbol,
            data_type=self.DATA_TYPE_BALANCE_SHEET,
            period=period,
            lang=lang,
        )

    async def get_cash_flow(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        """Fetch cash flow statement data for a given stock symbol."""
        return await self._get_financial_dataset(
            symbol=symbol,
            data_type=self.DATA_TYPE_CASHFLOW,
            period=period,
            lang=lang,
        )

    async def get_financial_ratios(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        """Fetch financial ratios for a given stock symbol."""
        return await self._get_financial_dataset(
            symbol=symbol,
            data_type=self.DATA_TYPE_RATIOS,
            period=period,
            lang=lang,
        )

    async def refresh_financial_dataset(
        self,
        symbol: str,
        data_type: str,
        period: str = 'quarter',
        lang: str = 'en',
        raise_on_failure: bool = False,
    ) -> List[Dict[str, Any]]:
        """Refresh a financial dataset from source and upsert cache, with stale fallback."""
        symbol_key = self._normalize_symbol(symbol)
        data_type_key = self._normalize_data_type(data_type)
        period_key = self._normalize_period(period)
        lang_key = self._normalize_lang(lang)

        lock = await self._get_refresh_lock(symbol_key, data_type_key, period_key, lang_key)
        async with lock:
            cached = await self._get_cache_entry(
                symbol=symbol_key,
                data_type=data_type_key,
                period=period_key,
                lang=lang_key,
            )
            if cached and not self._is_stale(cached.updated_at):
                return self._deserialize_cached_records(cached.data)

            try:
                records = await self._fetch_from_source(
                    symbol=symbol_key,
                    data_type=data_type_key,
                    period=period_key,
                    lang=lang_key,
                )
                await self._upsert_cache_entry(
                    symbol=symbol_key,
                    data_type=data_type_key,
                    period=period_key,
                    lang=lang_key,
                    data=records,
                )
                return records
            except Exception as e:
                logger.warning(
                    f"Finance refresh failed for {symbol_key} ({data_type_key}, {period_key}, {lang_key}): {e}"
                )
                if raise_on_failure:
                    raise
                if cached:
                    return self._deserialize_cached_records(cached.data)
                return []

    async def _get_financial_dataset(
        self,
        symbol: str,
        data_type: str,
        period: str,
        lang: str,
    ) -> List[Dict[str, Any]]:
        """DB-first financial data retrieval with quarterly stale refresh."""
        symbol_key = self._normalize_symbol(symbol)
        data_type_key = self._normalize_data_type(data_type)
        period_key = self._normalize_period(period)
        lang_key = self._normalize_lang(lang)

        cached = await self._get_cache_entry(
            symbol=symbol_key,
            data_type=data_type_key,
            period=period_key,
            lang=lang_key,
        )

        if cached and not self._is_stale(cached.updated_at):
            return self._deserialize_cached_records(cached.data)

        return await self.refresh_financial_dataset(
            symbol=symbol_key,
            data_type=data_type_key,
            period=period_key,
            lang=lang_key,
            raise_on_failure=False,
        )

    async def _fetch_from_source(
        self,
        symbol: str,
        data_type: str,
        period: str,
        lang: str,
    ) -> List[Dict[str, Any]]:
        fetcher = self._fetchers.get(data_type)
        if fetcher is None:
            raise ValueError(f"Unsupported finance data_type: {data_type}")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(frontend_executor, fetcher, symbol, period, lang)

    async def _get_refresh_lock(self, symbol: str, data_type: str, period: str, lang: str) -> asyncio.Lock:
        lock_key = f"{symbol}:{data_type}:{period}:{lang}"
        async with self._refresh_locks_guard:
            lock = self._refresh_locks.get(lock_key)
            if lock is None:
                lock = asyncio.Lock()
                self._refresh_locks[lock_key] = lock
            return lock

    async def _get_cache_entry(
        self,
        symbol: str,
        data_type: str,
        period: str,
        lang: str,
    ) -> StockFinancialDataCache | None:
        async with async_session() as session:
            stmt = select(StockFinancialDataCache).where(
                StockFinancialDataCache.symbol == symbol,
                StockFinancialDataCache.data_type == data_type,
                StockFinancialDataCache.period == period,
                StockFinancialDataCache.lang == lang,
            )
            return (await session.execute(stmt)).scalar_one_or_none()

    async def _upsert_cache_entry(
        self,
        symbol: str,
        data_type: str,
        period: str,
        lang: str,
        data: List[Dict[str, Any]],
    ) -> None:
        payload = self._serialize_records_for_cache(data)

        async with async_session() as session:
            stmt = select(StockFinancialDataCache).where(
                StockFinancialDataCache.symbol == symbol,
                StockFinancialDataCache.data_type == data_type,
                StockFinancialDataCache.period == period,
                StockFinancialDataCache.lang == lang,
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            now = datetime.utcnow()

            if row is None:
                session.add(
                    StockFinancialDataCache(
                        symbol=symbol,
                        data_type=data_type,
                        period=period,
                        lang=lang,
                        data=payload,
                        updated_at=now,
                    )
                )
            else:
                row.data = payload
                row.updated_at = now

            await session.commit()

    def _normalize_symbol(self, symbol: str) -> str:
        return str(symbol or "").strip().upper()[:3]

    def _normalize_data_type(self, data_type: str) -> str:
        normalized = str(data_type or "").strip().lower()
        if normalized not in self._fetchers:
            raise ValueError(f"Unsupported finance data_type: {data_type}")
        return normalized

    def _normalize_period(self, period: str) -> str:
        normalized = str(period or "").strip().lower()
        if normalized not in self.SUPPORTED_PERIODS:
            raise ValueError(f"Unsupported finance period '{period}'. Supported: quarter, year")
        return normalized

    def _normalize_lang(self, lang: str) -> str:
        normalized = str(lang or "en").strip().lower()
        return normalized or "en"

    def _current_quarter_start(self, reference: datetime | None = None) -> datetime:
        now = reference or datetime.utcnow()
        quarter_month = ((now.month - 1) // 3) * 3 + 1
        return datetime(now.year, quarter_month, 1)

    def _is_stale(self, updated_at: datetime | None) -> bool:
        if updated_at is None:
            return True
        return updated_at < self._current_quarter_start()

    def _serialize_records_for_cache(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(records, list):
            return []

        normalized_records: List[Dict[str, Any]] = []
        for row in records:
            if not isinstance(row, dict):
                continue
            normalized_records.append(
                {str(key): self._to_json_safe_value(value) for key, value in row.items()}
            )
        return normalized_records

    def _deserialize_cached_records(self, payload: Any) -> List[Dict[str, Any]]:
        if not isinstance(payload, list):
            return []

        normalized_records: List[Dict[str, Any]] = []
        for row in payload:
            if isinstance(row, dict):
                normalized_records.append(dict(row))
        return normalized_records

    def _to_json_safe_value(self, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, dict):
            return {str(k): self._to_json_safe_value(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._to_json_safe_value(item) for item in value]

        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                return None
            return value.isoformat()

        if isinstance(value, (datetime, date)):
            return value.isoformat()

        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                value = value.item()
            except Exception:
                pass

        try:
            if pd.api.types.is_scalar(value) and pd.isna(value):
                return None
        except Exception:
            pass

        return value

    def _normalize_finance_dataframe(self, df: pd.DataFrame | None) -> List[Dict[str, Any]]:
        if df is None or df.empty:
            return []

        df = _flatten_columns(df)
        records = df.to_dict('records')
        return self._serialize_records_for_cache(records)

    def _fetch_income_statement_sync(self, symbol: str, period: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch income statement synchronously from source API."""
        from vnstock import Vnstock

        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch income statement for {symbol}")

        try:
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            df = s.finance.income_statement(period=period, lang=lang)
            api_circuit_breaker.record_success()
            return self._normalize_finance_dataframe(df)
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching income statement for {symbol}: {e}")
            logger.warning(f"Error fetching income statement for {symbol}: {e}")
            raise

    def _fetch_balance_sheet_sync(self, symbol: str, period: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch balance sheet synchronously from source API."""
        from vnstock import Vnstock

        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch balance sheet for {symbol}")

        try:
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            df = s.finance.balance_sheet(period=period, lang=lang)
            api_circuit_breaker.record_success()
            return self._normalize_finance_dataframe(df)
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching balance sheet for {symbol}: {e}")
            logger.warning(f"Error fetching balance sheet for {symbol}: {e}")
            raise

    def _fetch_cash_flow_sync(self, symbol: str, period: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch cash flow synchronously from source API."""
        from vnstock import Vnstock

        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch cash flow for {symbol}")

        try:
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            df = s.finance.cash_flow(period=period, lang=lang)
            api_circuit_breaker.record_success()
            return self._normalize_finance_dataframe(df)
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching cash flow for {symbol}: {e}")
            logger.warning(f"Error fetching cash flow for {symbol}: {e}")
            raise

    def _fetch_financial_ratios_sync(self, symbol: str, period: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch financial ratios synchronously from source API."""
        from vnstock import Vnstock

        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch financial ratios for {symbol}")

        try:
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            df = s.finance.ratio(period=period, lang=lang)
            api_circuit_breaker.record_success()
            return self._normalize_finance_dataframe(df)
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching financial ratios for {symbol}: {e}")
            logger.warning(f"Error fetching financial ratios for {symbol}: {e}")
            raise

    def extract_latest_pe_ratio(self, ratio_records: List[Dict[str, Any]]) -> float | None:
        """Extract latest P/E ratio from cached ratio records."""
        if not ratio_records:
            return None

        first_row = ratio_records[0]
        if not isinstance(first_row, dict):
            return None

        for key, value in first_row.items():
            key_upper = str(key).upper().replace(" ", "")
            if "P/E" not in key_upper and key_upper not in {"PE", "P_E"} and not key_upper.endswith("_PE"):
                continue

            try:
                if value is None:
                    continue
                parsed = float(value)
                if pd.isna(parsed):
                    continue
                return parsed
            except (TypeError, ValueError):
                continue

        return None
