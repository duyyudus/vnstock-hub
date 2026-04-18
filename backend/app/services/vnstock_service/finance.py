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
    DEFAULT_PERIOD = "quarter"

    def __init__(self) -> None:
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._refresh_locks_guard = asyncio.Lock()
        self._fetchers: Dict[str, Callable[[str, str], List[Dict[str, Any]]]] = {
            self.DATA_TYPE_INCOME: self._fetch_income_statement_sync,
            self.DATA_TYPE_BALANCE_SHEET: self._fetch_balance_sheet_sync,
            self.DATA_TYPE_CASHFLOW: self._fetch_cash_flow_sync,
            self.DATA_TYPE_RATIOS: self._fetch_financial_ratios_sync,
        }

    async def get_income_statement(self, symbol: str, lang: str = 'en') -> List[Dict[str, Any]]:
        """Fetch income statement data for a given stock symbol."""
        return await self._get_financial_dataset(
            symbol=symbol,
            data_type=self.DATA_TYPE_INCOME,
            lang=lang,
        )

    async def get_balance_sheet(self, symbol: str, lang: str = 'en') -> List[Dict[str, Any]]:
        """Fetch balance sheet data for a given stock symbol."""
        return await self._get_financial_dataset(
            symbol=symbol,
            data_type=self.DATA_TYPE_BALANCE_SHEET,
            lang=lang,
        )

    async def get_cash_flow(self, symbol: str, lang: str = 'en') -> List[Dict[str, Any]]:
        """Fetch cash flow statement data for a given stock symbol."""
        return await self._get_financial_dataset(
            symbol=symbol,
            data_type=self.DATA_TYPE_CASHFLOW,
            lang=lang,
        )

    async def get_financial_ratios(self, symbol: str, lang: str = 'en') -> List[Dict[str, Any]]:
        """Fetch financial ratios for a given stock symbol."""
        return await self._get_financial_dataset(
            symbol=symbol,
            data_type=self.DATA_TYPE_RATIOS,
            lang=lang,
        )

    async def refresh_financial_dataset(
        self,
        symbol: str,
        data_type: str,
        lang: str = 'en',
        raise_on_failure: bool = False,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """Refresh a financial dataset from source and upsert cache, with stale fallback."""
        symbol_key = self._normalize_symbol(symbol)
        data_type_key = self._normalize_data_type(data_type)
        lang_key = self._normalize_lang(lang)

        lock = await self._get_refresh_lock(symbol_key, data_type_key, lang_key)
        async with lock:
            cached = await self._get_cache_entry(
                symbol=symbol_key,
                data_type=data_type_key,
                lang=lang_key,
            )
            if cached and not force_refresh and not self._is_stale(cached.updated_at):
                return self._deserialize_cached_records(cached.data)

            try:
                records = await self._fetch_from_source(
                    symbol=symbol_key,
                    data_type=data_type_key,
                    lang=lang_key,
                )
                await self._upsert_cache_entry(
                    symbol=symbol_key,
                    data_type=data_type_key,
                    lang=lang_key,
                    data=records,
                )
                return records
            except Exception as e:
                logger.warning(
                    f"Finance refresh failed for {symbol_key} ({data_type_key}, {lang_key}): {e}"
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
        lang: str,
    ) -> List[Dict[str, Any]]:
        """DB-first financial data retrieval with quarterly stale refresh."""
        symbol_key = self._normalize_symbol(symbol)
        data_type_key = self._normalize_data_type(data_type)
        lang_key = self._normalize_lang(lang)

        cached = await self._get_cache_entry(
            symbol=symbol_key,
            data_type=data_type_key,
            lang=lang_key,
        )

        if cached and not self._is_stale(cached.updated_at):
            return self._deserialize_cached_records(cached.data)

        return await self.refresh_financial_dataset(
            symbol=symbol_key,
            data_type=data_type_key,
            lang=lang_key,
            raise_on_failure=False,
        )

    async def _fetch_from_source(
        self,
        symbol: str,
        data_type: str,
        lang: str,
    ) -> List[Dict[str, Any]]:
        fetcher = self._fetchers.get(data_type)
        if fetcher is None:
            raise ValueError(f"Unsupported finance data_type: {data_type}")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(frontend_executor, fetcher, symbol, lang)

    async def _get_refresh_lock(self, symbol: str, data_type: str, lang: str) -> asyncio.Lock:
        lock_key = f"{symbol}:{data_type}:{lang}"
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
        lang: str,
    ) -> StockFinancialDataCache | None:
        async with async_session() as session:
            stmt = select(StockFinancialDataCache).where(
                StockFinancialDataCache.symbol == symbol,
                StockFinancialDataCache.data_type == data_type,
                StockFinancialDataCache.period == self.DEFAULT_PERIOD,
                StockFinancialDataCache.lang == lang,
            )
            return (await session.execute(stmt)).scalar_one_or_none()

    async def _upsert_cache_entry(
        self,
        symbol: str,
        data_type: str,
        lang: str,
        data: List[Dict[str, Any]],
    ) -> None:
        payload = self._serialize_records_for_cache(data)

        async with async_session() as session:
            stmt = select(StockFinancialDataCache).where(
                StockFinancialDataCache.symbol == symbol,
                StockFinancialDataCache.data_type == data_type,
                StockFinancialDataCache.period == self.DEFAULT_PERIOD,
                StockFinancialDataCache.lang == lang,
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            now = datetime.utcnow()

            if row is None:
                session.add(
                    StockFinancialDataCache(
                        symbol=symbol,
                        data_type=data_type,
                        period=self.DEFAULT_PERIOD,
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

    def _normalize_lang(self, lang: str) -> str:
        normalized = str(lang or "en").strip().lower()
        return normalized or "en"

    def _current_quarter_start(self, reference: datetime | None = None) -> datetime:
        now = reference or datetime.utcnow()
        quarter_month = ((now.month - 1) // 3) * 3 + 1
        return datetime(now.year, quarter_month, 1)

    def _last_completed_quarter_start(self, reference: datetime | None = None) -> datetime:
        current_quarter_start = self._current_quarter_start(reference)
        if current_quarter_start.month == 1:
            return datetime(current_quarter_start.year - 1, 10, 1)
        return datetime(current_quarter_start.year, current_quarter_start.month - 3, 1)

    def _is_stale(self, updated_at: datetime | None) -> bool:
        if updated_at is None:
            return True
        return updated_at < self._last_completed_quarter_start()

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

    def _build_vnstock_data_alt_client(self, symbol: str):
        from app.lib.vnstock_data_alt.api.financial import Finance

        return Finance(
            source="VCI",
            symbol=symbol[:3],
            period=self.DEFAULT_PERIOD,
            show_log=False,
        )

    def _sort_frame_by_metadata(
        self,
        df: pd.DataFrame,
        columns: List[str],
        ascending: bool,
    ) -> pd.DataFrame:
        sort_columns = [column for column in columns if column in df.columns]
        if not sort_columns:
            return df.copy()
        return df.sort_values(
            by=sort_columns,
            ascending=[ascending] * len(sort_columns),
            kind="mergesort",
            na_position="last",
        )

    def _normalize_statement_dataset(
        self,
        final_df: pd.DataFrame | None,
        raw_df: pd.DataFrame | None,
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        if final_df is None or final_df.empty:
            return []
        if raw_df is None or raw_df.empty:
            return []

        final_flat = _flatten_columns(final_df).copy()
        raw_flat = _flatten_columns(raw_df).copy()

        if len(final_flat) != len(raw_flat):
            raise ValueError(
                f"{dataset_name} raw/final row count mismatch: final={len(final_flat)} raw={len(raw_flat)}"
            )

        final_flat["__row_position"] = range(len(final_flat))
        raw_flat["__row_position"] = range(len(raw_flat))

        raw_ordered = self._sort_frame_by_metadata(
            raw_flat,
            columns=[
                "yearReport",
                "lengthReport",
                "updateDate",
                "publicDate",
                "__row_position",
            ],
            ascending=True,
        ).reset_index(drop=True)

        final_ordered = (
            final_flat.iloc[raw_ordered["__row_position"].tolist()]
            .reset_index(drop=True)
            .copy()
        )

        metadata = pd.DataFrame(
            {
                "ticker": raw_ordered.get("ticker"),
                "yearReport": raw_ordered.get("yearReport"),
                "lengthReport": raw_ordered.get("lengthReport"),
                "__updateDate": raw_ordered.get("updateDate"),
                "__publicDate": raw_ordered.get("publicDate"),
                "__row_position": raw_ordered.get("__row_position"),
            }
        )

        value_columns = final_ordered.drop(
            columns=[
                column
                for column in [
                    "ticker",
                    "yearReport",
                    "lengthReport",
                    "organCode",
                    "createDate",
                    "updateDate",
                    "publicDate",
                    "report_period",
                    "__row_position",
                ]
                if column in final_ordered.columns
            ],
            errors="ignore",
        )

        combined = pd.concat([metadata, value_columns], axis=1)
        combined = self._sort_frame_by_metadata(
            combined,
            columns=[
                "yearReport",
                "lengthReport",
                "__updateDate",
                "__publicDate",
                "__row_position",
            ],
            ascending=False,
        )
        combined = combined.drop(
            columns=["__updateDate", "__publicDate", "__row_position"],
            errors="ignore",
        ).reset_index(drop=True)

        return self._normalize_finance_dataframe(combined)

    def _normalize_ratio_dataset(self, raw_df: pd.DataFrame | None) -> List[Dict[str, Any]]:
        if raw_df is None or raw_df.empty:
            return []

        ratio_df = _flatten_columns(raw_df).copy()
        ratio_df["__row_position"] = range(len(ratio_df))

        if "Ratio Type" not in ratio_df.columns:
            raise ValueError("ratio payload missing 'Ratio Type'")
        if "yearReport" not in ratio_df.columns:
            raise ValueError("ratio payload missing 'yearReport'")
        if "Ratio TTM Id" not in ratio_df.columns:
            raise ValueError("ratio payload missing 'Ratio TTM Id'")

        ratio_df = ratio_df[ratio_df["Ratio Type"] == "RATIO_TTM"].copy()
        if ratio_df.empty:
            return []

        ratio_df = self._sort_frame_by_metadata(
            ratio_df,
            columns=["yearReport", "Ratio TTM Id", "__row_position"],
            ascending=True,
        ).reset_index(drop=True)

        ratio_df["Meta_lengthReport"] = (
            ratio_df.groupby("yearReport").cumcount() + 1
        )
        ratio_df["Meta_ticker"] = ratio_df.get("organCode")
        if "ticker" in ratio_df.columns:
            ratio_df["Meta_ticker"] = ratio_df["Meta_ticker"].fillna(ratio_df["ticker"])
        ratio_df["Meta_yearReport"] = ratio_df["yearReport"]
        ratio_df["Meta_period"] = ratio_df.apply(
            lambda row: f"{int(row['Meta_yearReport'])}-Q{int(row['Meta_lengthReport'])}",
            axis=1,
        )

        ratio_df = self._sort_frame_by_metadata(
            ratio_df,
            columns=["Meta_yearReport", "Meta_lengthReport", "Ratio TTM Id", "__row_position"],
            ascending=False,
        ).reset_index(drop=True)

        ratio_df = ratio_df.drop(
            columns=[
                column
                for column in [
                    "Ratio TTM Id",
                    "Ratio Type",
                    "Ratio Year Id",
                    "organCode",
                    "ticker",
                    "yearReport",
                    "__row_position",
                ]
                if column in ratio_df.columns
            ],
            errors="ignore",
        )

        ordered_columns = [
            column
            for column in [
                "Meta_ticker",
                "Meta_yearReport",
                "Meta_lengthReport",
                "Meta_period",
            ]
            if column in ratio_df.columns
        ]
        ordered_columns.extend(
            [column for column in ratio_df.columns if column not in ordered_columns]
        )
        ratio_df = ratio_df[ordered_columns]

        return self._normalize_finance_dataframe(ratio_df)

    def _fetch_statement_dataset_sync(
        self,
        symbol: str,
        lang: str,
        method_name: str,
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch {dataset_name} for {symbol}")

        try:
            client = self._build_vnstock_data_alt_client(symbol)
            fetcher = getattr(client, method_name)
            final_df = fetcher(period=self.DEFAULT_PERIOD, lang=lang, mode="final")
            raw_df = fetcher(period=self.DEFAULT_PERIOD, lang=lang, mode="raw")
            records = self._normalize_statement_dataset(final_df, raw_df, dataset_name=dataset_name)
            api_circuit_breaker.record_success()
            return records
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching {dataset_name} for {symbol}: {e}")
            logger.warning(f"Error fetching {dataset_name} for {symbol}: {e}")
            raise

    def _fetch_income_statement_sync(self, symbol: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch income statement synchronously from source API."""
        return self._fetch_statement_dataset_sync(
            symbol=symbol,
            lang=lang,
            method_name="income_statement",
            dataset_name="income statement",
        )

    def _fetch_balance_sheet_sync(self, symbol: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch balance sheet synchronously from source API."""
        return self._fetch_statement_dataset_sync(
            symbol=symbol,
            lang=lang,
            method_name="balance_sheet",
            dataset_name="balance sheet",
        )

    def _fetch_cash_flow_sync(self, symbol: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch cash flow synchronously from source API."""
        return self._fetch_statement_dataset_sync(
            symbol=symbol,
            lang=lang,
            method_name="cash_flow",
            dataset_name="cash flow",
        )

    def _fetch_financial_ratios_sync(self, symbol: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch financial ratios synchronously from source API."""
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch financial ratios for {symbol}")

        try:
            client = self._build_vnstock_data_alt_client(symbol)
            df = client.ratio(period=self.DEFAULT_PERIOD, lang=lang, mode="raw")
            records = self._normalize_ratio_dataset(df)
            api_circuit_breaker.record_success()
            return records
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
