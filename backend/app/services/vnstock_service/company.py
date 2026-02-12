from __future__ import annotations

from typing import Any, Callable, Dict, List
from datetime import date, datetime
import asyncio
import pandas as pd

from sqlalchemy import select

from app.db.database import async_session
from app.db.models import StockCompanyDataCache

from .core import (
    frontend_executor,
    logger,
    api_circuit_breaker,
    CircuitOpenError,
    _record_rate_limit,
    _is_rate_limit_error,
    _flatten_columns,
    _ensure_pandas_applymap,
)


class CompanyService:
    """Company information endpoints with DB-first caching."""

    DATA_TYPE_OVERVIEW = "overview"
    DATA_TYPE_SHAREHOLDERS = "shareholders"
    DATA_TYPE_OFFICERS = "officers"
    DATA_TYPE_SUBSIDIARIES = "subsidiaries"

    def __init__(self) -> None:
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._refresh_locks_guard = asyncio.Lock()
        self._fetchers: Dict[str, Callable[[str], List[Dict[str, Any]]]] = {
            self.DATA_TYPE_OVERVIEW: self._fetch_company_overview_sync,
            self.DATA_TYPE_SHAREHOLDERS: self._fetch_shareholders_sync,
            self.DATA_TYPE_OFFICERS: self._fetch_officers_sync,
            self.DATA_TYPE_SUBSIDIARIES: self._fetch_subsidiaries_sync,
        }

    async def get_company_overview(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch company overview for a given stock symbol."""
        return await self._get_company_dataset(
            symbol=symbol,
            data_type=self.DATA_TYPE_OVERVIEW,
        )

    async def get_shareholders(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch shareholders for a given stock symbol."""
        return await self._get_company_dataset(
            symbol=symbol,
            data_type=self.DATA_TYPE_SHAREHOLDERS,
        )

    async def get_officers(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch officers for a given stock symbol."""
        return await self._get_company_dataset(
            symbol=symbol,
            data_type=self.DATA_TYPE_OFFICERS,
        )

    async def get_subsidiaries(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch subsidiaries for a given stock symbol."""
        return await self._get_company_dataset(
            symbol=symbol,
            data_type=self.DATA_TYPE_SUBSIDIARIES,
        )

    async def refresh_company_dataset(
        self,
        symbol: str,
        data_type: str,
        raise_on_failure: bool = False,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """Refresh a company dataset from source and upsert cache."""
        symbol_key = self._normalize_symbol(symbol)
        data_type_key = self._normalize_data_type(data_type)

        lock = await self._get_refresh_lock(symbol_key, data_type_key)
        async with lock:
            cached = await self._get_cache_entry(
                symbol=symbol_key,
                data_type=data_type_key,
            )
            if cached and not force_refresh:
                return self._deserialize_cached_records(cached.data)

            try:
                records = await self._fetch_from_source(
                    symbol=symbol_key,
                    data_type=data_type_key,
                )
                await self._upsert_cache_entry(
                    symbol=symbol_key,
                    data_type=data_type_key,
                    data=records,
                )
                return records
            except Exception as e:
                logger.warning(
                    f"Company refresh failed for {symbol_key} ({data_type_key}): {e}"
                )
                if raise_on_failure:
                    raise
                if cached:
                    return self._deserialize_cached_records(cached.data)
                return []

    async def _get_company_dataset(
        self,
        symbol: str,
        data_type: str,
    ) -> List[Dict[str, Any]]:
        symbol_key = self._normalize_symbol(symbol)
        data_type_key = self._normalize_data_type(data_type)

        cached = await self._get_cache_entry(
            symbol=symbol_key,
            data_type=data_type_key,
        )
        if cached:
            return self._deserialize_cached_records(cached.data)

        return await self.refresh_company_dataset(
            symbol=symbol_key,
            data_type=data_type_key,
            raise_on_failure=False,
        )

    async def _fetch_from_source(
        self,
        symbol: str,
        data_type: str,
    ) -> List[Dict[str, Any]]:
        fetcher = self._fetchers.get(data_type)
        if fetcher is None:
            raise ValueError(f"Unsupported company data_type: {data_type}")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(frontend_executor, fetcher, symbol)

    async def _get_refresh_lock(self, symbol: str, data_type: str) -> asyncio.Lock:
        lock_key = f"{symbol}:{data_type}"
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
    ) -> StockCompanyDataCache | None:
        async with async_session() as session:
            stmt = select(StockCompanyDataCache).where(
                StockCompanyDataCache.symbol == symbol,
                StockCompanyDataCache.data_type == data_type,
            )
            return (await session.execute(stmt)).scalar_one_or_none()

    async def _upsert_cache_entry(
        self,
        symbol: str,
        data_type: str,
        data: List[Dict[str, Any]],
    ) -> None:
        payload = self._serialize_records_for_cache(data)

        async with async_session() as session:
            stmt = select(StockCompanyDataCache).where(
                StockCompanyDataCache.symbol == symbol,
                StockCompanyDataCache.data_type == data_type,
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            now = datetime.utcnow()

            if row is None:
                session.add(
                    StockCompanyDataCache(
                        symbol=symbol,
                        data_type=data_type,
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
            raise ValueError(f"Unsupported company data_type: {data_type}")
        return normalized

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

    def _normalize_company_payload(self, payload: Any) -> List[Dict[str, Any]]:
        if payload is None:
            return []

        if isinstance(payload, dict):
            payload = pd.DataFrame([payload])
        elif isinstance(payload, list):
            payload = pd.DataFrame(payload)

        if not isinstance(payload, pd.DataFrame) or payload.empty:
            return []

        payload = _flatten_columns(payload)
        records = payload.to_dict('records')
        return self._serialize_records_for_cache(records)

    def _build_company_client(self, symbol: str):
        from vnstock import Company

        _ensure_pandas_applymap()
        return Company(symbol=symbol[:3], source='VCI')

    def _fetch_company_overview_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch company overview synchronously from source API."""
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch company overview for {symbol}")

        try:
            company = self._build_company_client(symbol)
            payload = company.overview()
            api_circuit_breaker.record_success()
            return self._normalize_company_payload(payload)
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching company overview for {symbol}: {e}")
            logger.warning(f"Error fetching company overview for {symbol}: {e}")
            raise

    def _fetch_shareholders_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch shareholders synchronously from source API."""
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch shareholders for {symbol}")

        try:
            company = self._build_company_client(symbol)
            payload = company.shareholders()
            api_circuit_breaker.record_success()
            return self._normalize_company_payload(payload)
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching shareholders for {symbol}: {e}")
            logger.warning(f"Error fetching shareholders for {symbol}: {e}")
            raise

    def _fetch_officers_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch officers synchronously from source API."""
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch officers for {symbol}")

        try:
            company = self._build_company_client(symbol)
            payload = company.officers()
            api_circuit_breaker.record_success()
            return self._normalize_company_payload(payload)
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching officers for {symbol}: {e}")
            logger.warning(f"Error fetching officers for {symbol}: {e}")
            raise

    def _is_known_subsidiaries_no_data_error(self, error: BaseException) -> bool:
        # vnstock may wrap a KeyError("['organ_code'] not found in axis") in tenacity.RetryError.
        # Treat this as an upstream no-data shape mismatch for subsidiaries only.
        error_candidates: List[BaseException] = [error]
        try:
            from tenacity import RetryError
        except Exception:
            RetryError = None  # type: ignore[assignment]

        if RetryError is not None and isinstance(error, RetryError):
            last_attempt = getattr(error, "last_attempt", None)
            if last_attempt is not None and hasattr(last_attempt, "exception"):
                try:
                    nested_error = last_attempt.exception()
                except Exception:
                    nested_error = None
                if isinstance(nested_error, BaseException):
                    error_candidates.append(nested_error)

        for candidate in error_candidates:
            if isinstance(candidate, KeyError) and "organ_code" in str(candidate).lower():
                return True
        return False

    def _fetch_subsidiaries_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch subsidiaries synchronously from source API."""
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch subsidiaries for {symbol}")

        try:
            company = self._build_company_client(symbol)
            payload = company.subsidiaries()
            api_circuit_breaker.record_success()
            return self._normalize_company_payload(payload)
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching subsidiaries for {symbol}: {e}")
            if self._is_known_subsidiaries_no_data_error(e):
                logger.info(
                    "Subsidiaries data unavailable for %s due to known upstream schema mismatch "
                    "(missing organ_code); caching empty dataset",
                    symbol,
                )
                api_circuit_breaker.record_success()
                return []
            logger.warning(f"Error fetching subsidiaries for {symbol}: {e}")
            raise
