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
    _ensure_pandas_applymap,
)


class CompanyService:
    """Company information endpoints."""

    async def get_company_overview(self, symbol: str, source: str = "auto") -> List[Dict[str, Any]]:
        """Fetch company overview for a given stock symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(frontend_executor, self._fetch_company_overview_sync, symbol, source)

    def _fetch_company_overview_sync(self, symbol: str, source: str = "auto") -> List[Dict[str, Any]]:
        """Fetch company overview synchronously."""
        from vnstock import Company

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch company overview for {symbol}")

        source_normalized = (source or "auto").strip().lower()
        if source_normalized not in {"auto", "vci", "kbs"}:
            raise ValueError(f"Unsupported company overview source: {source}")

        def _normalize_records(df: Any) -> List[Dict[str, Any]]:
            if df is None:
                return []
            if isinstance(df, dict):
                df = pd.DataFrame([df])
            elif isinstance(df, list):
                df = pd.DataFrame(df)
            if not isinstance(df, pd.DataFrame) or df.empty:
                return []
            df = _flatten_columns(df)
            records = df.to_dict('records')
            for record in records:
                for key, value in record.items():
                    if value is None:
                        continue
                    # Avoid ambiguous truth values for list/dict-like objects
                    if pd.api.types.is_scalar(value) and pd.isna(value):
                        record[key] = None
            return records

        def _fetch_from_source(source_label: str) -> List[Dict[str, Any]]:
            _ensure_pandas_applymap()
            c = Company(symbol=symbol[:3], source=source_label)
            df = c.overview()
            api_circuit_breaker.record_success()
            return _normalize_records(df)

        def _handle_error(source_label: str, err: BaseException) -> None:
            if _is_rate_limit_error(err):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching company overview for {symbol}: {err}")
            # Unwrap tenacity RetryError to reveal the underlying exception
            try:
                from tenacity import RetryError
                if isinstance(err, RetryError) and err.last_attempt:
                    root = err.last_attempt.exception()
                    logger.warning(
                        f"Error fetching company overview for {symbol} from {source_label}: {err}; root={type(root).__name__}: {root}"
                    )
                    return
            except Exception:
                pass
            logger.warning(f"Error fetching company overview for {symbol} from {source_label}: {err}")

        if source_normalized == "vci":
            try:
                return _fetch_from_source('VCI')
            except CircuitOpenError:
                raise
            except (SystemExit, Exception) as e:
                _handle_error("VCI", e)
                return []

        if source_normalized == "kbs":
            try:
                return _fetch_from_source('KBS')
            except CircuitOpenError:
                raise
            except (SystemExit, Exception) as e:
                _handle_error("KBS", e)
                return []

        # auto: try VCI first, fallback to KBS
        try:
            records = _fetch_from_source('VCI')
            if records:
                return records
            logger.warning(f"Empty company overview for {symbol} from VCI, trying KBS.")
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            _handle_error("VCI", e)

        try:
            return _fetch_from_source('KBS')
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            _handle_error("KBS", e)
            return []

    async def get_shareholders(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch shareholders for a given stock symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(frontend_executor, self._fetch_shareholders_sync, symbol)

    def _fetch_shareholders_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch shareholders synchronously."""
        from vnstock import Company

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch shareholders for {symbol}")

        try:
            c = Company(symbol=symbol[:3], source='VCI')
            df = c.shareholders()
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
                raise CircuitOpenError(f"Rate limited fetching shareholders for {symbol}: {e}")
            logger.warning(f"Error fetching shareholders for {symbol}: {e}")
            return []

    async def get_officers(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch officers for a given stock symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(frontend_executor, self._fetch_officers_sync, symbol)

    def _fetch_officers_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch officers synchronously."""
        from vnstock import Company

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch officers for {symbol}")

        try:
            c = Company(symbol=symbol[:3], source='VCI')
            df = c.officers()
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
                raise CircuitOpenError(f"Rate limited fetching officers for {symbol}: {e}")
            logger.warning(f"Error fetching officers for {symbol}: {e}")
            return []

    async def get_subsidiaries(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch subsidiaries for a given stock symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(frontend_executor, self._fetch_subsidiaries_sync, symbol)

    def _fetch_subsidiaries_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch subsidiaries synchronously."""
        from vnstock import Company

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch subsidiaries for {symbol}")

        try:
            c = Company(symbol=symbol[:3], source='VCI')
            df = c.subsidiaries()
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
                raise CircuitOpenError(f"Rate limited fetching subsidiaries for {symbol}: {e}")
            logger.warning(f"Error fetching subsidiaries for {symbol}: {e}")
            return []
