"""
Shared utilities and infrastructure for vnstock service.
"""
from __future__ import annotations

from typing import List, Dict, Callable, TypeVar, Any
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import time
import pandas as pd

from app.core.circuit_breaker import api_circuit_breaker, CircuitOpenError
from app.core.config import settings
from app.core.logging_config import (
    get_main_logger,
    get_background_logger
)
from app.services.sync_status import sync_status

# Initialize loggers
logger = get_main_logger()
bg_logger = get_background_logger()

# Determine worker counts based on CPU cores
_cpu_count = os.cpu_count() or 4

# Frontend executor: handles user-facing API calls
# Sized larger to handle concurrent user requests with good responsiveness
frontend_executor = ThreadPoolExecutor(
    max_workers=max(8, _cpu_count * 2),
    thread_name_prefix="frontend"
)

# Background executor: handles sync operations
# Sized smaller to avoid overwhelming the vnstock API with concurrent requests
background_executor = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="bg_sync"
)

# Thread-local storage for vnstock Fund API instances
_fund_api_local = threading.local()

# Cached synchronous SQLAlchemy engine
_sync_engine = None
_sync_engine_lock = threading.Lock()


def _ensure_pandas_applymap() -> None:
    """Provide DataFrame.applymap for pandas versions where it was removed."""
    if hasattr(pd.DataFrame, "applymap"):
        return
    def _applymap(self, func):
        return self.apply(lambda col: col.map(func))
    pd.DataFrame.applymap = _applymap  # type: ignore[attr-defined]


class RateLimitError(Exception):
    """Custom exception raised when API rate limit is hit to prevent SystemExit."""
    pass


# Rate limit detection keywords
RATE_LIMIT_KEYWORDS = [
    "Rate limit", "rate limit", "429",
    "quá nhiều request", "GIỚI HẠN API"
]

T = TypeVar('T')


def _is_rate_limit_error(error: BaseException) -> bool:
    """Check if an exception indicates a rate limit error."""
    if isinstance(error, (RateLimitError, CircuitOpenError)):
        return True
    if isinstance(error, SystemExit):
        return True
    error_msg = str(error)
    return any(keyword in error_msg for keyword in RATE_LIMIT_KEYWORDS)


def _record_rate_limit(reset_seconds: float = 30.0) -> None:
    """Record a rate limit event across circuit breaker and sync status."""
    api_circuit_breaker.record_failure(reset_timeout=reset_seconds)
    sync_status.set_rate_limited(reset_in_seconds=reset_seconds)


def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 2,
    rate_limit_keywords: List[str] | None = None,
) -> T:
    """
    Execute a function with retry logic. ALWAYS fails fast on rate limit.

    This is the synchronous version for use in executor threads.
    It NEVER uses time.sleep() to avoid blocking executor threads.
    """
    if rate_limit_keywords is None:
        rate_limit_keywords = RATE_LIMIT_KEYWORDS

    # Circuit breaker check BEFORE attempting - fail fast
    if not api_circuit_breaker.can_proceed():
        raise CircuitOpenError("Circuit breaker is open - API rate limited")

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            result = func()
            api_circuit_breaker.record_success()
            return result
        except (SystemExit, Exception) as e:
            last_exception = e
            is_rate_limit = _is_rate_limit_error(e)

            if is_rate_limit:
                # Record failure in circuit breaker
                _record_rate_limit(reset_seconds=30.0)

                # Convert SystemExit to RateLimitError to prevent process termination
                if isinstance(e, SystemExit):
                    logger.warning("Caught SystemExit (Rate Limit) - converting to RateLimitError")
                    last_exception = RateLimitError(str(e))

                # ALWAYS fail fast on rate limit - no blocking sleep
                logger.warning(f"Rate limit hit - failing fast (attempt {attempt + 1})")
                raise last_exception
            else:
                # Non-rate-limit error - may retry
                if attempt < max_retries:
                    logger.warning(f"Error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    continue
                raise

    raise last_exception


async def async_retry_with_backoff(
    func: Callable[..., T],
    executor: ThreadPoolExecutor | None = None,
    max_retries: int = 3,
    initial_delay: float = 30.0,
    backoff_multiplier: float = 2.0,
    rate_limit_keywords: List[str] | None = None
) -> T:
    """
    Async retry with non-blocking delays using asyncio.sleep.

    This should be used for background sync tasks. The func is run in
    an executor, and delays use asyncio.sleep (non-blocking).
    """
    if rate_limit_keywords is None:
        rate_limit_keywords = RATE_LIMIT_KEYWORDS
    if executor is None:
        executor = background_executor

    loop = asyncio.get_event_loop()
    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        # Check circuit breaker before each attempt
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError("Circuit breaker is open - API rate limited")

        try:
            result = await loop.run_in_executor(executor, func)
            api_circuit_breaker.record_success()
            return result
        except (SystemExit, RateLimitError, CircuitOpenError, Exception) as e:
            last_exception = e
            is_rate_limit = _is_rate_limit_error(e)

            if is_rate_limit:
                _record_rate_limit(reset_seconds=delay)

                # Convert SystemExit
                if isinstance(e, SystemExit):
                    last_exception = RateLimitError(str(e))

                if attempt < max_retries:
                    bg_logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Waiting {delay:.1f}s (non-blocking)..."
                    )
                    # Non-blocking sleep - event loop can handle other tasks
                    await asyncio.sleep(delay)
                    delay *= backoff_multiplier
                else:
                    raise last_exception
            else:
                # Non-rate-limit error
                raise

    raise last_exception


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-level column names."""
    if df.columns.nlevels > 1:
        new_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                # Join non-NaN parts with underscore
                new_cols.append('_'.join([str(c) for c in col if pd.notna(c)]))
            else:
                new_cols.append(str(col))
        df.columns = new_cols
    return df


def get_sync_engine():
    """Get or create cached synchronous SQLAlchemy engine."""
    global _sync_engine
    if _sync_engine is None:
        with _sync_engine_lock:
            if _sync_engine is None:
                from sqlalchemy import create_engine
                sync_url = settings.database_url.replace('+asyncpg', '')
                _sync_engine = create_engine(sync_url)
    return _sync_engine


def get_thread_local_fund_api():
    """Get or create a thread-local vnstock Fund API instance."""
    if not api_circuit_breaker.can_proceed():
        raise CircuitOpenError("Circuit breaker open - cannot initialize fund API")

    fund_api = getattr(_fund_api_local, "fund_api", None)
    if fund_api is not None:
        return fund_api

    try:
        from vnstock import Fund
        fund_api = Fund()
        _fund_api_local.fund_api = fund_api
        return fund_api
    except (SystemExit, Exception) as e:
        if _is_rate_limit_error(e):
            _record_rate_limit(reset_seconds=60.0)
            raise CircuitOpenError("Rate limited while initializing fund API")
        raise
