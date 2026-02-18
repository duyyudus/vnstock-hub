from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from tools.cafef_scraper.types import DownloadResult


class RateLimiter:
    """Simple interval-based rate limiter (requests-per-second)."""

    def __init__(
        self,
        rate_limit_rps: float,
        adaptive: bool = False,
        min_rps: float = 0.5,
        recovery_multiplier: float = 1.05,
        cooldown_seconds: float = 20.0,
        cooldown_trigger_streak: int = 3,
        logger: logging.Logger | None = None,
    ):
        self._base_rps = max(0.1, rate_limit_rps)
        self._current_rps = self._base_rps
        self._adaptive = adaptive
        self._min_rps = min(max(0.1, min_rps), self._base_rps)
        self._recovery_multiplier = max(1.01, recovery_multiplier)
        self._cooldown_seconds = max(0.0, cooldown_seconds)
        self._cooldown_trigger_streak = max(1, cooldown_trigger_streak)
        self._logger = logger
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0
        self._retryable_failure_streak = 0
        self._cooldown_until = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            interval = 1.0 / max(0.1, self._current_rps)
            now = time.monotonic()
            if now < self._cooldown_until:
                await asyncio.sleep(self._cooldown_until - now)
                now = time.monotonic()
            wait_seconds = self._next_allowed - now
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
                now = time.monotonic()
            self._next_allowed = now + interval

    async def report_result(
        self,
        *,
        success: bool,
        status_code: int | None,
        retryable: bool,
    ) -> None:
        if not self._adaptive:
            return
        async with self._lock:
            previous = self._current_rps
            reason = ""
            now = time.monotonic()
            if success:
                # Recover gradually back toward configured baseline.
                self._retryable_failure_streak = 0
                if self._current_rps < self._base_rps:
                    self._current_rps = min(
                        self._base_rps,
                        self._current_rps * self._recovery_multiplier,
                    )
                    reason = "success recovery"
            elif retryable:
                self._retryable_failure_streak += 1
                # Back off aggressively on throttling, moderately on transient server errors.
                factor = 0.55 if status_code == 429 else 0.75
                self._current_rps = max(self._min_rps, self._current_rps * factor)
                reason = f"retryable failure status={status_code or 'request-error'}"
                if (
                    self._cooldown_seconds > 0
                    and self._retryable_failure_streak >= self._cooldown_trigger_streak
                ):
                    self._current_rps = self._min_rps
                    self._cooldown_until = max(self._cooldown_until, now + self._cooldown_seconds)
                    reason = (
                        f"{reason}; cooldown={self._cooldown_seconds:.0f}s "
                        f"streak={self._retryable_failure_streak}"
                    )
            else:
                self._retryable_failure_streak = 0

            if self._logger and previous > 0 and abs(self._current_rps - previous) / previous >= 0.08:
                self._logger.debug(
                    "Adaptive rate limiter adjusted rps %.2f -> %.2f (%s)",
                    previous,
                    self._current_rps,
                    reason or "no-op",
                )

    @property
    def current_rps(self) -> float:
        return self._current_rps


class HttpDownloader:
    def __init__(
        self,
        client: httpx.AsyncClient,
        limiter: RateLimiter,
        max_retries: int = 5,
        timeout_seconds: float = 30.0,
    ):
        self._client = client
        self._limiter = limiter
        self._max_retries = max(1, max_retries)
        self._timeout_seconds = timeout_seconds

    async def fetch_text(
        self,
        url: str,
        stage: str,
        on_failure: Callable[[str, str, int | None, str | None, int], None] | None = None,
        max_retries: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str | None:
        response, error, attempt, _status_code = await self._request_with_retry(
            url=url,
            stage=stage,
            on_failure=on_failure,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )
        if response is None:
            return None
        try:
            return response.text
        except UnicodeDecodeError:
            if on_failure:
                on_failure(url, stage, response.status_code, "DecodeError", attempt)
            return None

    async def download_pdf(
        self,
        url: str,
        target_path: Path,
        stage: str,
        on_failure: Callable[[str, str, int | None, str | None, int], None] | None = None,
    ) -> DownloadResult:
        response, error, attempt, status_code = await self._request_with_retry(
            url=url,
            stage=stage,
            on_failure=on_failure,
        )
        if response is None:
            return DownloadResult(
                success=False,
                status_code=status_code,
                error=error or "Unknown request error",
            )

        content = response.content
        content_type = response.headers.get("content-type", "")
        if not is_pdf_response(content_type=content_type, content=content):
            if on_failure:
                on_failure(
                    url,
                    stage,
                    response.status_code,
                    f"Not PDF content-type={content_type}",
                    attempt,
                )
            return DownloadResult(
                success=False,
                status_code=response.status_code,
                error=f"Response is not PDF (content-type={content_type})",
            )

        sha256 = hashlib.sha256(content).hexdigest()
        size_bytes = len(content)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(target_path.suffix + ".part")
        temp_path.write_bytes(content)
        temp_path.replace(target_path)
        return DownloadResult(
            success=True,
            status_code=response.status_code,
            error=None,
            local_path=target_path,
            sha256=sha256,
            size_bytes=size_bytes,
        )

    async def _request_with_retry(
        self,
        url: str,
        stage: str,
        on_failure: Callable[[str, str, int | None, str | None, int], None] | None,
        max_retries: int | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[httpx.Response | None, str | None, int, int | None]:
        effective_max_retries = max(1, int(max_retries)) if max_retries is not None else self._max_retries
        effective_timeout = (
            max(0.1, float(timeout_seconds))
            if timeout_seconds is not None
            else self._timeout_seconds
        )
        last_error: str | None = None
        for attempt in range(1, effective_max_retries + 1):
            await self._limiter.acquire()
            try:
                response = await self._client.get(
                    url,
                    follow_redirects=True,
                    timeout=effective_timeout,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
                        )
                    },
                )
            except httpx.RequestError as exc:
                last_error = str(exc)
                await self._limiter.report_result(success=False, status_code=None, retryable=True)
                if on_failure:
                    on_failure(url, stage, None, last_error, attempt)
                if attempt < effective_max_retries:
                    await asyncio.sleep(_retry_delay(attempt))
                continue

            if response.status_code == 200:
                await self._limiter.report_result(success=True, status_code=200, retryable=False)
                return response, None, attempt, response.status_code

            # Retry transient responses.
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}"
                await self._limiter.report_result(
                    success=False,
                    status_code=response.status_code,
                    retryable=True,
                )
                if on_failure:
                    on_failure(url, stage, response.status_code, last_error, attempt)
                if attempt < effective_max_retries:
                    await asyncio.sleep(_retry_delay(attempt))
                continue

            # Non-retryable status.
            last_error = f"HTTP {response.status_code}"
            await self._limiter.report_result(
                success=False,
                status_code=response.status_code,
                retryable=False,
            )
            if on_failure:
                on_failure(url, stage, response.status_code, last_error, attempt)
            return None, last_error, attempt, response.status_code

        return None, last_error or "Request failed after retries", effective_max_retries, None


def _retry_delay(attempt: int) -> float:
    base = 0.5 * (2 ** (attempt - 1))
    jitter = random.uniform(0.0, 0.25)
    return min(20.0, base + jitter)


def is_pdf_response(content_type: str | None, content: bytes) -> bool:
    content_type_value = (content_type or "").lower()
    if "application/pdf" in content_type_value:
        return True
    return content.startswith(b"%PDF")
