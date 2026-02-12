from __future__ import annotations

import asyncio
import time

from app.services.sync_status import sync_status

from .core import api_circuit_breaker


class SharedRateLimitPauseController:
    """Coordinates a shared pause window across sync workers."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._pause_deadline_monotonic: float | None = None

    async def wait_if_paused(self) -> None:
        """
        Block until the current pause window has expired.

        Re-checks pause state after waking to handle concurrent extensions.
        """
        while True:
            async with self._lock:
                deadline = self._pause_deadline_monotonic

            if deadline is None:
                return

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                async with self._lock:
                    if self._pause_deadline_monotonic is not None:
                        if self._pause_deadline_monotonic - time.monotonic() <= 0:
                            self._pause_deadline_monotonic = None
                            return
                continue

            await asyncio.sleep(remaining)

    async def register_rate_limit_and_get_wait(self, fixed_wait_seconds: float) -> float:
        """
        Register a new rate-limit event and return wait seconds until resume.

        Wait duration honors whichever cooldown is currently longest:
        - configured fixed wait
        - sync_status reset timer
        - circuit breaker half-open timer
        """
        fixed_wait = max(0.0, float(fixed_wait_seconds))
        status_wait = max(0.0, float(sync_status.rate_limit_seconds_remaining or 0.0))
        circuit_wait = max(0.0, float(api_circuit_breaker.time_until_half_open or 0.0))

        wait_seconds = max(fixed_wait, status_wait, circuit_wait)
        now = time.monotonic()
        candidate_deadline = now + wait_seconds

        async with self._lock:
            if self._pause_deadline_monotonic is None:
                self._pause_deadline_monotonic = candidate_deadline
            else:
                self._pause_deadline_monotonic = max(
                    self._pause_deadline_monotonic,
                    candidate_deadline,
                )
            return max(0.0, self._pause_deadline_monotonic - now)


shared_rate_limit_pause_controller = SharedRateLimitPauseController()

