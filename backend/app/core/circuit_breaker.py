"""
Thread-safe circuit breaker for API rate limit management.

The circuit breaker prevents cascading failures by stopping API calls
when rate limits are detected, allowing the system to recover gracefully.

States:
- CLOSED: Normal operation, calls proceed
- OPEN: Rate limited, all calls are rejected immediately
- HALF_OPEN: Testing if rate limit has cleared (allows limited calls)
"""
import threading
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, TypeVar

from app.core.logging_config import get_main_logger

logger = get_main_logger()


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Rate limited, rejecting all calls
    HALF_OPEN = "half_open" # Testing if rate limit cleared


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 2      # Failures before opening circuit
    recovery_timeout: float = 30.0  # Seconds before transitioning to half-open
    half_open_max_calls: int = 1    # Test calls allowed in half-open state


T = TypeVar('T')


class CircuitOpenError(Exception):
    """
    Raised when circuit breaker is open.

    This indicates the system is rate limited and the caller should
    not attempt the operation. The frontend should handle this gracefully
    by showing cached data or an appropriate error message.
    """
    def __init__(self, message: str = "Circuit breaker is open - API rate limited"):
        self.message = message
        super().__init__(self.message)


class CircuitBreaker:
    """
    Thread-safe circuit breaker for API rate limit handling.

    Usage:
        cb = CircuitBreaker()

        if cb.can_proceed():
            try:
                result = api_call()
                cb.record_success()
                return result
            except RateLimitError:
                cb.record_failure(reset_timeout=30.0)
                raise
        else:
            raise CircuitOpenError()

    The circuit breaker automatically transitions between states:
    - CLOSED -> OPEN: When failure_threshold is reached
    - OPEN -> HALF_OPEN: After recovery_timeout seconds
    - HALF_OPEN -> CLOSED: On successful call
    - HALF_OPEN -> OPEN: On failed call
    """

    def __init__(self, config: CircuitBreakerConfig = None, name: str = "default"):
        self.config = config or CircuitBreakerConfig()
        self.name = name
        self._lock = threading.RLock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._success_count = 0

    @property
    def state(self) -> CircuitState:
        """Get current state, potentially transitioning to HALF_OPEN."""
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rate limited)."""
        return self.state == CircuitState.OPEN

    @property
    def time_until_half_open(self) -> Optional[float]:
        """
        Get seconds remaining until circuit transitions to half-open.
        Returns None if not in OPEN state.
        """
        with self._lock:
            if self._state != CircuitState.OPEN or not self._last_failure_time:
                return None
            elapsed = time.time() - self._last_failure_time
            remaining = self.config.recovery_timeout - elapsed
            return max(0.0, remaining)

    def can_proceed(self) -> bool:
        """
        Check if a call can proceed. Thread-safe.

        Returns:
            True if call can proceed, False if circuit is open/limiting
        """
        with self._lock:
            self._maybe_transition_to_half_open()

            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            else:  # OPEN
                return False

    def record_success(self) -> None:
        """
        Record a successful API call. Thread-safe.

        In HALF_OPEN state, this closes the circuit.
        In CLOSED state, this resets the failure count.
        """
        with self._lock:
            self._success_count += 1

            if self._state == CircuitState.HALF_OPEN:
                # Success in half-open closes the circuit
                logger.info(f"Circuit breaker [{self.name}]: HALF_OPEN -> CLOSED (success)")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self, reset_timeout: float = None) -> None:
        """
        Record a failed API call (rate limit hit). Thread-safe.

        Args:
            reset_timeout: Override the recovery timeout based on API response.
                          For example, if API says "retry after 60s", pass 60.0.
        """
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if reset_timeout:
                # Override recovery timeout based on API response
                self.config.recovery_timeout = reset_timeout

            if self._state == CircuitState.HALF_OPEN:
                # Failure in half-open opens the circuit again
                logger.warning(f"Circuit breaker [{self.name}]: HALF_OPEN -> OPEN (failure in half-open)")
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
            elif self._failure_count >= self.config.failure_threshold:
                if self._state != CircuitState.OPEN:
                    logger.warning(
                        f"Circuit breaker [{self.name}]: {self._state.value} -> OPEN "
                        f"(failures={self._failure_count}, threshold={self.config.failure_threshold})"
                    )
                    self._state = CircuitState.OPEN

    def force_open(self, duration: float = None) -> None:
        """
        Force the circuit open immediately.

        Args:
            duration: How long to keep circuit open (seconds).
                     If provided, overrides recovery_timeout.
        """
        with self._lock:
            if self._state != CircuitState.OPEN:
                logger.warning(f"Circuit breaker [{self.name}]: Forced OPEN for {duration or self.config.recovery_timeout}s")
            self._state = CircuitState.OPEN
            self._last_failure_time = time.time()
            if duration:
                self.config.recovery_timeout = duration

    def reset(self) -> None:
        """Reset circuit breaker to initial closed state."""
        with self._lock:
            logger.info(f"Circuit breaker [{self.name}]: Reset to CLOSED")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0

    def _maybe_transition_to_half_open(self) -> None:
        """
        Check if we should transition from OPEN to HALF_OPEN.
        Called internally when checking state.
        """
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.config.recovery_timeout:
                logger.info(
                    f"Circuit breaker [{self.name}]: OPEN -> HALF_OPEN "
                    f"(elapsed={elapsed:.1f}s >= timeout={self.config.recovery_timeout}s)"
                )
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0

    def get_stats(self) -> dict:
        """Get circuit breaker statistics for monitoring."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "time_until_half_open": self.time_until_half_open
            }


# Global circuit breaker instance for vnstock API
api_circuit_breaker = CircuitBreaker(
    CircuitBreakerConfig(
        failure_threshold=2,      # Open after 2 consecutive failures
        recovery_timeout=30.0,    # Wait 30 seconds before testing again
        half_open_max_calls=1     # Allow 1 test call in half-open
    ),
    name="vnstock_api"
)
