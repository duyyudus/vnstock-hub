"""
Thread-safe global sync status tracking for background tasks.

All state modifications are protected by RLock to prevent race conditions
when multiple threads/coroutines access the sync status concurrently.
"""
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class SyncStatusData:
    """Status for a specific sync operation (immutable snapshot)."""
    is_syncing: bool = False
    last_sync: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0


class GlobalSyncStatus:
    """
    Thread-safe global sync status tracker for all background sync operations.

    Uses RLock for all state modifications to prevent race conditions.
    Property accessors return immutable copies to prevent external mutations.
    """

    def __init__(self):
        self._lock = threading.RLock()

        # Fund performance sync status
        self._fund_performance_is_syncing = False
        self._fund_performance_last_sync: Optional[str] = None
        self._fund_performance_error: Optional[str] = None
        self._fund_performance_started_at: Optional[str] = None
        self._fund_performance_progress: float = 0.0

        # Rate limit status
        self._is_rate_limited = False
        self._rate_limit_reset_at: Optional[datetime] = None

    @property
    def fund_performance(self) -> SyncStatusData:
        """
        Get fund performance sync status as an immutable snapshot.

        Returns a copy to prevent external modifications.
        """
        with self._lock:
            return SyncStatusData(
                is_syncing=self._fund_performance_is_syncing,
                last_sync=self._fund_performance_last_sync,
                error=self._fund_performance_error,
                started_at=self._fund_performance_started_at,
                progress=self._fund_performance_progress
            )

    @property
    def is_rate_limited(self) -> bool:
        """
        Check if rate limited, auto-clearing if expired. Thread-safe.

        Returns:
            True if currently rate limited, False otherwise.
        """
        with self._lock:
            if self._is_rate_limited and self._rate_limit_reset_at:
                if datetime.now() > self._rate_limit_reset_at:
                    # Rate limit has expired - auto-clear
                    self._is_rate_limited = False
                    self._rate_limit_reset_at = None
                    return False
            return self._is_rate_limited

    @property
    def rate_limit_reset_at(self) -> Optional[str]:
        """
        Get the rate limit reset time as ISO format string. Thread-safe.

        Returns:
            ISO format datetime string or None if not rate limited.
        """
        with self._lock:
            if self._rate_limit_reset_at:
                return self._rate_limit_reset_at.isoformat()
            return None

    @property
    def rate_limit_seconds_remaining(self) -> Optional[float]:
        """
        Get seconds remaining until rate limit expires. Thread-safe.

        Returns:
            Seconds remaining or None if not rate limited.
        """
        with self._lock:
            if self._is_rate_limited and self._rate_limit_reset_at:
                remaining = (self._rate_limit_reset_at - datetime.now()).total_seconds()
                return max(0.0, remaining)
            return None

    def start_fund_performance_sync(self) -> None:
        """Mark fund performance sync as started. Thread-safe."""
        with self._lock:
            self._fund_performance_is_syncing = True
            self._fund_performance_started_at = datetime.now().isoformat()
            self._fund_performance_error = None
            self._fund_performance_progress = 0.0

    def update_fund_performance_progress(self, progress: float) -> None:
        """
        Update fund performance sync progress. Thread-safe.

        Args:
            progress: Progress value between 0.0 and 1.0
        """
        with self._lock:
            self._fund_performance_progress = min(1.0, max(0.0, progress))

    def complete_fund_performance_sync(
        self,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        Mark fund performance sync as completed. Thread-safe.

        Args:
            success: Whether sync completed successfully.
            error: Error message if sync failed.
        """
        with self._lock:
            self._fund_performance_is_syncing = False
            self._fund_performance_last_sync = datetime.now().isoformat()
            self._fund_performance_progress = 1.0 if success else self._fund_performance_progress
            if not success:
                self._fund_performance_error = error

    def set_rate_limited(self, reset_in_seconds: float = 60.0) -> None:
        """
        Mark the system as rate limited with auto-expiry. Thread-safe.

        Args:
            reset_in_seconds: How long until rate limit expires.
        """
        with self._lock:
            self._is_rate_limited = True
            self._rate_limit_reset_at = datetime.now() + timedelta(seconds=reset_in_seconds)

    def clear_rate_limit(self) -> None:
        """Clear the rate limit status. Thread-safe."""
        with self._lock:
            self._is_rate_limited = False
            self._rate_limit_reset_at = None

    def get_status_dict(self) -> dict:
        """
        Get complete status as a dictionary for API response. Thread-safe.

        Returns:
            Dictionary containing all sync status information.
        """
        with self._lock:
            return {
                "fund_performance": {
                    "is_syncing": self._fund_performance_is_syncing,
                    "last_sync": self._fund_performance_last_sync,
                    "error": self._fund_performance_error,
                    "started_at": self._fund_performance_started_at,
                    "progress": self._fund_performance_progress
                },
                "rate_limit": {
                    "is_limited": self._is_rate_limited,
                    "reset_at": self._rate_limit_reset_at.isoformat() if self._rate_limit_reset_at else None,
                    "seconds_remaining": self.rate_limit_seconds_remaining
                }
            }


# Global singleton instance
sync_status = GlobalSyncStatus()
