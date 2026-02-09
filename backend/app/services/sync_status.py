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


@dataclass
class PriceBootstrapStatusData:
    """Runtime status snapshot for price bootstrap job."""
    state: str = "idle"  # idle | running | completed | failed
    total_symbols: int = 0
    processed_symbols: int = 0
    success_symbols: int = 0
    failed_symbols: int = 0
    current_symbol: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    progress: float = 0.0


@dataclass
class PriceJobStatusData:
    """Runtime status snapshot for price incremental or repair job."""
    is_running: bool = False
    total_symbols: int = 0
    processed_symbols: int = 0
    current_symbol: Optional[str] = None
    last_run_at: Optional[str] = None
    started_at: Optional[str] = None
    error: Optional[str] = None
    progress: float = 0.0


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

        # Price bootstrap status
        self._price_bootstrap_state = "idle"
        self._price_bootstrap_total_symbols = 0
        self._price_bootstrap_processed_symbols = 0
        self._price_bootstrap_success_symbols = 0
        self._price_bootstrap_failed_symbols = 0
        self._price_bootstrap_current_symbol: Optional[str] = None
        self._price_bootstrap_started_at: Optional[str] = None
        self._price_bootstrap_completed_at: Optional[str] = None
        self._price_bootstrap_error: Optional[str] = None
        self._price_bootstrap_progress: float = 0.0

        # Price incremental status
        self._price_incremental_is_running = False
        self._price_incremental_total_symbols = 0
        self._price_incremental_processed_symbols = 0
        self._price_incremental_current_symbol: Optional[str] = None
        self._price_incremental_last_run_at: Optional[str] = None
        self._price_incremental_started_at: Optional[str] = None
        self._price_incremental_error: Optional[str] = None
        self._price_incremental_progress: float = 0.0

        # Price repair status
        self._price_repair_is_running = False
        self._price_repair_total_symbols = 0
        self._price_repair_processed_symbols = 0
        self._price_repair_current_symbol: Optional[str] = None
        self._price_repair_last_run_at: Optional[str] = None
        self._price_repair_started_at: Optional[str] = None
        self._price_repair_error: Optional[str] = None
        self._price_repair_progress: float = 0.0

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
    def price_bootstrap(self) -> PriceBootstrapStatusData:
        """Get price bootstrap runtime status as immutable snapshot."""
        with self._lock:
            return PriceBootstrapStatusData(
                state=self._price_bootstrap_state,
                total_symbols=self._price_bootstrap_total_symbols,
                processed_symbols=self._price_bootstrap_processed_symbols,
                success_symbols=self._price_bootstrap_success_symbols,
                failed_symbols=self._price_bootstrap_failed_symbols,
                current_symbol=self._price_bootstrap_current_symbol,
                started_at=self._price_bootstrap_started_at,
                completed_at=self._price_bootstrap_completed_at,
                error=self._price_bootstrap_error,
                progress=self._price_bootstrap_progress,
            )

    @property
    def price_incremental(self) -> PriceJobStatusData:
        """Get price incremental runtime status as immutable snapshot."""
        with self._lock:
            return PriceJobStatusData(
                is_running=self._price_incremental_is_running,
                total_symbols=self._price_incremental_total_symbols,
                processed_symbols=self._price_incremental_processed_symbols,
                current_symbol=self._price_incremental_current_symbol,
                last_run_at=self._price_incremental_last_run_at,
                started_at=self._price_incremental_started_at,
                error=self._price_incremental_error,
                progress=self._price_incremental_progress,
            )

    @property
    def price_repair(self) -> PriceJobStatusData:
        """Get price repair runtime status as immutable snapshot."""
        with self._lock:
            return PriceJobStatusData(
                is_running=self._price_repair_is_running,
                total_symbols=self._price_repair_total_symbols,
                processed_symbols=self._price_repair_processed_symbols,
                current_symbol=self._price_repair_current_symbol,
                last_run_at=self._price_repair_last_run_at,
                started_at=self._price_repair_started_at,
                error=self._price_repair_error,
                progress=self._price_repair_progress,
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

    def start_price_bootstrap(self, total_symbols: int) -> None:
        """Mark price bootstrap as running."""
        with self._lock:
            self._price_bootstrap_state = "running"
            self._price_bootstrap_total_symbols = max(0, total_symbols)
            self._price_bootstrap_processed_symbols = 0
            self._price_bootstrap_success_symbols = 0
            self._price_bootstrap_failed_symbols = 0
            self._price_bootstrap_current_symbol = None
            self._price_bootstrap_started_at = datetime.now().isoformat()
            self._price_bootstrap_completed_at = None
            self._price_bootstrap_error = None
            self._price_bootstrap_progress = 0.0

    def update_price_bootstrap_progress(
        self,
        processed_symbols: int,
        success_symbols: int,
        failed_symbols: int,
        current_symbol: Optional[str],
    ) -> None:
        """Update bootstrap progress counters."""
        with self._lock:
            self._price_bootstrap_processed_symbols = max(0, processed_symbols)
            self._price_bootstrap_success_symbols = max(0, success_symbols)
            self._price_bootstrap_failed_symbols = max(0, failed_symbols)
            self._price_bootstrap_current_symbol = current_symbol
            if self._price_bootstrap_total_symbols > 0:
                progress = self._price_bootstrap_processed_symbols / self._price_bootstrap_total_symbols
                self._price_bootstrap_progress = min(1.0, max(0.0, progress))

    def complete_price_bootstrap(self, success: bool, error: Optional[str] = None) -> None:
        """Mark bootstrap as completed/failed."""
        with self._lock:
            self._price_bootstrap_state = "completed" if success else "failed"
            self._price_bootstrap_current_symbol = None
            self._price_bootstrap_completed_at = datetime.now().isoformat()
            self._price_bootstrap_progress = 1.0 if success else self._price_bootstrap_progress
            self._price_bootstrap_error = error

    def start_price_incremental(self, total_symbols: int) -> None:
        """Mark price incremental sync as started."""
        with self._lock:
            self._price_incremental_is_running = True
            self._price_incremental_total_symbols = max(0, total_symbols)
            self._price_incremental_processed_symbols = 0
            self._price_incremental_current_symbol = None
            self._price_incremental_started_at = datetime.now().isoformat()
            self._price_incremental_error = None
            self._price_incremental_progress = 0.0

    def update_price_incremental_progress(self, processed_symbols: int, current_symbol: Optional[str]) -> None:
        """Update incremental sync progress."""
        with self._lock:
            self._price_incremental_processed_symbols = max(0, processed_symbols)
            self._price_incremental_current_symbol = current_symbol
            if self._price_incremental_total_symbols > 0:
                progress = self._price_incremental_processed_symbols / self._price_incremental_total_symbols
                self._price_incremental_progress = min(1.0, max(0.0, progress))

    def complete_price_incremental(self, success: bool, error: Optional[str] = None) -> None:
        """Mark price incremental sync as completed."""
        with self._lock:
            self._price_incremental_is_running = False
            self._price_incremental_current_symbol = None
            self._price_incremental_last_run_at = datetime.now().isoformat()
            self._price_incremental_progress = 1.0 if success else self._price_incremental_progress
            self._price_incremental_error = error

    def start_price_repair(self, total_symbols: int) -> None:
        """Mark price repair sync as started."""
        with self._lock:
            self._price_repair_is_running = True
            self._price_repair_total_symbols = max(0, total_symbols)
            self._price_repair_processed_symbols = 0
            self._price_repair_current_symbol = None
            self._price_repair_started_at = datetime.now().isoformat()
            self._price_repair_error = None
            self._price_repair_progress = 0.0

    def update_price_repair_progress(self, processed_symbols: int, current_symbol: Optional[str]) -> None:
        """Update repair sync progress."""
        with self._lock:
            self._price_repair_processed_symbols = max(0, processed_symbols)
            self._price_repair_current_symbol = current_symbol
            if self._price_repair_total_symbols > 0:
                progress = self._price_repair_processed_symbols / self._price_repair_total_symbols
                self._price_repair_progress = min(1.0, max(0.0, progress))

    def complete_price_repair(self, success: bool, error: Optional[str] = None) -> None:
        """Mark price repair sync as completed."""
        with self._lock:
            self._price_repair_is_running = False
            self._price_repair_current_symbol = None
            self._price_repair_last_run_at = datetime.now().isoformat()
            self._price_repair_progress = 1.0 if success else self._price_repair_progress
            self._price_repair_error = error

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
                "price_sync": {
                    "bootstrap": {
                        "state": self._price_bootstrap_state,
                        "total_symbols": self._price_bootstrap_total_symbols,
                        "processed_symbols": self._price_bootstrap_processed_symbols,
                        "success_symbols": self._price_bootstrap_success_symbols,
                        "failed_symbols": self._price_bootstrap_failed_symbols,
                        "current_symbol": self._price_bootstrap_current_symbol,
                        "started_at": self._price_bootstrap_started_at,
                        "completed_at": self._price_bootstrap_completed_at,
                        "error": self._price_bootstrap_error,
                        "progress": self._price_bootstrap_progress,
                    },
                    "incremental": {
                        "is_running": self._price_incremental_is_running,
                        "total_symbols": self._price_incremental_total_symbols,
                        "processed_symbols": self._price_incremental_processed_symbols,
                        "current_symbol": self._price_incremental_current_symbol,
                        "last_run_at": self._price_incremental_last_run_at,
                        "started_at": self._price_incremental_started_at,
                        "error": self._price_incremental_error,
                        "progress": self._price_incremental_progress,
                    },
                    "repair": {
                        "is_running": self._price_repair_is_running,
                        "total_symbols": self._price_repair_total_symbols,
                        "processed_symbols": self._price_repair_processed_symbols,
                        "current_symbol": self._price_repair_current_symbol,
                        "last_run_at": self._price_repair_last_run_at,
                        "started_at": self._price_repair_started_at,
                        "error": self._price_repair_error,
                        "progress": self._price_repair_progress,
                    },
                },
                "rate_limit": {
                    "is_limited": self._is_rate_limited,
                    "reset_at": self._rate_limit_reset_at.isoformat() if self._rate_limit_reset_at else None,
                    "seconds_remaining": self.rate_limit_seconds_remaining
                }
            }


# Global singleton instance
sync_status = GlobalSyncStatus()
