"""
Thread-safe global sync status tracking for background tasks.

All state modifications are protected by RLock to prevent race conditions
when multiple threads/coroutines access the sync status concurrently.
"""
import threading
from dataclasses import dataclass, field
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
    total_symbols: int = 0
    processed_symbols: int = 0


@dataclass
class HistoryJobStatusData:
    """Runtime status snapshot for a symbol-based sync job."""
    is_running: bool = False
    total_symbols: int = 0
    processed_symbols: int = 0
    success_symbols: int = 0
    failed_symbols: int = 0
    failed_tickers: list[str] = field(default_factory=list)
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
        self._fund_performance_total_symbols = 0
        self._fund_performance_processed_symbols = 0

        # History unified sync status
        self._history_sync_is_running = False
        self._history_sync_total_symbols = 0
        self._history_sync_processed_symbols = 0
        self._history_sync_success_symbols = 0
        self._history_sync_failed_symbols = 0
        self._history_sync_failed_tickers: list[str] = []
        self._history_sync_current_symbol: Optional[str] = None
        self._history_sync_last_run_at: Optional[str] = None
        self._history_sync_started_at: Optional[str] = None
        self._history_sync_error: Optional[str] = None
        self._history_sync_progress: float = 0.0

        # History audit status
        self._history_audit_is_running = False
        self._history_audit_total_symbols = 0
        self._history_audit_processed_symbols = 0
        self._history_audit_success_symbols = 0
        self._history_audit_failed_symbols = 0
        self._history_audit_failed_tickers: list[str] = []
        self._history_audit_current_symbol: Optional[str] = None
        self._history_audit_last_run_at: Optional[str] = None
        self._history_audit_started_at: Optional[str] = None
        self._history_audit_error: Optional[str] = None
        self._history_audit_progress: float = 0.0

        # History repair status
        self._history_repair_is_running = False
        self._history_repair_total_symbols = 0
        self._history_repair_processed_symbols = 0
        self._history_repair_success_symbols = 0
        self._history_repair_failed_symbols = 0
        self._history_repair_failed_tickers: list[str] = []
        self._history_repair_current_symbol: Optional[str] = None
        self._history_repair_last_run_at: Optional[str] = None
        self._history_repair_started_at: Optional[str] = None
        self._history_repair_error: Optional[str] = None
        self._history_repair_progress: float = 0.0

        # Finance sync status
        self._finance_sync_is_running = False
        self._finance_sync_total_symbols = 0
        self._finance_sync_processed_symbols = 0
        self._finance_sync_success_symbols = 0
        self._finance_sync_failed_symbols = 0
        self._finance_sync_failed_tickers: list[str] = []
        self._finance_sync_current_symbol: Optional[str] = None
        self._finance_sync_last_run_at: Optional[str] = None
        self._finance_sync_started_at: Optional[str] = None
        self._finance_sync_error: Optional[str] = None
        self._finance_sync_progress: float = 0.0

        # Company sync status
        self._company_sync_is_running = False
        self._company_sync_total_symbols = 0
        self._company_sync_processed_symbols = 0
        self._company_sync_success_symbols = 0
        self._company_sync_failed_symbols = 0
        self._company_sync_failed_tickers: list[str] = []
        self._company_sync_current_symbol: Optional[str] = None
        self._company_sync_last_run_at: Optional[str] = None
        self._company_sync_started_at: Optional[str] = None
        self._company_sync_error: Optional[str] = None
        self._company_sync_progress: float = 0.0

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
                progress=self._fund_performance_progress,
                total_symbols=self._fund_performance_total_symbols,
                processed_symbols=self._fund_performance_processed_symbols,
            )

    @property
    def history_sync(self) -> HistoryJobStatusData:
        """Get unified history sync runtime status as immutable snapshot."""
        with self._lock:
            return HistoryJobStatusData(
                is_running=self._history_sync_is_running,
                total_symbols=self._history_sync_total_symbols,
                processed_symbols=self._history_sync_processed_symbols,
                success_symbols=self._history_sync_success_symbols,
                failed_symbols=self._history_sync_failed_symbols,
                failed_tickers=list(self._history_sync_failed_tickers),
                current_symbol=self._history_sync_current_symbol,
                last_run_at=self._history_sync_last_run_at,
                started_at=self._history_sync_started_at,
                error=self._history_sync_error,
                progress=self._history_sync_progress,
            )

    @property
    def history_audit(self) -> HistoryJobStatusData:
        """Get history audit runtime status as immutable snapshot."""
        with self._lock:
            return HistoryJobStatusData(
                is_running=self._history_audit_is_running,
                total_symbols=self._history_audit_total_symbols,
                processed_symbols=self._history_audit_processed_symbols,
                success_symbols=self._history_audit_success_symbols,
                failed_symbols=self._history_audit_failed_symbols,
                failed_tickers=list(self._history_audit_failed_tickers),
                current_symbol=self._history_audit_current_symbol,
                last_run_at=self._history_audit_last_run_at,
                started_at=self._history_audit_started_at,
                error=self._history_audit_error,
                progress=self._history_audit_progress,
            )

    @property
    def history_repair(self) -> HistoryJobStatusData:
        """Get history repair runtime status as immutable snapshot."""
        with self._lock:
            return HistoryJobStatusData(
                is_running=self._history_repair_is_running,
                total_symbols=self._history_repair_total_symbols,
                processed_symbols=self._history_repair_processed_symbols,
                success_symbols=self._history_repair_success_symbols,
                failed_symbols=self._history_repair_failed_symbols,
                failed_tickers=list(self._history_repair_failed_tickers),
                current_symbol=self._history_repair_current_symbol,
                last_run_at=self._history_repair_last_run_at,
                started_at=self._history_repair_started_at,
                error=self._history_repair_error,
                progress=self._history_repair_progress,
            )

    @property
    def finance_sync(self) -> HistoryJobStatusData:
        """Get finance sync runtime status as immutable snapshot."""
        with self._lock:
            return HistoryJobStatusData(
                is_running=self._finance_sync_is_running,
                total_symbols=self._finance_sync_total_symbols,
                processed_symbols=self._finance_sync_processed_symbols,
                success_symbols=self._finance_sync_success_symbols,
                failed_symbols=self._finance_sync_failed_symbols,
                failed_tickers=list(self._finance_sync_failed_tickers),
                current_symbol=self._finance_sync_current_symbol,
                last_run_at=self._finance_sync_last_run_at,
                started_at=self._finance_sync_started_at,
                error=self._finance_sync_error,
                progress=self._finance_sync_progress,
            )

    @property
    def company_sync(self) -> HistoryJobStatusData:
        """Get company sync runtime status as immutable snapshot."""
        with self._lock:
            return HistoryJobStatusData(
                is_running=self._company_sync_is_running,
                total_symbols=self._company_sync_total_symbols,
                processed_symbols=self._company_sync_processed_symbols,
                success_symbols=self._company_sync_success_symbols,
                failed_symbols=self._company_sync_failed_symbols,
                failed_tickers=list(self._company_sync_failed_tickers),
                current_symbol=self._company_sync_current_symbol,
                last_run_at=self._company_sync_last_run_at,
                started_at=self._company_sync_started_at,
                error=self._company_sync_error,
                progress=self._company_sync_progress,
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
            self._fund_performance_total_symbols = 0
            self._fund_performance_processed_symbols = 0

    def update_fund_performance_progress(
        self,
        progress: float,
        processed_symbols: Optional[int] = None,
        total_symbols: Optional[int] = None,
    ) -> None:
        """
        Update fund performance sync progress. Thread-safe.

        Args:
            progress: Progress value between 0.0 and 1.0
        """
        with self._lock:
            self._fund_performance_progress = min(1.0, max(0.0, progress))
            if total_symbols is not None:
                self._fund_performance_total_symbols = max(0, total_symbols)
            if processed_symbols is not None:
                self._fund_performance_processed_symbols = max(0, processed_symbols)

    def complete_fund_performance_sync(
        self,
        success: bool = True,
        error: Optional[str] = None,
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
            if success and self._fund_performance_total_symbols:
                self._fund_performance_processed_symbols = self._fund_performance_total_symbols
            if not success:
                self._fund_performance_error = error

    def _start_symbol_job(self, prefix: str, total_symbols: int) -> None:
        setattr(self, f"_{prefix}_is_running", True)
        setattr(self, f"_{prefix}_total_symbols", max(0, total_symbols))
        setattr(self, f"_{prefix}_processed_symbols", 0)
        setattr(self, f"_{prefix}_success_symbols", 0)
        setattr(self, f"_{prefix}_failed_symbols", 0)
        setattr(self, f"_{prefix}_failed_tickers", [])
        setattr(self, f"_{prefix}_current_symbol", None)
        setattr(self, f"_{prefix}_started_at", datetime.now().isoformat())
        setattr(self, f"_{prefix}_error", None)
        setattr(self, f"_{prefix}_progress", 0.0)

    def _update_symbol_job_progress(
        self,
        prefix: str,
        processed_symbols: int,
        success_symbols: int,
        failed_symbols: int,
        current_symbol: Optional[str],
        failed_tickers: Optional[list[str]] = None,
    ) -> None:
        setattr(self, f"_{prefix}_processed_symbols", max(0, processed_symbols))
        setattr(self, f"_{prefix}_success_symbols", max(0, success_symbols))
        setattr(self, f"_{prefix}_failed_symbols", max(0, failed_symbols))
        if failed_tickers is not None:
            setattr(self, f"_{prefix}_failed_tickers", list(dict.fromkeys(failed_tickers)))
        setattr(self, f"_{prefix}_current_symbol", current_symbol)

        total_symbols = getattr(self, f"_{prefix}_total_symbols")
        if total_symbols > 0:
            progress = getattr(self, f"_{prefix}_processed_symbols") / total_symbols
            setattr(self, f"_{prefix}_progress", min(1.0, max(0.0, progress)))

    def _complete_symbol_job(self, prefix: str, success: bool, error: Optional[str] = None) -> None:
        setattr(self, f"_{prefix}_is_running", False)
        setattr(self, f"_{prefix}_current_symbol", None)
        setattr(self, f"_{prefix}_last_run_at", datetime.now().isoformat())

        current_progress = getattr(self, f"_{prefix}_progress")
        setattr(self, f"_{prefix}_progress", 1.0 if success else current_progress)
        setattr(self, f"_{prefix}_error", error)

    def start_history_sync(self, total_symbols: int) -> None:
        """Mark unified history sync as started."""
        with self._lock:
            self._start_symbol_job("history_sync", total_symbols)

    def update_history_sync_progress(
        self,
        processed_symbols: int,
        success_symbols: int,
        failed_symbols: int,
        current_symbol: Optional[str],
        failed_tickers: Optional[list[str]] = None,
    ) -> None:
        """Update unified history sync progress."""
        with self._lock:
            self._update_symbol_job_progress(
                "history_sync",
                processed_symbols,
                success_symbols,
                failed_symbols,
                current_symbol,
                failed_tickers,
            )

    def complete_history_sync(self, success: bool, error: Optional[str] = None) -> None:
        """Mark unified history sync as completed."""
        with self._lock:
            self._complete_symbol_job("history_sync", success, error)

    def start_history_audit(self, total_symbols: int) -> None:
        """Mark history audit as started."""
        with self._lock:
            self._start_symbol_job("history_audit", total_symbols)

    def update_history_audit_progress(
        self,
        processed_symbols: int,
        success_symbols: int,
        failed_symbols: int,
        current_symbol: Optional[str],
        failed_tickers: Optional[list[str]] = None,
    ) -> None:
        """Update history audit progress."""
        with self._lock:
            self._update_symbol_job_progress(
                "history_audit",
                processed_symbols,
                success_symbols,
                failed_symbols,
                current_symbol,
                failed_tickers,
            )

    def complete_history_audit(self, success: bool, error: Optional[str] = None) -> None:
        """Mark history audit as completed."""
        with self._lock:
            self._complete_symbol_job("history_audit", success, error)

    def start_history_repair(self, total_symbols: int) -> None:
        """Mark history repair sync as started."""
        with self._lock:
            self._start_symbol_job("history_repair", total_symbols)

    def update_history_repair_progress(
        self,
        processed_symbols: int,
        success_symbols: int,
        failed_symbols: int,
        current_symbol: Optional[str],
        failed_tickers: Optional[list[str]] = None,
    ) -> None:
        """Update history repair progress."""
        with self._lock:
            self._update_symbol_job_progress(
                "history_repair",
                processed_symbols,
                success_symbols,
                failed_symbols,
                current_symbol,
                failed_tickers,
            )

    def complete_history_repair(self, success: bool, error: Optional[str] = None) -> None:
        """Mark history repair sync as completed."""
        with self._lock:
            self._complete_symbol_job("history_repair", success, error)

    def start_finance_sync(self, total_symbols: int) -> None:
        """Mark finance sync as started."""
        with self._lock:
            self._start_symbol_job("finance_sync", total_symbols)

    def update_finance_sync_progress(
        self,
        processed_symbols: int,
        success_symbols: int,
        failed_symbols: int,
        current_symbol: Optional[str],
        failed_tickers: Optional[list[str]] = None,
    ) -> None:
        """Update finance sync progress."""
        with self._lock:
            self._update_symbol_job_progress(
                "finance_sync",
                processed_symbols,
                success_symbols,
                failed_symbols,
                current_symbol,
                failed_tickers,
            )

    def complete_finance_sync(self, success: bool, error: Optional[str] = None) -> None:
        """Mark finance sync as completed."""
        with self._lock:
            self._complete_symbol_job("finance_sync", success, error)

    def start_company_sync(self, total_symbols: int) -> None:
        """Mark company sync as started."""
        with self._lock:
            self._start_symbol_job("company_sync", total_symbols)

    def update_company_sync_progress(
        self,
        processed_symbols: int,
        success_symbols: int,
        failed_symbols: int,
        current_symbol: Optional[str],
        failed_tickers: Optional[list[str]] = None,
    ) -> None:
        """Update company sync progress."""
        with self._lock:
            self._update_symbol_job_progress(
                "company_sync",
                processed_symbols,
                success_symbols,
                failed_symbols,
                current_symbol,
                failed_tickers,
            )

    def complete_company_sync(self, success: bool, error: Optional[str] = None) -> None:
        """Mark company sync as completed."""
        with self._lock:
            self._complete_symbol_job("company_sync", success, error)

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
                    "progress": self._fund_performance_progress,
                    "total_symbols": self._fund_performance_total_symbols,
                    "processed_symbols": self._fund_performance_processed_symbols,
                },
                "history_sync": {
                    "sync": {
                        "is_running": self._history_sync_is_running,
                        "total_symbols": self._history_sync_total_symbols,
                        "processed_symbols": self._history_sync_processed_symbols,
                        "success_symbols": self._history_sync_success_symbols,
                        "failed_symbols": self._history_sync_failed_symbols,
                        "failed_tickers": list(self._history_sync_failed_tickers),
                        "current_symbol": self._history_sync_current_symbol,
                        "last_run_at": self._history_sync_last_run_at,
                        "started_at": self._history_sync_started_at,
                        "error": self._history_sync_error,
                        "progress": self._history_sync_progress,
                    },
                    "audit": {
                        "is_running": self._history_audit_is_running,
                        "total_symbols": self._history_audit_total_symbols,
                        "processed_symbols": self._history_audit_processed_symbols,
                        "success_symbols": self._history_audit_success_symbols,
                        "failed_symbols": self._history_audit_failed_symbols,
                        "failed_tickers": list(self._history_audit_failed_tickers),
                        "current_symbol": self._history_audit_current_symbol,
                        "last_run_at": self._history_audit_last_run_at,
                        "started_at": self._history_audit_started_at,
                        "error": self._history_audit_error,
                        "progress": self._history_audit_progress,
                    },
                    "repair": {
                        "is_running": self._history_repair_is_running,
                        "total_symbols": self._history_repair_total_symbols,
                        "processed_symbols": self._history_repair_processed_symbols,
                        "success_symbols": self._history_repair_success_symbols,
                        "failed_symbols": self._history_repair_failed_symbols,
                        "failed_tickers": list(self._history_repair_failed_tickers),
                        "current_symbol": self._history_repair_current_symbol,
                        "last_run_at": self._history_repair_last_run_at,
                        "started_at": self._history_repair_started_at,
                        "error": self._history_repair_error,
                        "progress": self._history_repair_progress,
                    },
                },
                "finance_sync": {
                    "is_running": self._finance_sync_is_running,
                    "total_symbols": self._finance_sync_total_symbols,
                    "processed_symbols": self._finance_sync_processed_symbols,
                    "success_symbols": self._finance_sync_success_symbols,
                    "failed_symbols": self._finance_sync_failed_symbols,
                    "failed_tickers": list(self._finance_sync_failed_tickers),
                    "current_symbol": self._finance_sync_current_symbol,
                    "last_run_at": self._finance_sync_last_run_at,
                    "started_at": self._finance_sync_started_at,
                    "error": self._finance_sync_error,
                    "progress": self._finance_sync_progress,
                },
                "company_sync": {
                    "is_running": self._company_sync_is_running,
                    "total_symbols": self._company_sync_total_symbols,
                    "processed_symbols": self._company_sync_processed_symbols,
                    "success_symbols": self._company_sync_success_symbols,
                    "failed_symbols": self._company_sync_failed_symbols,
                    "failed_tickers": list(self._company_sync_failed_tickers),
                    "current_symbol": self._company_sync_current_symbol,
                    "last_run_at": self._company_sync_last_run_at,
                    "started_at": self._company_sync_started_at,
                    "error": self._company_sync_error,
                    "progress": self._company_sync_progress,
                },
                "rate_limit": {
                    "is_limited": self._is_rate_limited,
                    "reset_at": self._rate_limit_reset_at.isoformat() if self._rate_limit_reset_at else None,
                    "seconds_remaining": self.rate_limit_seconds_remaining,
                },
            }


# Global singleton instance
sync_status = GlobalSyncStatus()
