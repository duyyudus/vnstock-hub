"""
Global sync status tracking for background tasks.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SyncStatusData:
    """Status for a specific sync operation."""
    is_syncing: bool = False
    last_sync: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None


@dataclass
class GlobalSyncStatus:
    """Global sync status tracker for all background sync operations."""
    fund_performance: SyncStatusData = field(default_factory=SyncStatusData)
    
    def start_fund_performance_sync(self):
        """Mark fund performance sync as started."""
        self.fund_performance.is_syncing = True
        self.fund_performance.started_at = datetime.now().isoformat()
        self.fund_performance.error = None
    
    def complete_fund_performance_sync(self, success: bool = True, error: Optional[str] = None):
        """Mark fund performance sync as completed."""
        self.fund_performance.is_syncing = False
        self.fund_performance.last_sync = datetime.now().isoformat()
        if not success:
            self.fund_performance.error = error


# Global singleton instance
sync_status = GlobalSyncStatus()
