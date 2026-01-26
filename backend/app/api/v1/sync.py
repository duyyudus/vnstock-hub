"""
Sync status API endpoints.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from app.services.sync_status import sync_status

router = APIRouter(prefix="/sync", tags=["sync"])


class SyncStatusItem(BaseModel):
    """Status for a single sync operation."""
    is_syncing: bool
    last_sync: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None


class SyncStatusResponse(BaseModel):
    """Response model for sync status endpoint."""
    fund_performance: SyncStatusItem
    is_rate_limited: bool = False
    rate_limit_reset_at: Optional[str] = None


@router.get("/status", response_model=SyncStatusResponse)
async def get_sync_status():
    """
    Get current background sync status for all operations.
    
    Returns:
        SyncStatusResponse with status for each sync operation type.
    """
    return SyncStatusResponse(
        fund_performance=SyncStatusItem(
            is_syncing=sync_status.fund_performance.is_syncing,
            last_sync=sync_status.fund_performance.last_sync,
            error=sync_status.fund_performance.error,
            started_at=sync_status.fund_performance.started_at
        ),
        is_rate_limited=sync_status.is_rate_limited,
        rate_limit_reset_at=sync_status.rate_limit_reset_at
    )
