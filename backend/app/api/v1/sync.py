"""
Sync status and control API endpoints.
"""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.deps import get_current_admin_user
from app.services.sync_status import sync_status
from app.services.vnstock_service import vnstock_service

router = APIRouter(prefix="/sync", tags=["sync"])


class SyncStatusItem(BaseModel):
    """Status for a single sync operation."""
    is_syncing: bool
    last_sync: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None


class PriceBootstrapStatusResponse(BaseModel):
    state: str
    total_symbols: int
    processed_symbols: int
    success_symbols: int
    failed_symbols: int
    current_symbol: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    progress: float = 0.0


class PriceJobStatusResponse(BaseModel):
    is_running: bool
    total_symbols: int
    processed_symbols: int
    current_symbol: Optional[str] = None
    last_run_at: Optional[str] = None
    started_at: Optional[str] = None
    error: Optional[str] = None
    progress: float = 0.0


class PriceSyncStatusResponse(BaseModel):
    bootstrap: PriceBootstrapStatusResponse
    incremental: PriceJobStatusResponse
    repair: PriceJobStatusResponse


class SyncStatusResponse(BaseModel):
    """Response model for sync status endpoint."""
    fund_performance: SyncStatusItem
    price_sync: PriceSyncStatusResponse
    is_rate_limited: bool = False
    rate_limit_reset_at: Optional[str] = None


class PriceBootstrapStartRequest(BaseModel):
    force_restart: bool = False


class PriceIncrementalRunRequest(BaseModel):
    heal_window_days: int = Field(default=7, ge=1, le=60)


class PriceRepairRunRequest(BaseModel):
    symbols: List[str] = Field(min_length=1)
    start_date: str
    end_date: str


class PriceSyncActionResponse(BaseModel):
    started: bool
    message: str
    processed_symbols: int = 0
    success_symbols: int = 0
    failed_symbols: int = 0
    state: Optional[str] = None
    window_start_date: Optional[str] = None
    window_end_date: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class PriceBootstrapDetailedStatusResponse(BaseModel):
    state: str
    total_symbols: int
    processed_symbols: int
    success_symbols: int
    failed_symbols: int
    current_symbol: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    progress: float = 0.0
    db_summary: dict


@router.get("/status", response_model=SyncStatusResponse)
async def get_sync_status():
    """
    Get current background sync status for all operations.
    """
    price_bootstrap = sync_status.price_bootstrap
    price_incremental = sync_status.price_incremental
    price_repair = sync_status.price_repair

    return SyncStatusResponse(
        fund_performance=SyncStatusItem(
            is_syncing=sync_status.fund_performance.is_syncing,
            last_sync=sync_status.fund_performance.last_sync,
            error=sync_status.fund_performance.error,
            started_at=sync_status.fund_performance.started_at,
        ),
        price_sync=PriceSyncStatusResponse(
            bootstrap=PriceBootstrapStatusResponse(
                state=price_bootstrap.state,
                total_symbols=price_bootstrap.total_symbols,
                processed_symbols=price_bootstrap.processed_symbols,
                success_symbols=price_bootstrap.success_symbols,
                failed_symbols=price_bootstrap.failed_symbols,
                current_symbol=price_bootstrap.current_symbol,
                started_at=price_bootstrap.started_at,
                completed_at=price_bootstrap.completed_at,
                error=price_bootstrap.error,
                progress=price_bootstrap.progress,
            ),
            incremental=PriceJobStatusResponse(
                is_running=price_incremental.is_running,
                total_symbols=price_incremental.total_symbols,
                processed_symbols=price_incremental.processed_symbols,
                current_symbol=price_incremental.current_symbol,
                last_run_at=price_incremental.last_run_at,
                started_at=price_incremental.started_at,
                error=price_incremental.error,
                progress=price_incremental.progress,
            ),
            repair=PriceJobStatusResponse(
                is_running=price_repair.is_running,
                total_symbols=price_repair.total_symbols,
                processed_symbols=price_repair.processed_symbols,
                current_symbol=price_repair.current_symbol,
                last_run_at=price_repair.last_run_at,
                started_at=price_repair.started_at,
                error=price_repair.error,
                progress=price_repair.progress,
            ),
        ),
        is_rate_limited=sync_status.is_rate_limited,
        rate_limit_reset_at=sync_status.rate_limit_reset_at,
    )


@router.post("/prices/bootstrap/start", response_model=PriceSyncActionResponse)
async def start_price_bootstrap(
    payload: PriceBootstrapStartRequest,
    _current_admin=Depends(get_current_admin_user),
):
    result = await vnstock_service.start_price_bootstrap(force_restart=payload.force_restart)
    return PriceSyncActionResponse(**result)


@router.get("/prices/bootstrap/status", response_model=PriceBootstrapDetailedStatusResponse)
async def get_price_bootstrap_status(
    _current_admin=Depends(get_current_admin_user),
):
    result = await vnstock_service.get_price_bootstrap_status()
    return PriceBootstrapDetailedStatusResponse(**result)


@router.post("/prices/incremental/run", response_model=PriceSyncActionResponse)
async def run_price_incremental_sync(
    payload: PriceIncrementalRunRequest,
    _current_admin=Depends(get_current_admin_user),
):
    result = await vnstock_service.run_price_incremental_sync(
        heal_window_days=payload.heal_window_days,
    )
    return PriceSyncActionResponse(**result)


@router.post("/prices/repair/run", response_model=PriceSyncActionResponse)
async def run_price_repair_sync(
    payload: PriceRepairRunRequest,
    _current_admin=Depends(get_current_admin_user),
):
    try:
        start_date = datetime.strptime(payload.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(payload.end_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date and end_date must be YYYY-MM-DD",
        )

    if end_date < start_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="end_date must be on or after start_date",
        )

    result = await vnstock_service.run_price_repair_sync(
        symbols=payload.symbols,
        start_date=payload.start_date,
        end_date=payload.end_date,
    )
    return PriceSyncActionResponse(**result)
