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
from app.services.vnstock_service.scheduler import (
    ScheduledSyncAction,
    ScheduledSyncIntervalUnit,
    ScheduledSyncRunStatus,
    ScheduledSyncType,
    SchedulerNotFoundError,
    SchedulerValidationError,
)

router = APIRouter(prefix="/sync", tags=["sync"])


class SyncStatusItem(BaseModel):
    """Status for a single sync operation."""
    is_syncing: bool
    last_sync: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None


class HistoryJobStatusResponse(BaseModel):
    is_running: bool
    total_symbols: int
    processed_symbols: int
    success_symbols: int
    failed_symbols: int
    failed_tickers: List[str] = Field(default_factory=list)
    current_symbol: Optional[str] = None
    last_run_at: Optional[str] = None
    started_at: Optional[str] = None
    error: Optional[str] = None
    progress: float = 0.0


class HistorySyncStatusResponse(BaseModel):
    sync: HistoryJobStatusResponse
    audit: HistoryJobStatusResponse
    repair: HistoryJobStatusResponse


class SyncStatusResponse(BaseModel):
    """Response model for sync status endpoint."""
    fund_performance: SyncStatusItem
    history_sync: HistorySyncStatusResponse
    finance_sync: HistoryJobStatusResponse
    company_sync: HistoryJobStatusResponse
    is_rate_limited: bool = False
    rate_limit_reset_at: Optional[str] = None


class HistorySyncRunRequest(BaseModel):
    force_restart: bool = False
    symbols: Optional[List[str]] = None
    index_symbol: Optional[str] = None


class FinanceSyncRunRequest(BaseModel):
    force_restart: bool = False
    symbols: Optional[List[str]] = None
    index_symbol: Optional[str] = None
    quick_sync: bool = False


class CompanySyncRunRequest(BaseModel):
    force_restart: bool = False
    symbols: Optional[List[str]] = None
    index_symbol: Optional[str] = None
    quick_sync: bool = False


class HistoryAuditRunRequest(BaseModel):
    symbols: Optional[List[str]] = None
    index_symbol: Optional[str] = None
    start_date: str
    end_date: str
    auto_repair: bool = False


class HistoryRepairRunRequest(BaseModel):
    symbols: Optional[List[str]] = None
    index_symbol: Optional[str] = None
    start_date: str
    end_date: str


class HistorySyncActionResponse(BaseModel):
    started: bool
    message: str
    processed_symbols: int = 0
    success_symbols: int = 0
    failed_symbols: int = 0
    state: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class HistoryAuditSymbolResultResponse(BaseModel):
    symbol: str
    local_dates: int
    upstream_dates: int
    missing_dates: int
    repaired_dates: int
    missing_date_samples: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class HistoryAuditActionResponse(HistorySyncActionResponse):
    audited_symbols: int = 0
    symbols_with_gaps: int = 0
    total_missing_dates: int = 0
    total_repaired_dates: int = 0
    results: List[HistoryAuditSymbolResultResponse] = Field(default_factory=list)


class ScheduledSyncJobCreateRequest(BaseModel):
    name: str
    enabled: bool = True
    sync_type: ScheduledSyncType
    sync_action: ScheduledSyncAction
    index_symbol: Optional[str] = None
    symbols: List[str] = Field(default_factory=list)
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    auto_repair: bool = False
    starts_at: str
    interval_value: int
    interval_unit: ScheduledSyncIntervalUnit
    timezone: str = "Asia/Ho_Chi_Minh"
    max_retries: int = 3


class ScheduledSyncJobUpdateRequest(BaseModel):
    name: Optional[str] = None
    enabled: Optional[bool] = None
    sync_type: Optional[ScheduledSyncType] = None
    sync_action: Optional[ScheduledSyncAction] = None
    index_symbol: Optional[str] = None
    symbols: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    auto_repair: Optional[bool] = None
    starts_at: Optional[str] = None
    interval_value: Optional[int] = None
    interval_unit: Optional[ScheduledSyncIntervalUnit] = None
    timezone: Optional[str] = None
    max_retries: Optional[int] = None


class ScheduledSyncJobResponse(BaseModel):
    id: int
    name: str
    enabled: bool
    sync_type: ScheduledSyncType
    sync_action: ScheduledSyncAction
    index_symbol: Optional[str] = None
    symbols: List[str] = Field(default_factory=list)
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    auto_repair: bool = False
    starts_at: str
    interval_value: int
    interval_unit: ScheduledSyncIntervalUnit
    timezone: str
    max_retries: int
    next_run_at: Optional[str] = None
    last_run_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ScheduledSyncJobListResponse(BaseModel):
    jobs: List[ScheduledSyncJobResponse] = Field(default_factory=list)
    count: int


class ScheduledSyncJobRunResponse(BaseModel):
    id: int
    job_id: int
    job_name: str
    sync_type: ScheduledSyncType
    sync_action: ScheduledSyncAction
    attempt_number: int
    status: ScheduledSyncRunStatus
    scheduled_for: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    summary: dict = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ScheduledSyncJobRunListResponse(BaseModel):
    runs: List[ScheduledSyncJobRunResponse] = Field(default_factory=list)
    count: int


@router.get("/status", response_model=SyncStatusResponse)
async def get_sync_status():
    """
    Get current background sync status for all operations.
    """
    history_sync_runtime = sync_status.history_sync
    history_audit_runtime = sync_status.history_audit
    history_repair_runtime = sync_status.history_repair
    finance_sync_runtime = sync_status.finance_sync
    company_sync_runtime = sync_status.company_sync

    return SyncStatusResponse(
        fund_performance=SyncStatusItem(
            is_syncing=sync_status.fund_performance.is_syncing,
            last_sync=sync_status.fund_performance.last_sync,
            error=sync_status.fund_performance.error,
            started_at=sync_status.fund_performance.started_at,
        ),
        history_sync=HistorySyncStatusResponse(
            sync=HistoryJobStatusResponse(
                is_running=history_sync_runtime.is_running,
                total_symbols=history_sync_runtime.total_symbols,
                processed_symbols=history_sync_runtime.processed_symbols,
                success_symbols=history_sync_runtime.success_symbols,
                failed_symbols=history_sync_runtime.failed_symbols,
                failed_tickers=history_sync_runtime.failed_tickers,
                current_symbol=history_sync_runtime.current_symbol,
                last_run_at=history_sync_runtime.last_run_at,
                started_at=history_sync_runtime.started_at,
                error=history_sync_runtime.error,
                progress=history_sync_runtime.progress,
            ),
            audit=HistoryJobStatusResponse(
                is_running=history_audit_runtime.is_running,
                total_symbols=history_audit_runtime.total_symbols,
                processed_symbols=history_audit_runtime.processed_symbols,
                success_symbols=history_audit_runtime.success_symbols,
                failed_symbols=history_audit_runtime.failed_symbols,
                failed_tickers=history_audit_runtime.failed_tickers,
                current_symbol=history_audit_runtime.current_symbol,
                last_run_at=history_audit_runtime.last_run_at,
                started_at=history_audit_runtime.started_at,
                error=history_audit_runtime.error,
                progress=history_audit_runtime.progress,
            ),
            repair=HistoryJobStatusResponse(
                is_running=history_repair_runtime.is_running,
                total_symbols=history_repair_runtime.total_symbols,
                processed_symbols=history_repair_runtime.processed_symbols,
                success_symbols=history_repair_runtime.success_symbols,
                failed_symbols=history_repair_runtime.failed_symbols,
                failed_tickers=history_repair_runtime.failed_tickers,
                current_symbol=history_repair_runtime.current_symbol,
                last_run_at=history_repair_runtime.last_run_at,
                started_at=history_repair_runtime.started_at,
                error=history_repair_runtime.error,
                progress=history_repair_runtime.progress,
            ),
        ),
        finance_sync=HistoryJobStatusResponse(
            is_running=finance_sync_runtime.is_running,
            total_symbols=finance_sync_runtime.total_symbols,
            processed_symbols=finance_sync_runtime.processed_symbols,
            success_symbols=finance_sync_runtime.success_symbols,
            failed_symbols=finance_sync_runtime.failed_symbols,
            failed_tickers=finance_sync_runtime.failed_tickers,
            current_symbol=finance_sync_runtime.current_symbol,
            last_run_at=finance_sync_runtime.last_run_at,
            started_at=finance_sync_runtime.started_at,
            error=finance_sync_runtime.error,
            progress=finance_sync_runtime.progress,
        ),
        company_sync=HistoryJobStatusResponse(
            is_running=company_sync_runtime.is_running,
            total_symbols=company_sync_runtime.total_symbols,
            processed_symbols=company_sync_runtime.processed_symbols,
            success_symbols=company_sync_runtime.success_symbols,
            failed_symbols=company_sync_runtime.failed_symbols,
            failed_tickers=company_sync_runtime.failed_tickers,
            current_symbol=company_sync_runtime.current_symbol,
            last_run_at=company_sync_runtime.last_run_at,
            started_at=company_sync_runtime.started_at,
            error=company_sync_runtime.error,
            progress=company_sync_runtime.progress,
        ),
        is_rate_limited=sync_status.is_rate_limited,
        rate_limit_reset_at=sync_status.rate_limit_reset_at,
    )


@router.post("/history/run", response_model=HistorySyncActionResponse)
async def run_history_sync(
    payload: HistorySyncRunRequest,
    _current_admin=Depends(get_current_admin_user),
):
    try:
        result = await vnstock_service.run_history_sync(
            force_restart=payload.force_restart,
            symbols=payload.symbols,
            index_symbol=payload.index_symbol,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    return HistorySyncActionResponse(**result)


@router.post("/history/audit/run", response_model=HistoryAuditActionResponse)
async def run_history_audit_sync(
    payload: HistoryAuditRunRequest,
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

    try:
        result = await vnstock_service.run_history_audit_sync(
            symbols=payload.symbols,
            index_symbol=payload.index_symbol,
            start_date=payload.start_date,
            end_date=payload.end_date,
            auto_repair=payload.auto_repair,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    return HistoryAuditActionResponse(**result)


@router.post("/history/repair/run", response_model=HistorySyncActionResponse)
async def run_history_repair_sync(
    payload: HistoryRepairRunRequest,
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

    try:
        result = await vnstock_service.run_history_repair_sync(
            symbols=payload.symbols,
            start_date=payload.start_date,
            end_date=payload.end_date,
            index_symbol=payload.index_symbol,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    return HistorySyncActionResponse(**result)


@router.post("/finance/run", response_model=HistorySyncActionResponse)
async def run_finance_sync(
    payload: FinanceSyncRunRequest,
    _current_admin=Depends(get_current_admin_user),
):
    try:
        result = await vnstock_service.run_finance_sync(
            force_restart=payload.force_restart,
            symbols=payload.symbols,
            index_symbol=payload.index_symbol,
            quick_sync=payload.quick_sync,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    return HistorySyncActionResponse(**result)


@router.post("/company/run", response_model=HistorySyncActionResponse)
async def run_company_sync(
    payload: CompanySyncRunRequest,
    _current_admin=Depends(get_current_admin_user),
):
    try:
        result = await vnstock_service.run_company_sync(
            force_restart=payload.force_restart,
            symbols=payload.symbols,
            index_symbol=payload.index_symbol,
            quick_sync=payload.quick_sync,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    return HistorySyncActionResponse(**result)


@router.get("/scheduler/jobs", response_model=ScheduledSyncJobListResponse)
async def list_scheduled_sync_jobs(
    _current_admin=Depends(get_current_admin_user),
):
    jobs = await vnstock_service.scheduler.list_jobs()
    return ScheduledSyncJobListResponse(jobs=jobs, count=len(jobs))


@router.post("/scheduler/jobs", response_model=ScheduledSyncJobResponse)
async def create_scheduled_sync_job(
    payload: ScheduledSyncJobCreateRequest,
    _current_admin=Depends(get_current_admin_user),
):
    try:
        job = await vnstock_service.scheduler.create_job(payload.model_dump(mode="json"))
    except SchedulerValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    return ScheduledSyncJobResponse(**job)


@router.patch("/scheduler/jobs/{job_id}", response_model=ScheduledSyncJobResponse)
async def update_scheduled_sync_job(
    job_id: int,
    payload: ScheduledSyncJobUpdateRequest,
    _current_admin=Depends(get_current_admin_user),
):
    try:
        job = await vnstock_service.scheduler.update_job(
            job_id,
            payload.model_dump(exclude_unset=True, mode="json"),
        )
    except SchedulerNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except SchedulerValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    return ScheduledSyncJobResponse(**job)


@router.delete("/scheduler/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_scheduled_sync_job(
    job_id: int,
    _current_admin=Depends(get_current_admin_user),
):
    try:
        await vnstock_service.scheduler.delete_job(job_id)
    except SchedulerNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get("/scheduler/runs", response_model=ScheduledSyncJobRunListResponse)
async def list_scheduled_sync_runs(
    limit: int = 50,
    _current_admin=Depends(get_current_admin_user),
):
    runs = await vnstock_service.scheduler.list_runs(limit=limit)
    return ScheduledSyncJobRunListResponse(runs=runs, count=len(runs))
