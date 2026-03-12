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


class PriceJobStatusResponse(BaseModel):
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


class PriceSyncStatusResponse(BaseModel):
    sync: PriceJobStatusResponse
    audit: PriceJobStatusResponse
    repair: PriceJobStatusResponse


class SyncStatusResponse(BaseModel):
    """Response model for sync status endpoint."""
    fund_performance: SyncStatusItem
    price_sync: PriceSyncStatusResponse
    finance_sync: PriceJobStatusResponse
    company_sync: PriceJobStatusResponse
    is_rate_limited: bool = False
    rate_limit_reset_at: Optional[str] = None


class PriceSyncRunRequest(BaseModel):
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


class PriceAuditRunRequest(BaseModel):
    symbols: Optional[List[str]] = None
    index_symbol: Optional[str] = None
    start_date: str
    end_date: str
    auto_repair: bool = False


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
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class PriceAuditSymbolResultResponse(BaseModel):
    symbol: str
    local_dates: int
    upstream_dates: int
    missing_dates: int
    repaired_dates: int
    missing_date_samples: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class PriceAuditActionResponse(PriceSyncActionResponse):
    audited_symbols: int = 0
    symbols_with_gaps: int = 0
    total_missing_dates: int = 0
    total_repaired_dates: int = 0
    results: List[PriceAuditSymbolResultResponse] = Field(default_factory=list)


@router.get("/status", response_model=SyncStatusResponse)
async def get_sync_status():
    """
    Get current background sync status for all operations.
    """
    price_sync_runtime = sync_status.price_sync
    price_audit_runtime = sync_status.price_audit
    price_repair_runtime = sync_status.price_repair
    finance_sync_runtime = sync_status.finance_sync
    company_sync_runtime = sync_status.company_sync

    return SyncStatusResponse(
        fund_performance=SyncStatusItem(
            is_syncing=sync_status.fund_performance.is_syncing,
            last_sync=sync_status.fund_performance.last_sync,
            error=sync_status.fund_performance.error,
            started_at=sync_status.fund_performance.started_at,
        ),
        price_sync=PriceSyncStatusResponse(
            sync=PriceJobStatusResponse(
                is_running=price_sync_runtime.is_running,
                total_symbols=price_sync_runtime.total_symbols,
                processed_symbols=price_sync_runtime.processed_symbols,
                success_symbols=price_sync_runtime.success_symbols,
                failed_symbols=price_sync_runtime.failed_symbols,
                failed_tickers=price_sync_runtime.failed_tickers,
                current_symbol=price_sync_runtime.current_symbol,
                last_run_at=price_sync_runtime.last_run_at,
                started_at=price_sync_runtime.started_at,
                error=price_sync_runtime.error,
                progress=price_sync_runtime.progress,
            ),
            audit=PriceJobStatusResponse(
                is_running=price_audit_runtime.is_running,
                total_symbols=price_audit_runtime.total_symbols,
                processed_symbols=price_audit_runtime.processed_symbols,
                success_symbols=price_audit_runtime.success_symbols,
                failed_symbols=price_audit_runtime.failed_symbols,
                failed_tickers=price_audit_runtime.failed_tickers,
                current_symbol=price_audit_runtime.current_symbol,
                last_run_at=price_audit_runtime.last_run_at,
                started_at=price_audit_runtime.started_at,
                error=price_audit_runtime.error,
                progress=price_audit_runtime.progress,
            ),
            repair=PriceJobStatusResponse(
                is_running=price_repair_runtime.is_running,
                total_symbols=price_repair_runtime.total_symbols,
                processed_symbols=price_repair_runtime.processed_symbols,
                success_symbols=price_repair_runtime.success_symbols,
                failed_symbols=price_repair_runtime.failed_symbols,
                failed_tickers=price_repair_runtime.failed_tickers,
                current_symbol=price_repair_runtime.current_symbol,
                last_run_at=price_repair_runtime.last_run_at,
                started_at=price_repair_runtime.started_at,
                error=price_repair_runtime.error,
                progress=price_repair_runtime.progress,
            ),
        ),
        finance_sync=PriceJobStatusResponse(
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
        company_sync=PriceJobStatusResponse(
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


@router.post("/prices/run", response_model=PriceSyncActionResponse)
async def run_price_sync(
    payload: PriceSyncRunRequest,
    _current_admin=Depends(get_current_admin_user),
):
    try:
        result = await vnstock_service.run_price_sync(
            force_restart=payload.force_restart,
            symbols=payload.symbols,
            index_symbol=payload.index_symbol,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    return PriceSyncActionResponse(**result)


@router.post("/prices/audit/run", response_model=PriceAuditActionResponse)
async def run_price_audit_sync(
    payload: PriceAuditRunRequest,
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
        result = await vnstock_service.run_price_audit_sync(
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
    return PriceAuditActionResponse(**result)


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


@router.post("/finance/run", response_model=PriceSyncActionResponse)
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
    return PriceSyncActionResponse(**result)


@router.post("/company/run", response_model=PriceSyncActionResponse)
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
    return PriceSyncActionResponse(**result)
