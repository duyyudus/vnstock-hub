"""Trading position endpoints for authenticated users."""
from datetime import date
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import get_current_user
from app.core.logging_config import get_portfolio_import_logger
from app.db.database import get_db
from app.db.models import TradingPosition
from app.services.llm import POSITION_IMAGE_EXTRACTION_TASK
from app.services.portfolio_import import (
    extract_positions_from_image,
    get_broker,
    is_image_file,
    merge_image_positions,
)

router = APIRouter(prefix="/trading", tags=["trading"])
import_logger = get_portfolio_import_logger()


class TradingPositionResponse(BaseModel):
    id: int
    account_label: str
    ticker: str
    quantity: float
    average_entry_cost: float
    opened_date: Optional[str] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TradingPositionsResponse(BaseModel):
    positions: List[TradingPositionResponse]
    count: int


class TradingPositionCreateRequest(BaseModel):
    account_label: str = Field(..., min_length=1, max_length=120)
    ticker: str = Field(..., min_length=1, max_length=10)
    quantity: float = Field(..., gt=0)
    average_entry_cost: float = Field(..., gt=0)
    opened_date: Optional[date] = None
    target_price: Optional[float] = Field(None, gt=0)
    stop_loss: Optional[float] = Field(None, gt=0)
    notes: Optional[str] = Field(None, max_length=1000)


class TradingPositionUpdateRequest(BaseModel):
    account_label: Optional[str] = Field(None, min_length=1, max_length=120)
    ticker: Optional[str] = Field(None, min_length=1, max_length=10)
    quantity: Optional[float] = Field(None, gt=0)
    average_entry_cost: Optional[float] = Field(None, gt=0)
    opened_date: Optional[date] = None
    target_price: Optional[float] = Field(None, gt=0)
    stop_loss: Optional[float] = Field(None, gt=0)
    notes: Optional[str] = Field(None, max_length=1000)


class TradingImportPosition(BaseModel):
    ticker: str
    quantity: Optional[float] = None
    average_entry_cost: Optional[float] = None


class TradingImportOutcome(BaseModel):
    ticker: str
    quantity: Optional[float] = None
    average_entry_cost: Optional[float] = None
    status: Literal["created", "updated", "skipped"]
    reason: Optional[str] = None


class TradingImportResponse(BaseModel):
    imported_positions: List[TradingImportPosition]
    import_outcomes: List[TradingImportOutcome]
    created_count: int
    updated_count: int
    deleted_count: int
    skipped_count: int
    positions: List[TradingPositionResponse]


def _normalize_account_label(account_label: str) -> str:
    return account_label.strip()


def _normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def _normalize_notes(notes: str | None) -> str | None:
    if notes is None:
        return None
    normalized = notes.strip()
    return normalized or None


def _require_llm_providers() -> List[dict]:
    try:
        providers = settings.resolve_llm_providers(POSITION_IMAGE_EXTRACTION_TASK)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LLM providers configuration is invalid",
        ) from exc
    if not providers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LLM providers are not configured",
        )
    return providers


def _build_position_response(position: TradingPosition) -> TradingPositionResponse:
    return TradingPositionResponse(
        id=position.id,
        account_label=position.account_label,
        ticker=position.ticker,
        quantity=position.quantity,
        average_entry_cost=position.average_entry_cost,
        opened_date=position.opened_date.isoformat() if position.opened_date else None,
        target_price=position.target_price,
        stop_loss=position.stop_loss,
        notes=position.notes,
        created_at=position.created_at.isoformat() if position.created_at else None,
        updated_at=position.updated_at.isoformat() if position.updated_at else None,
    )


async def _find_duplicate_position(
    db: AsyncSession,
    *,
    user_id: int,
    account_label: str,
    ticker: str,
    exclude_position_id: int | None = None,
) -> TradingPosition | None:
    query = select(TradingPosition).where(
        TradingPosition.user_id == user_id,
        func.lower(TradingPosition.account_label) == account_label.lower(),
        TradingPosition.ticker == ticker,
    )
    if exclude_position_id is not None:
        query = query.where(TradingPosition.id != exclude_position_id)
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def _list_user_positions(db: AsyncSession, user_id: int) -> List[TradingPosition]:
    result = await db.execute(
        select(TradingPosition)
        .where(TradingPosition.user_id == user_id)
        .order_by(TradingPosition.account_label, TradingPosition.created_at, TradingPosition.id)
    )
    return result.scalars().all()


@router.get("/positions", response_model=TradingPositionsResponse)
async def list_positions(
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    positions = await _list_user_positions(db, current_user.id)
    return TradingPositionsResponse(
        positions=[_build_position_response(position) for position in positions],
        count=len(positions),
    )


@router.post("/positions", response_model=TradingPositionResponse, status_code=status.HTTP_201_CREATED)
async def create_position(
    payload: TradingPositionCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    account_label = _normalize_account_label(payload.account_label)
    ticker = _normalize_ticker(payload.ticker)
    notes = _normalize_notes(payload.notes)

    if not account_label:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Account label is required")
    if not ticker:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ticker is required")

    existing = await _find_duplicate_position(
        db,
        user_id=current_user.id,
        account_label=account_label,
        ticker=ticker,
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Trading position already exists for this account and ticker",
        )

    position = TradingPosition(
        user_id=current_user.id,
        account_label=account_label,
        ticker=ticker,
        quantity=payload.quantity,
        average_entry_cost=payload.average_entry_cost,
        opened_date=payload.opened_date,
        target_price=payload.target_price,
        stop_loss=payload.stop_loss,
        notes=notes,
    )
    db.add(position)
    await db.commit()
    await db.refresh(position)
    return _build_position_response(position)


@router.post("/import", response_model=TradingImportResponse)
async def import_trading_positions(
    file: List[UploadFile] | None = File(None),
    broker_id: str | None = Form(None),
    account_label: str | None = Form(None),
    opened_date: date | None = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    if not file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is required")
    if not broker_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Broker is required")

    normalized_account_label = _normalize_account_label(account_label or "")
    if not normalized_account_label:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Account label is required")

    broker = get_broker(broker_id)
    if broker is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Broker is invalid")

    files = list(file)
    if any(not is_image_file(upload.filename, upload.content_type) for upload in files):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only screenshot image files are supported for trading import",
        )

    providers = _require_llm_providers()
    imported_positions: List[TradingImportPosition] = []
    import_outcomes: List[TradingImportOutcome] = []
    created_count = 0
    updated_count = 0
    deleted_count = 0
    skipped_count = 0
    aggregated_positions = []

    for upload in files:
        try:
            extracted_positions = await extract_positions_from_image(
                upload,
                providers,
                settings.llm_request_timeout_seconds,
            )
        except Exception as exc:
            import_logger.error(
                "trading_import_llm_error user_id=%s error=%s filename=%s",
                current_user.id,
                exc,
                upload.filename,
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unable to extract positions: {exc}",
            ) from exc

        if not extracted_positions:
            import_logger.warning(
                "trading_import_llm_empty user_id=%s filename=%s",
                current_user.id,
                upload.filename,
            )
            continue
        aggregated_positions.extend(extracted_positions)

    if not aggregated_positions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="LLM did not return any positions",
        )

    scaled_positions = [
        position.__class__(
            ticker=position.ticker,
            average_cost=position.average_cost * broker.average_cost_multiplier,
            quantity=position.quantity,
        )
        for position in aggregated_positions
    ]
    merged_positions, conflicted_tickers = merge_image_positions(scaled_positions)
    if conflicted_tickers:
        skipped_count += len(conflicted_tickers)
        for ticker in conflicted_tickers:
            import_outcomes.append(
                TradingImportOutcome(
                    ticker=ticker,
                    status="skipped",
                    reason="Conflicting values for this ticker across screenshots",
                )
            )

    if not merged_positions and not conflicted_tickers:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="LLM did not return any positions",
        )

    tickers = [position.ticker.strip().upper() for position in merged_positions]
    seen_tickers = set(tickers) | {ticker.strip().upper() for ticker in conflicted_tickers}
    existing_result = await db.execute(
        select(TradingPosition).where(
            TradingPosition.user_id == current_user.id,
            func.lower(TradingPosition.account_label) == normalized_account_label.lower(),
        )
    )
    existing_positions = {
        position.ticker.upper(): position
        for position in existing_result.scalars().all()
    }

    for position in merged_positions:
        ticker = position.ticker.strip().upper()
        average_entry_cost = float(position.average_cost)
        quantity = float(position.quantity) if position.quantity is not None else None
        if not ticker or average_entry_cost <= 0:
            skipped_count += 1
            continue

        imported_positions.append(
            TradingImportPosition(
                ticker=ticker,
                quantity=quantity,
                average_entry_cost=average_entry_cost,
            )
        )

        existing = existing_positions.get(ticker)
        if existing:
            existing.average_entry_cost = average_entry_cost
            if quantity is not None and quantity > 0:
                existing.quantity = quantity
            import_outcomes.append(
                TradingImportOutcome(
                    ticker=ticker,
                    quantity=quantity,
                    average_entry_cost=average_entry_cost,
                    status="updated",
                    reason="Matched an existing trade in this account",
                )
            )
            updated_count += 1
            continue

        if quantity is None or quantity <= 0:
            skipped_count += 1
            import_outcomes.append(
                TradingImportOutcome(
                    ticker=ticker,
                    quantity=quantity,
                    average_entry_cost=average_entry_cost,
                    status="skipped",
                    reason="Quantity is required to create a new trade",
                )
            )
            continue

        db.add(TradingPosition(
            user_id=current_user.id,
            account_label=normalized_account_label,
            ticker=ticker,
            quantity=quantity,
            average_entry_cost=average_entry_cost,
            opened_date=opened_date,
            target_price=None,
            stop_loss=None,
            notes=None,
        ))
        import_outcomes.append(
            TradingImportOutcome(
                ticker=ticker,
                quantity=quantity,
                average_entry_cost=average_entry_cost,
                status="created",
                reason="Created a new trade in this account",
            )
        )
        created_count += 1

    for existing_ticker, existing_position in existing_positions.items():
        if existing_ticker not in seen_tickers:
            await db.delete(existing_position)
            deleted_count += 1

    await db.commit()
    positions = await _list_user_positions(db, current_user.id)
    import_logger.info(
        "trading_import_complete user_id=%s account=%s created=%s updated=%s deleted=%s skipped=%s",
        current_user.id,
        normalized_account_label,
        created_count,
        updated_count,
        deleted_count,
        skipped_count,
    )
    return TradingImportResponse(
        imported_positions=imported_positions,
        import_outcomes=import_outcomes,
        created_count=created_count,
        updated_count=updated_count,
        deleted_count=deleted_count,
        skipped_count=skipped_count,
        positions=[_build_position_response(position) for position in positions],
    )


@router.patch("/positions/{position_id}", response_model=TradingPositionResponse)
async def update_position(
    position_id: int,
    payload: TradingPositionUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    result = await db.execute(
        select(TradingPosition).where(
            TradingPosition.id == position_id,
            TradingPosition.user_id == current_user.id,
        )
    )
    position = result.scalar_one_or_none()
    if not position:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trading position not found")

    if not payload.__fields_set__:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No updates provided")

    next_account_label = (
        _normalize_account_label(payload.account_label)
        if "account_label" in payload.__fields_set__ and payload.account_label is not None
        else position.account_label
    )
    next_ticker = (
        _normalize_ticker(payload.ticker)
        if "ticker" in payload.__fields_set__ and payload.ticker is not None
        else position.ticker
    )
    if not next_account_label:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Account label is required")
    if not next_ticker:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ticker is required")

    duplicate = await _find_duplicate_position(
        db,
        user_id=current_user.id,
        account_label=next_account_label,
        ticker=next_ticker,
        exclude_position_id=position.id,
    )
    if duplicate:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Trading position already exists for this account and ticker",
        )

    if "account_label" in payload.__fields_set__:
        position.account_label = next_account_label
    if "ticker" in payload.__fields_set__:
        position.ticker = next_ticker
    if "quantity" in payload.__fields_set__:
        if payload.quantity is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Quantity must be greater than zero")
        position.quantity = payload.quantity
    if "average_entry_cost" in payload.__fields_set__:
        if payload.average_entry_cost is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Average entry cost must be greater than zero",
            )
        position.average_entry_cost = payload.average_entry_cost
    if "opened_date" in payload.__fields_set__:
        position.opened_date = payload.opened_date
    if "target_price" in payload.__fields_set__:
        position.target_price = payload.target_price
    if "stop_loss" in payload.__fields_set__:
        position.stop_loss = payload.stop_loss
    if "notes" in payload.__fields_set__:
        position.notes = _normalize_notes(payload.notes)

    await db.commit()
    await db.refresh(position)
    return _build_position_response(position)


@router.delete("/positions/{position_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_position(
    position_id: int,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    result = await db.execute(
        select(TradingPosition).where(
            TradingPosition.id == position_id,
            TradingPosition.user_id == current_user.id,
        )
    )
    position = result.scalar_one_or_none()
    if not position:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trading position not found")

    await db.delete(position)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
