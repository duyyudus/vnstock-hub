"""Portfolio position endpoints for authenticated users."""
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.core.config import settings
from app.core.logging_config import get_portfolio_import_logger
from app.db.database import get_db
from app.db.models import PortfolioPosition
from app.services.portfolio_import import (
    CropSettings,
    extract_positions_from_rows,
    get_broker,
    list_brokers,
    load_cropped_rows,
)

router = APIRouter(prefix="/portfolio", tags=["portfolio"])
import_logger = get_portfolio_import_logger()


class PortfolioPositionResponse(BaseModel):
    id: int
    ticker: str
    quantity: float
    average_cost: Optional[float] = None
    purchase_date: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class PortfolioPositionsResponse(BaseModel):
    positions: List[PortfolioPositionResponse]
    count: int


class PortfolioPositionCreateRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    quantity: float = Field(..., gt=0)
    average_cost: Optional[float] = Field(None, gt=0)
    purchase_date: Optional[date] = None


class PortfolioPositionUpdateRequest(BaseModel):
    quantity: Optional[float] = Field(None, gt=0)
    average_cost: Optional[float] = Field(None, gt=0)
    purchase_date: Optional[date] = None


class BrokerProfileResponse(BaseModel):
    id: str
    name: str
    sheet: Optional[str]
    top_left: str
    bottom_right: str


class PortfolioImportPosition(BaseModel):
    ticker: str
    quantity: float


class PortfolioImportResponse(BaseModel):
    imported_positions: List[PortfolioImportPosition]
    created_count: int
    updated_count: int
    skipped_count: int
    positions: List[PortfolioPositionResponse]


def _normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def _build_position_response(position: PortfolioPosition) -> PortfolioPositionResponse:
    return PortfolioPositionResponse(
        id=position.id,
        ticker=position.ticker,
        quantity=position.quantity,
        average_cost=position.average_cost,
        purchase_date=position.purchase_date.isoformat() if position.purchase_date else None,
        created_at=position.created_at.isoformat() if position.created_at else None,
        updated_at=position.updated_at.isoformat() if position.updated_at else None
    )


@router.get("/positions", response_model=PortfolioPositionsResponse)
async def list_positions(
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    result = await db.execute(
        select(PortfolioPosition)
        .where(PortfolioPosition.user_id == current_user.id)
        .order_by(PortfolioPosition.created_at)
    )
    positions = result.scalars().all()
    return PortfolioPositionsResponse(
        positions=[_build_position_response(position) for position in positions],
        count=len(positions)
    )


@router.post("/positions", response_model=PortfolioPositionResponse, status_code=status.HTTP_201_CREATED)
async def create_position(
    payload: PortfolioPositionCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    ticker = _normalize_ticker(payload.ticker)
    if not ticker:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ticker is required")

    existing = await db.execute(
        select(PortfolioPosition).where(
            PortfolioPosition.user_id == current_user.id,
            PortfolioPosition.ticker == ticker
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Position already exists for this ticker")

    position = PortfolioPosition(
        user_id=current_user.id,
        ticker=ticker,
        quantity=payload.quantity,
        average_cost=payload.average_cost,
        purchase_date=payload.purchase_date
    )
    db.add(position)
    await db.commit()
    await db.refresh(position)
    return _build_position_response(position)


@router.patch("/positions/{position_id}", response_model=PortfolioPositionResponse)
async def update_position(
    position_id: int,
    payload: PortfolioPositionUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    result = await db.execute(
        select(PortfolioPosition).where(
            PortfolioPosition.id == position_id,
            PortfolioPosition.user_id == current_user.id
        )
    )
    position = result.scalar_one_or_none()
    if not position:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Position not found")

    if not payload.__fields_set__:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No updates provided")

    if 'quantity' in payload.__fields_set__:
        if payload.quantity is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Quantity must be greater than zero")
        position.quantity = payload.quantity
    if 'average_cost' in payload.__fields_set__:
        position.average_cost = payload.average_cost
    if 'purchase_date' in payload.__fields_set__:
        position.purchase_date = payload.purchase_date

    await db.commit()
    await db.refresh(position)
    return _build_position_response(position)


@router.delete("/positions/{position_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_position(
    position_id: int,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    result = await db.execute(
        select(PortfolioPosition).where(
            PortfolioPosition.id == position_id,
            PortfolioPosition.user_id == current_user.id
        )
    )
    position = result.scalar_one_or_none()
    if not position:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Position not found")

    await db.delete(position)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/import/brokers", response_model=List[BrokerProfileResponse])
async def list_import_brokers(
    current_user=Depends(get_current_user)
):
    try:
        brokers = list_brokers()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    return [
        BrokerProfileResponse(
            id=broker.id,
            name=broker.name,
            sheet=broker.sheet,
            top_left=broker.top_left,
            bottom_right=broker.bottom_right,
        )
        for broker in brokers
    ]


@router.post("/import", response_model=PortfolioImportResponse)
async def import_portfolio_positions(
    file: UploadFile | None = File(None),
    broker_id: str | None = Form(None),
    sheet: str | None = Form(None),
    top_left: str | None = Form(None),
    bottom_right: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    if not file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is required")
    if not broker_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Broker is required")

    broker = get_broker(broker_id)
    if broker is None:
        try:
            list_brokers()
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
    if not broker:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown broker")

    crop_settings = CropSettings(
        sheet=sheet or broker.sheet,
        top_left=top_left or broker.top_left,
        bottom_right=bottom_right or broker.bottom_right,
    )
    import_logger.info(
        "portfolio_import_start user_id=%s broker=%s filename=%s sheet=%s top_left=%s bottom_right=%s",
        current_user.id,
        broker.id,
        file.filename,
        crop_settings.sheet,
        crop_settings.top_left,
        crop_settings.bottom_right,
    )

    try:
        rows = await load_cropped_rows(file, crop_settings)
    except ValueError as exc:
        import_logger.error("portfolio_import_crop_error user_id=%s error=%s", current_user.id, exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not rows:
        import_logger.warning("portfolio_import_no_rows user_id=%s", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No data found in the cropped range",
        )
    import_logger.info(
        "portfolio_import_rows_loaded user_id=%s rows=%s cols=%s",
        current_user.id,
        len(rows),
        len(rows[0]) if rows else 0,
    )

    try:
        providers = settings.llm_providers_list
    except Exception as exc:
        import_logger.error("portfolio_import_llm_config_error user_id=%s error=%s", current_user.id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LLM providers configuration is invalid",
        ) from exc
    if not providers:
        import_logger.warning("portfolio_import_llm_missing user_id=%s", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LLM providers are not configured",
        )

    try:
        extracted_positions = await extract_positions_from_rows(
            rows,
            providers,
            settings.llm_request_timeout_seconds,
        )
    except Exception as exc:
        import_logger.error("portfolio_import_llm_error user_id=%s error=%s", current_user.id, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unable to extract positions: {exc}",
        ) from exc

    if not extracted_positions:
        import_logger.warning("portfolio_import_llm_empty user_id=%s", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="LLM did not return any positions",
        )
    import_logger.info(
        "portfolio_import_llm_positions user_id=%s positions=%s",
        current_user.id,
        len(extracted_positions),
    )

    tickers = [position.ticker.strip().upper() for position in extracted_positions]
    existing_result = await db.execute(
        select(PortfolioPosition).where(
            PortfolioPosition.user_id == current_user.id,
            PortfolioPosition.ticker.in_(tickers)
        )
    )
    existing_positions = {position.ticker.upper(): position for position in existing_result.scalars().all()}

    created_count = 0
    updated_count = 0
    skipped_count = 0
    imported_positions: List[PortfolioImportPosition] = []

    for position in extracted_positions:
        ticker = position.ticker.strip().upper()
        quantity = float(position.quantity)
        if not ticker or quantity <= 0:
            skipped_count += 1
            continue
        imported_positions.append(PortfolioImportPosition(ticker=ticker, quantity=quantity))

        existing = existing_positions.get(ticker)
        if existing:
            existing.quantity = quantity
            updated_count += 1
        else:
            db.add(PortfolioPosition(
                user_id=current_user.id,
                ticker=ticker,
                quantity=quantity,
                average_cost=None,
                purchase_date=None,
            ))
            created_count += 1

    await db.commit()

    refreshed = await db.execute(
        select(PortfolioPosition)
        .where(PortfolioPosition.user_id == current_user.id)
        .order_by(PortfolioPosition.created_at)
    )
    positions = refreshed.scalars().all()

    import_logger.info(
        "portfolio_import_complete user_id=%s created=%s updated=%s skipped=%s",
        current_user.id,
        created_count,
        updated_count,
        skipped_count,
    )

    return PortfolioImportResponse(
        imported_positions=imported_positions,
        created_count=created_count,
        updated_count=updated_count,
        skipped_count=skipped_count,
        positions=[_build_position_response(position) for position in positions],
    )
