"""Portfolio position endpoints for authenticated users."""
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.database import get_db
from app.db.models import PortfolioPosition

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


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
