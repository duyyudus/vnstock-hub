"""Bookmark group endpoints for user favorite stocks."""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.db.database import get_db
from app.db.models import BookmarkGroup, BookmarkStock
from app.services.vnstock_service import vnstock_service
from app.api.v1.stocks import StockResponse

router = APIRouter(prefix="/bookmarks", tags=["bookmarks"])


class BookmarkGroupResponse(BaseModel):
    id: int
    name: str
    tickers: List[str]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class BookmarkGroupsResponse(BaseModel):
    groups: List[BookmarkGroupResponse]
    count: int


class BookmarkGroupCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


class BookmarkGroupUpdateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


class BookmarkStockRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)


class BookmarkGroupStocksResponse(BaseModel):
    stocks: List[StockResponse]
    count: int
    group_id: int
    group_name: str


def _normalize_group_name(name: str) -> str:
    return name.strip()


def _normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


async def _get_group_or_404(db: AsyncSession, user_id: int, group_id: int) -> BookmarkGroup:
    result = await db.execute(
        select(BookmarkGroup).where(
            BookmarkGroup.id == group_id,
            BookmarkGroup.user_id == user_id
        )
    )
    group = result.scalar_one_or_none()
    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bookmark group not found")
    return group


async def _build_group_response(db: AsyncSession, group: BookmarkGroup) -> BookmarkGroupResponse:
    result = await db.execute(
        select(BookmarkStock.ticker).where(BookmarkStock.group_id == group.id)
    )
    tickers = sorted([row[0] for row in result.all()])
    return BookmarkGroupResponse(
        id=group.id,
        name=group.name,
        tickers=tickers,
        created_at=group.created_at.isoformat() if group.created_at else None,
        updated_at=group.updated_at.isoformat() if group.updated_at else None
    )


@router.get("/groups", response_model=BookmarkGroupsResponse)
async def list_bookmark_groups(
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    result = await db.execute(
        select(BookmarkGroup)
        .where(BookmarkGroup.user_id == current_user.id)
        .order_by(BookmarkGroup.created_at)
    )
    groups = result.scalars().all()
    if not groups:
        return BookmarkGroupsResponse(groups=[], count=0)

    group_ids = [group.id for group in groups]
    tickers_map = {group_id: [] for group_id in group_ids}
    ticker_rows = await db.execute(
        select(BookmarkStock.group_id, BookmarkStock.ticker)
        .where(BookmarkStock.group_id.in_(group_ids))
    )
    for group_id, ticker in ticker_rows.all():
        tickers_map[group_id].append(ticker)

    response_groups = [
        BookmarkGroupResponse(
            id=group.id,
            name=group.name,
            tickers=sorted(tickers_map.get(group.id, [])),
            created_at=group.created_at.isoformat() if group.created_at else None,
            updated_at=group.updated_at.isoformat() if group.updated_at else None
        )
        for group in groups
    ]
    return BookmarkGroupsResponse(groups=response_groups, count=len(response_groups))


@router.post("/groups", response_model=BookmarkGroupResponse, status_code=status.HTTP_201_CREATED)
async def create_bookmark_group(
    payload: BookmarkGroupCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    name = _normalize_group_name(payload.name)
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Group name is required")

    result = await db.execute(
        select(BookmarkGroup).where(
            BookmarkGroup.user_id == current_user.id,
            func.lower(BookmarkGroup.name) == name.lower()
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Bookmark group already exists")

    group = BookmarkGroup(user_id=current_user.id, name=name)
    db.add(group)
    await db.commit()
    await db.refresh(group)
    return await _build_group_response(db, group)


@router.post("/groups/{group_id}/stocks", response_model=BookmarkGroupResponse)
async def add_stock_to_group(
    group_id: int,
    payload: BookmarkStockRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    group = await _get_group_or_404(db, current_user.id, group_id)
    ticker = _normalize_ticker(payload.ticker)
    if not ticker:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ticker is required")

    result = await db.execute(
        select(BookmarkStock).where(
            BookmarkStock.group_id == group.id,
            BookmarkStock.ticker == ticker
        )
    )
    existing = result.scalar_one_or_none()
    if not existing:
        db.add(BookmarkStock(group_id=group.id, ticker=ticker))
        await db.commit()
    return await _build_group_response(db, group)


@router.delete("/groups/{group_id}/stocks/{ticker}", response_model=BookmarkGroupResponse)
async def remove_stock_from_group(
    group_id: int,
    ticker: str,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    group = await _get_group_or_404(db, current_user.id, group_id)
    normalized_ticker = _normalize_ticker(ticker)

    result = await db.execute(
        select(BookmarkStock).where(
            BookmarkStock.group_id == group.id,
            BookmarkStock.ticker == normalized_ticker
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        await db.delete(existing)
        await db.commit()
    return await _build_group_response(db, group)


@router.patch("/groups/{group_id}", response_model=BookmarkGroupResponse)
async def rename_bookmark_group(
    group_id: int,
    payload: BookmarkGroupUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    group = await _get_group_or_404(db, current_user.id, group_id)
    name = _normalize_group_name(payload.name)
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Group name is required")

    result = await db.execute(
        select(BookmarkGroup).where(
            BookmarkGroup.user_id == current_user.id,
            func.lower(BookmarkGroup.name) == name.lower(),
            BookmarkGroup.id != group.id
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Bookmark group already exists")

    group.name = name
    await db.commit()
    await db.refresh(group)
    return await _build_group_response(db, group)


@router.delete("/groups/{group_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_bookmark_group(
    group_id: int,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    group = await _get_group_or_404(db, current_user.id, group_id)
    await db.delete(group)
    await db.commit()


@router.get("/groups/{group_id}/stocks", response_model=BookmarkGroupStocksResponse)
async def get_group_stocks(
    group_id: int,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    group = await _get_group_or_404(db, current_user.id, group_id)

    result = await db.execute(
        select(BookmarkStock.ticker).where(BookmarkStock.group_id == group.id)
    )
    tickers = [row[0] for row in result.all()]
    if not tickers:
        return BookmarkGroupStocksResponse(
            stocks=[],
            count=0,
            group_id=group.id,
            group_name=group.name
        )

    stocks = await vnstock_service.get_symbol_stocks(tickers)
    return BookmarkGroupStocksResponse(
        stocks=[
            StockResponse(
                ticker=stock.ticker,
                price=stock.price,
                market_cap=stock.market_cap,
                company_name=stock.company_name,
                exchange=stock.exchange,
                charter_capital=stock.charter_capital,
                pe_ratio=stock.pe_ratio,
                accumulated_value=stock.accumulated_value,
                price_change_24h=stock.price_change_24h,
                price_change_1w=stock.price_change_1w,
                price_change_1m=stock.price_change_1m,
                price_change_6m=stock.price_change_6m,
                price_change_1y=stock.price_change_1y,
                price_change_2y=stock.price_change_2y,
                price_change_3y=stock.price_change_3y
            )
            for stock in stocks
        ],
        count=len(stocks),
        group_id=group.id,
        group_name=group.name
    )
