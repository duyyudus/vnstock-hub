"""Portfolio position endpoints for authenticated users."""
import csv
import io
import math
import re
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.core.config import settings
from app.core.logging_config import get_portfolio_import_logger
from app.db.database import get_db
from app.db.models import PortfolioPosition
from app.services.llm import (
    POSITION_IMAGE_EXTRACTION_TASK,
    POSITION_TABLE_EXTRACTION_TASK,
)
from app.services.portfolio_import import (
    CropSettings,
    extract_positions_from_image,
    extract_positions_from_rows,
    get_broker,
    is_image_file,
    list_brokers,
    load_cropped_rows,
    merge_image_positions,
)

router = APIRouter(prefix="/portfolio", tags=["portfolio"])
import_logger = get_portfolio_import_logger()
PORTFOLIO_CSV_FIELDS = ["ticker", "quantity", "average_cost", "purchase_date"]


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
    average_cost_multiplier: float


class PortfolioImportPosition(BaseModel):
    ticker: str
    quantity: Optional[float] = None
    average_cost: Optional[float] = None


class PortfolioImportResponse(BaseModel):
    imported_positions: List[PortfolioImportPosition]
    created_count: int
    updated_count: int
    skipped_count: int
    positions: List[PortfolioPositionResponse]


class PortfolioFreshImportResponse(BaseModel):
    created_count: int
    deleted_count: int
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


def _is_finite_positive_number(value: float) -> bool:
    return math.isfinite(value) and value > 0


def _validate_csv_upload(upload: UploadFile) -> None:
    allowed_content_types = {
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",
        "text/plain",
        "application/octet-stream",
    }
    filename = upload.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be a CSV")

    if upload.content_type and upload.content_type.lower() not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content type must be CSV",
        )


def _build_export_filename(user_email: str | None, user_id: int | None) -> str:
    if user_email:
        user_part = user_email.split("@", 1)[0].strip()
    else:
        user_part = ""
    sanitized_user = re.sub(r"[^A-Za-z0-9._-]+", "_", user_part).strip("._-")
    if not sanitized_user:
        fallback_id = user_id if user_id is not None else "user"
        sanitized_user = f"user_{fallback_id}"
    return f"{sanitized_user}_{date.today().isoformat()}.csv"


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


@router.get("/export/csv")
async def export_positions_csv(
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    result = await db.execute(
        select(PortfolioPosition)
        .where(PortfolioPosition.user_id == current_user.id)
        .order_by(PortfolioPosition.created_at)
    )
    positions = result.scalars().all()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(PORTFOLIO_CSV_FIELDS)
    for position in positions:
        writer.writerow([
            position.ticker,
            position.quantity,
            position.average_cost if position.average_cost is not None else "",
            position.purchase_date.isoformat() if position.purchase_date else "",
        ])

    export_filename = _build_export_filename(getattr(current_user, "email", None), getattr(current_user, "id", None))

    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{export_filename}"'},
    )


@router.post("/import/fresh", response_model=PortfolioFreshImportResponse)
async def fresh_import_positions_csv(
    file: UploadFile | None = File(None),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    if file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is required")

    _validate_csv_upload(file)
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSV file is empty")

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSV must be UTF-8 encoded") from exc

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != PORTFOLIO_CSV_FIELDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"CSV header must be exactly: {','.join(PORTFOLIO_CSV_FIELDS)}",
        )

    parsed_rows: List[dict] = []
    seen_tickers: set[str] = set()

    for line_number, row in enumerate(reader, start=2):
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid CSV row at line {line_number}",
            )

        ticker = _normalize_ticker(row.get("ticker") or "")
        if not ticker:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ticker is required at line {line_number}",
            )
        if len(ticker) > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ticker must be at most 10 characters at line {line_number}",
            )
        if ticker in seen_tickers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Duplicate ticker '{ticker}' at line {line_number}",
            )

        quantity_raw = (row.get("quantity") or "").strip()
        try:
            quantity = float(quantity_raw)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Quantity must be a number at line {line_number}",
            ) from exc
        if not _is_finite_positive_number(quantity):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Quantity must be greater than zero at line {line_number}",
            )

        average_cost_raw = (row.get("average_cost") or "").strip()
        average_cost: float | None = None
        if average_cost_raw:
            try:
                average_cost = float(average_cost_raw)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Average cost must be a number at line {line_number}",
                ) from exc
            if not _is_finite_positive_number(average_cost):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Average cost must be greater than zero at line {line_number}",
                )

        purchase_date_raw = (row.get("purchase_date") or "").strip()
        purchase_date: date | None = None
        if purchase_date_raw:
            try:
                purchase_date = date.fromisoformat(purchase_date_raw)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Purchase date must be ISO YYYY-MM-DD at line {line_number}",
                ) from exc

        parsed_rows.append({
            "ticker": ticker,
            "quantity": quantity,
            "average_cost": average_cost,
            "purchase_date": purchase_date,
        })
        seen_tickers.add(ticker)

    if not parsed_rows:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV contains no position rows",
        )

    existing_result = await db.execute(
        select(PortfolioPosition).where(PortfolioPosition.user_id == current_user.id)
    )
    existing_positions = existing_result.scalars().all()
    deleted_count = len(existing_positions)

    await db.execute(
        delete(PortfolioPosition).where(PortfolioPosition.user_id == current_user.id)
    )
    for row in parsed_rows:
        db.add(PortfolioPosition(
            user_id=current_user.id,
            ticker=row["ticker"],
            quantity=row["quantity"],
            average_cost=row["average_cost"],
            purchase_date=row["purchase_date"],
        ))
    await db.commit()

    refreshed = await db.execute(
        select(PortfolioPosition)
        .where(PortfolioPosition.user_id == current_user.id)
        .order_by(PortfolioPosition.created_at)
    )
    positions = refreshed.scalars().all()
    return PortfolioFreshImportResponse(
        created_count=len(parsed_rows),
        deleted_count=deleted_count,
        positions=[_build_position_response(position) for position in positions],
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
            average_cost_multiplier=broker.average_cost_multiplier,
        )
        for broker in brokers
    ]


@router.post("/import", response_model=PortfolioImportResponse)
async def import_portfolio_positions(
    file: List[UploadFile] | None = File(None),
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

    files = file
    if len(files) > 1:
        non_images = [
            upload for upload in files
            if not is_image_file(upload.filename, upload.content_type)
        ]
        if non_images:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Multiple files are only supported for image uploads",
            )
    is_image = is_image_file(files[0].filename, files[0].content_type)
    crop_settings = CropSettings(
        sheet=sheet or broker.sheet,
        top_left=top_left or broker.top_left,
        bottom_right=bottom_right or broker.bottom_right,
    )
    import_logger.info(
        "portfolio_import_start user_id=%s broker=%s files=%s sheet=%s top_left=%s bottom_right=%s",
        current_user.id,
        broker.id,
        [upload.filename for upload in files],
        crop_settings.sheet,
        crop_settings.top_left,
        crop_settings.bottom_right,
    )

    created_count = 0
    updated_count = 0
    skipped_count = 0
    imported_positions: List[PortfolioImportPosition] = []

    if is_image:
        try:
            providers = settings.resolve_llm_providers(POSITION_IMAGE_EXTRACTION_TASK)
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
                    "portfolio_import_llm_error user_id=%s error=%s filename=%s",
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
                    "portfolio_import_llm_empty user_id=%s filename=%s",
                    current_user.id,
                    upload.filename,
                )
                continue
            aggregated_positions.extend(extracted_positions)

        if not aggregated_positions:
            import_logger.warning("portfolio_import_llm_empty user_id=%s", current_user.id)
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

        if not merged_positions:
            import_logger.warning("portfolio_import_llm_empty user_id=%s", current_user.id)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="LLM did not return any positions",
            )

        import_logger.info(
            "portfolio_import_llm_positions user_id=%s positions=%s mode=image files=%s",
            current_user.id,
            len(merged_positions),
            len(files),
        )

        tickers = [position.ticker.strip().upper() for position in merged_positions]
        existing_result = await db.execute(
            select(PortfolioPosition).where(
                PortfolioPosition.user_id == current_user.id,
                PortfolioPosition.ticker.in_(tickers)
            )
        )
        existing_positions = {position.ticker.upper(): position for position in existing_result.scalars().all()}

        for position in merged_positions:
            ticker = position.ticker.strip().upper()
            average_cost = float(position.average_cost)
            quantity = float(position.quantity) if position.quantity is not None else None
            if not ticker or average_cost <= 0:
                skipped_count += 1
                continue

            imported_positions.append(
                PortfolioImportPosition(
                    ticker=ticker,
                    quantity=quantity,
                    average_cost=average_cost,
                )
            )

            existing = existing_positions.get(ticker)
            if existing:
                existing.average_cost = average_cost
                if quantity is not None and quantity > 0:
                    existing.quantity = quantity
                updated_count += 1
            else:
                if quantity is None or quantity <= 0:
                    skipped_count += 1
                    continue
                db.add(PortfolioPosition(
                    user_id=current_user.id,
                    ticker=ticker,
                    quantity=quantity,
                    average_cost=average_cost,
                    purchase_date=None,
                ))
                created_count += 1
    else:
        try:
            rows = await load_cropped_rows(files[0], crop_settings)
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
            providers = settings.resolve_llm_providers(POSITION_TABLE_EXTRACTION_TASK)
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
