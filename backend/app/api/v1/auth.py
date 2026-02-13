"""
Authentication endpoints (register/login) using JWT.
"""
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import get_current_user
from app.core.security import create_access_token
from app.db.database import get_db
from app.services.auth_service import create_user, authenticate_user, get_user_by_email

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    download_folder: str | None = None
    company_export_category: str | None = None
    finance_export_category: str | None = None
    is_active: bool
    created_at: str
    last_login: str | None = None


class UserSettingsResponse(BaseModel):
    download_folder: str | None = None
    company_export_category: str | None = None
    finance_export_category: str | None = None


class UserSettingsUpdateRequest(BaseModel):
    download_folder: str | None = Field(default=None, max_length=512)
    company_export_category: str | None = Field(default=None, max_length=120)
    finance_export_category: str | None = Field(default=None, max_length=120)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


def _normalize_user_setting_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _build_auth_response(user) -> AuthResponse:
    expires = timedelta(minutes=settings.access_token_expire_minutes)
    token = create_access_token(
        subject=str(user.id),
        expires_delta=expires,
        additional_claims={"email": user.email}
    )
    return AuthResponse(
        access_token=token,
        expires_in=int(expires.total_seconds()),
        user=UserResponse(
            id=user.id,
            email=user.email,
            download_folder=user.download_folder,
            company_export_category=user.company_export_category,
            finance_export_category=user.finance_export_category,
            is_active=user.is_active,
            created_at=user.created_at.isoformat(),
            last_login=user.last_login.isoformat() if user.last_login else None
        )
    )


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await get_user_by_email(db, payload.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is already registered"
        )
    user = await create_user(db, payload.email, payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to register with that email"
        )
    return _build_auth_response(user)


@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, payload.email, payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _build_auth_response(user)


@router.get("/settings", response_model=UserSettingsResponse)
async def get_user_settings(current_user=Depends(get_current_user)):
    return UserSettingsResponse(
        download_folder=current_user.download_folder,
        company_export_category=current_user.company_export_category,
        finance_export_category=current_user.finance_export_category,
    )


@router.patch("/settings", response_model=UserSettingsResponse)
async def update_user_settings(
    payload: UserSettingsUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    payload_data = payload.model_dump(exclude_unset=True)

    if "download_folder" in payload_data:
        current_user.download_folder = _normalize_user_setting_value(payload.download_folder)
    if "company_export_category" in payload_data:
        current_user.company_export_category = _normalize_user_setting_value(payload.company_export_category)
    if "finance_export_category" in payload_data:
        current_user.finance_export_category = _normalize_user_setting_value(payload.finance_export_category)

    await db.commit()
    await db.refresh(current_user)
    return UserSettingsResponse(
        download_folder=current_user.download_folder,
        company_export_category=current_user.company_export_category,
        finance_export_category=current_user.finance_export_category,
    )
