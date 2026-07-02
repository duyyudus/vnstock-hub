"""
FastAPI application entry point.
"""
from contextlib import asynccontextmanager
from importlib.metadata import version as pkg_version, PackageNotFoundError
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging_config import setup_logging, get_main_logger
from app.core.exceptions import register_exception_handlers
from app.lib.vnstock_runtime import install_vnstock_aliases, runtime_vnstock_targets

# Initialize logging before anything else
setup_logging()
logger = get_main_logger()
install_vnstock_aliases()
runtime_targets = runtime_vnstock_targets()
logger.info(
    "Using vnstock runtime bindings: vnstock=%s, vnstock_data=%s",
    runtime_targets["vnstock"],
    runtime_targets["vnstock_data"],
)
from app.api.v1.stocks import router as stocks_router
from app.api.v1.funds import router as funds_router
from app.api.v1.sync import router as sync_router
from app.api.v1.auth import router as auth_router
from app.api.v1.bookmarks import router as bookmarks_router
from app.api.v1.portfolio import router as portfolio_router
from app.api.v1.trading import router as trading_router
from app.api.v1.info import router as info_router
from app.db.database import engine, Base
import app.db.models  # Ensure models are registered
from app.services.vnstock_service import vnstock_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Sync indices from vnstock
    try:
        await vnstock_service.sync_indices()
    except Exception as e:
        logger.error(f"Error syncing indices on startup: {e}")

    try:
        await vnstock_service.start_background_tasks()
    except Exception as e:
        logger.error(f"Error starting background workers: {e}")

    yield

    try:
        await vnstock_service.stop_background_tasks()
    except Exception as e:
        logger.error(f"Error stopping background workers: {e}")

# Create FastAPI app
try:
    _api_version = pkg_version("vnstock-hub-backend")
except PackageNotFoundError:
    _api_version = "unknown"

app = FastAPI(
    title="VNStock Hub API",
    description="API for Vietnam stock market data and trading",
    version=_api_version,
    lifespan=lifespan
)

# Register global exception handlers
register_exception_handlers(app)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# Include routers
app.include_router(stocks_router, prefix=settings.api_v1_prefix)
app.include_router(funds_router, prefix=settings.api_v1_prefix)
app.include_router(sync_router, prefix=settings.api_v1_prefix)
app.include_router(auth_router, prefix=settings.api_v1_prefix)
app.include_router(bookmarks_router, prefix=settings.api_v1_prefix)
app.include_router(portfolio_router, prefix=settings.api_v1_prefix)
app.include_router(trading_router, prefix=settings.api_v1_prefix)
app.include_router(info_router, prefix=settings.api_v1_prefix)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "VNStock Hub API", "version": _api_version}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
