"""
Global exception handlers for VNStock Hub backend.

Catches exceptions and returns user-friendly error responses
while logging appropriately (RateLimitExceeded as warning, others as error).
"""
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging_config import get_main_logger
from app.core.circuit_breaker import CircuitOpenError

logger = get_main_logger()


# Known rate limit exception class name from vnai library
RATE_LIMIT_EXCEPTION_NAMES = {"RateLimitExceeded", "RateLimitError"}


def is_rate_limit_exception(exc: BaseException) -> bool:
    """Check if exception is a rate limit error from vnai/vnstock."""
    # Check by class name
    if type(exc).__name__ in RATE_LIMIT_EXCEPTION_NAMES:
        return True
    # Check for CircuitOpenError (indicates rate limit)
    if isinstance(exc, CircuitOpenError):
        return True
    # vnai library calls sys.exit() with rate limit message
    if isinstance(exc, SystemExit):
        msg = str(exc).lower()
        if "rate limit" in msg:
            return True
    return False


class SystemExitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to catch SystemExit exceptions.
    vnai library calls sys.exit() on rate limit - this catches it before
    it crashes the ASGI application.
    """

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except SystemExit as exc:
            if is_rate_limit_exception(exc):
                logger.warning(f"Rate limit exceeded: {request.method} {request.url.path}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded. Please try again later.",
                        "error_type": "rate_limit",
                        "retry_after": 30
                    }
                )
            # For other SystemExit, log and return 500
            logger.error(f"SystemExit: {request.method} {request.url.path} - {exc}")
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "An internal error occurred. Please try again later.",
                    "error_type": "internal_error"
                }
            )


async def rate_limit_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle rate limit exceptions.
    Returns 429 with retry information.
    """
    logger.warning(f"Rate limit: {request.method} {request.url.path} - {type(exc).__name__}")
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Please try again later.",
            "error_type": "rate_limit",
            "retry_after": 30
        }
    )


async def circuit_breaker_exception_handler(request: Request, exc: CircuitOpenError) -> JSONResponse:
    """
    Handle circuit breaker open errors.
    Returns 503 Service Unavailable with retry information.
    """
    logger.warning(f"Circuit breaker open: {request.method} {request.url.path}")
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Service temporarily unavailable due to rate limiting. Please try again shortly.",
            "error_type": "circuit_breaker_open",
            "retry_after": 30
        }
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle all other unhandled exceptions.
    Logs as error and returns 500 response.
    """
    # Check if it's actually a rate limit exception (duck typing check)
    if is_rate_limit_exception(exc):
        return await rate_limit_exception_handler(request, exc)

    logger.error(f"Unhandled exception: {request.method} {request.url.path} - {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred. Please try again later.",
            "error_type": "internal_error"
        }
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle HTTP exceptions (404, 403, etc.).
    These are expected and logged at debug level.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


def register_exception_handlers(app):
    """Register all exception handlers with the FastAPI app."""
    from starlette.exceptions import HTTPException as StarletteHTTPException

    # Handle HTTP exceptions (404, etc.)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    # Handle circuit breaker open errors specifically
    app.add_exception_handler(CircuitOpenError, circuit_breaker_exception_handler)

    # Handle all other exceptions
    app.add_exception_handler(Exception, generic_exception_handler)

    # Add middleware for SystemExit (BaseException, can't use add_exception_handler)
    app.add_middleware(SystemExitMiddleware)
