"""Compatibility helpers replacing vnai-dependent behavior."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional


def _identity_decorator(func: Optional[Callable[..., Any]] = None, *_args: Any, **_kwargs: Any):
    """Return a no-op decorator that preserves the wrapped callable."""

    def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(inner)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return inner(*args, **kwargs)

        return wrapped

    if func is not None and callable(func):
        return decorator(func)
    return decorator


def agg_execution(*args: Any, **kwargs: Any):
    return _identity_decorator(*args, **kwargs)


def optimize_execution(*args: Any, **kwargs: Any):
    return _identity_decorator(*args, **kwargs)


def setup(*_args: Any, **_kwargs: Any) -> None:
    """No-op replacement for vnai.setup()."""


def setup_api_key(*_args: Any, **_kwargs: Any) -> bool:
    """API-key flows are intentionally unsupported in the alt packages."""
    return False


def check_api_key_status(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Return a stable guest-like status without enabling auth flows."""
    return {
        "has_api_key": False,
        "api_key_preview": None,
        "tier": "guest",
        "limits": {},
    }


def accept_license_terms(*_args: Any, **_kwargs: Any) -> bool:
    return True


class _Inspector:
    @staticmethod
    def fingerprint() -> str:
        return "vnstock-alt"


inspector = _Inspector()
