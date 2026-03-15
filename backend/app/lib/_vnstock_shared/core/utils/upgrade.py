"""
No-op replacement for upstream upgrade and notebook helpers.

The vendored vnstock packages intentionally avoid package upgrade notices,
notebook-specific display logic, and PyPI version checks.
"""

from __future__ import annotations


def detect_environment() -> str:
    """Return a simple environment label for compatibility."""
    return "Terminal"


def update_notice(verbose: bool = False) -> None:
    """Compatibility no-op."""
    return None


def show_full_notice() -> None:
    """Compatibility no-op."""
    return None
