"""
Charting support is intentionally excluded from the vendored vnstock packages.

The original upstream packages expose optional visualization helpers that depend
on external charting libraries. Those helpers are out of scope for this
reimplementation, so the module remains as a compatibility stub only.
"""

from __future__ import annotations

from typing import Any


_ERROR_MESSAGE = (
    "Charting helpers are intentionally not bundled in the vendored "
    "vnstock compatibility packages. Use a dedicated visualization layer "
    "instead of app.lib._vnstock_shared.common.viz."
)

HAS_VNSTOCK_CHART = False
HAS_VNSTOCK_EZCHART = False


class Chart:
    """Compatibility stub for the upstream chart wrapper."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise ImportError(_ERROR_MESSAGE)


def get_chart(*_args: Any, **_kwargs: Any) -> Chart:
    """Compatibility stub for the upstream helper."""

    raise ImportError(_ERROR_MESSAGE)


__all__ = ["Chart", "get_chart", "HAS_VNSTOCK_CHART", "HAS_VNSTOCK_EZCHART"]
