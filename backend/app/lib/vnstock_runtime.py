from __future__ import annotations

import importlib
import sys
from typing import Dict

from app.core.config import settings


_UPSTREAM_TARGETS = {
    "vnstock": "vnstock",
    "vnstock_data": "vnstock_data",
}

_ALT_TARGETS = {
    "vnstock": "app.lib.vnstock_alt",
    "vnstock_data": "app.lib.vnstock_data_alt",
}


def using_vnstock_alt(use_alt: bool | None = None) -> bool:
    """Resolve whether runtime should use the vendored vnstock implementation."""
    if use_alt is None:
        return bool(settings.use_vnstock_alt)
    return bool(use_alt)


def using_vnstock_data_alt(use_alt: bool | None = None) -> bool:
    """Resolve whether runtime should use the vendored vnstock_data implementation."""
    if use_alt is None:
        return bool(settings.use_vnstock_data_alt)
    return bool(use_alt)


def runtime_vnstock_targets(
    use_vnstock_alt_flag: bool | None = None,
    use_vnstock_data_alt_flag: bool | None = None,
) -> Dict[str, str]:
    """Return the module targets bound to public vnstock import names."""
    return {
        "vnstock": _ALT_TARGETS["vnstock"] if using_vnstock_alt(use_vnstock_alt_flag) else _UPSTREAM_TARGETS["vnstock"],
        "vnstock_data": (
            _ALT_TARGETS["vnstock_data"]
            if using_vnstock_data_alt(use_vnstock_data_alt_flag)
            else _UPSTREAM_TARGETS["vnstock_data"]
        ),
    }


def _purge_public_module(public_name: str) -> None:
    for module_name in list(sys.modules):
        if module_name == public_name or module_name.startswith(f"{public_name}."):
            sys.modules.pop(module_name, None)


def install_vnstock_aliases(
    use_vnstock_alt_flag: bool | None = None,
    use_vnstock_data_alt_flag: bool | None = None,
) -> Dict[str, str]:
    """
    Bind public import names to either upstream or vendored packages.

    This keeps the rest of the backend code importing `vnstock` while allowing
    a runtime switch to `app.lib.vnstock_alt` and `app.lib.vnstock_data_alt`.
    """
    all_targets = runtime_vnstock_targets(
        use_vnstock_alt_flag=use_vnstock_alt_flag,
        use_vnstock_data_alt_flag=use_vnstock_data_alt_flag,
    )
    targets = {"vnstock": all_targets["vnstock"]}
    if using_vnstock_data_alt(use_vnstock_data_alt_flag):
        targets["vnstock_data"] = all_targets["vnstock_data"]

    for public_name, target_name in targets.items():
        current = sys.modules.get(public_name)
        current_name = getattr(current, "__name__", None)
        if current_name == target_name:
            continue

        _purge_public_module(public_name)
        sys.modules[public_name] = importlib.import_module(target_name)

    return targets
