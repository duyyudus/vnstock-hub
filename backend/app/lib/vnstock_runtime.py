from __future__ import annotations

import importlib
import sys
from typing import Dict

_TARGETS = {
    "vnstock": "app.lib.vnstock_alt",
    "vnstock_data": "app.lib.vnstock_data_alt",
}


def _purge_public_module(public_name: str) -> None:
    for module_name in list(sys.modules):
        if module_name == public_name or module_name.startswith(f"{public_name}."):
            sys.modules.pop(module_name, None)


def install_vnstock_aliases() -> Dict[str, str]:
    """
    Bind public vnstock import names to the vendored packages.

    This keeps backend service code importing `vnstock` and `vnstock_data`
    without depending on the upstream packages.
    """
    for public_name, target_name in _TARGETS.items():
        current = sys.modules.get(public_name)
        current_name = getattr(current, "__name__", None)
        if current_name == target_name:
            continue

        _purge_public_module(public_name)
        sys.modules[public_name] = importlib.import_module(target_name)

    return dict(_TARGETS)
