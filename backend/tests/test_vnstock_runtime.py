from __future__ import annotations

import sys
import types

import pytest

from app.lib.vnstock_runtime import install_vnstock_aliases


EXPECTED_TARGETS = {
    "vnstock": "app.lib.vnstock_alt",
    "vnstock_data": "app.lib.vnstock_data_alt",
}


def _purge_public_modules() -> None:
    for prefix in ("vnstock", "vnstock_data"):
        for module_name in list(sys.modules):
            if module_name == prefix or module_name.startswith(f"{prefix}."):
                sys.modules.pop(module_name, None)


@pytest.fixture(autouse=True)
def _prepare_runtime_imports() -> None:
    _purge_public_modules()

    yield

    _purge_public_modules()


def test_install_vnstock_aliases_always_uses_vendored_packages() -> None:
    targets = install_vnstock_aliases()

    assert targets == EXPECTED_TARGETS
    assert sys.modules["vnstock"].__name__ == "app.lib.vnstock_alt"
    assert sys.modules["vnstock_data"].__name__ == "app.lib.vnstock_data_alt"


def test_install_vnstock_aliases_replaces_preloaded_public_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "vnstock", types.ModuleType("vnstock"))
    monkeypatch.setitem(sys.modules, "vnstock.core", types.ModuleType("vnstock.core"))
    monkeypatch.setitem(sys.modules, "vnstock_data", types.ModuleType("vnstock_data"))
    monkeypatch.setitem(sys.modules, "vnstock_data.ui", types.ModuleType("vnstock_data.ui"))

    targets = install_vnstock_aliases()

    assert targets == EXPECTED_TARGETS
    assert sys.modules["vnstock"].__name__ == "app.lib.vnstock_alt"
    assert sys.modules["vnstock_data"].__name__ == "app.lib.vnstock_data_alt"
    assert "vnstock.core" not in sys.modules
    assert "vnstock_data.ui" not in sys.modules
