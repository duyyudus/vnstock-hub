from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from app.core.config import settings
from app.lib.vnstock_runtime import install_vnstock_aliases, runtime_vnstock_targets


def _purge_public_modules() -> None:
    for prefix in ("vnstock", "vnstock_data"):
        for module_name in list(sys.modules):
            if module_name == prefix or module_name.startswith(f"{prefix}."):
                sys.modules.pop(module_name, None)


@pytest.fixture(autouse=True)
def _prepare_runtime_imports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _purge_public_modules()

    fake_home = tmp_path / "fake-home"
    (fake_home / ".vnstock").mkdir(parents=True, exist_ok=True)
    (fake_home / ".vnstock" / "user.json").write_text('{"user": true}')
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))

    stub = types.ModuleType("vnii")
    stub.lc_init = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "vnii", stub)

    yield

    _purge_public_modules()


def test_install_vnstock_aliases_uses_upstream_when_disabled() -> None:
    targets = install_vnstock_aliases(
        use_vnstock_alt_flag=False,
        use_vnstock_data_alt_flag=False,
    )

    assert targets == {"vnstock": "vnstock"}
    assert sys.modules["vnstock"].__name__ == "vnstock"
    assert "vnstock_data" not in sys.modules


def test_install_vnstock_aliases_uses_vnstock_alt_only_when_enabled() -> None:
    targets = install_vnstock_aliases(
        use_vnstock_alt_flag=True,
        use_vnstock_data_alt_flag=False,
    )

    assert targets == {"vnstock": "app.lib.vnstock_alt"}
    assert sys.modules["vnstock"].__name__ == "app.lib.vnstock_alt"
    assert "vnstock_data" not in sys.modules


def test_install_vnstock_aliases_uses_vnstock_data_alt_only_when_enabled() -> None:
    targets = install_vnstock_aliases(
        use_vnstock_alt_flag=False,
        use_vnstock_data_alt_flag=True,
    )

    assert targets == {
        "vnstock": "vnstock",
        "vnstock_data": "app.lib.vnstock_data_alt",
    }
    assert sys.modules["vnstock"].__name__ == "vnstock"
    assert sys.modules["vnstock_data"].__name__ == "app.lib.vnstock_data_alt"


def test_install_vnstock_aliases_uses_both_vendored_packages_when_enabled() -> None:
    targets = install_vnstock_aliases(
        use_vnstock_alt_flag=True,
        use_vnstock_data_alt_flag=True,
    )

    assert targets == {
        "vnstock": "app.lib.vnstock_alt",
        "vnstock_data": "app.lib.vnstock_data_alt",
    }
    assert sys.modules["vnstock"].__name__ == "app.lib.vnstock_alt"
    assert sys.modules["vnstock_data"].__name__ == "app.lib.vnstock_data_alt"


def test_runtime_vnstock_targets_follow_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "use_vnstock_alt", True)
    monkeypatch.setattr(settings, "use_vnstock_data_alt", False)
    assert runtime_vnstock_targets() == {
        "vnstock": "app.lib.vnstock_alt",
        "vnstock_data": "vnstock_data",
    }

    monkeypatch.setattr(settings, "use_vnstock_alt", False)
    monkeypatch.setattr(settings, "use_vnstock_data_alt", True)
    assert runtime_vnstock_targets() == {
        "vnstock": "vnstock",
        "vnstock_data": "app.lib.vnstock_data_alt",
    }

    monkeypatch.setattr(settings, "use_vnstock_alt", False)
    monkeypatch.setattr(settings, "use_vnstock_data_alt", False)
    assert runtime_vnstock_targets() == {
        "vnstock": "vnstock",
        "vnstock_data": "vnstock_data",
    }
