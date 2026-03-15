"""
Local-only replacement for legacy Google Colab helpers.

The vendored vnstock packages do not support Colab-specific installation,
Drive mounting, or migration behavior. This module keeps the old entry points
available as low-risk local-path shims.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def is_google_colab() -> bool:
    """Return whether the current process runs on Google Colab."""
    return False


def is_drive_mounted() -> bool:
    """Legacy Colab compatibility shim."""
    return False


def mount_drive(force_remount: bool = False) -> bool:
    """Legacy Colab compatibility shim."""
    return False


def initialize_colab_environment() -> bool:
    """Legacy Colab compatibility shim."""
    return False


def get_vnstock_directory() -> Path:
    """Return the local vnstock data directory."""
    return Path.home() / ".vnstock"


def get_vnstock_data_dir() -> Path:
    """Return the vnstock data directory, honoring an explicit override."""
    data_dir = os.environ.get("VNSTOCK_DATA_DIR")
    if data_dir:
        return Path(data_dir).expanduser().resolve()
    return get_vnstock_directory()


def get_install_target() -> Optional[str]:
    """Legacy Colab compatibility shim."""
    return None


def show_setup_guide() -> None:
    """No-op replacement for legacy notebook guidance."""
    return None


def get_install_command() -> str:
    """Legacy Colab compatibility shim."""
    return ""


def migrate_vnstock_data_colab(new_dir: Optional[str] = None) -> bool:
    """Legacy Colab compatibility shim."""
    return False


def setup_colab_drive(auto_mount: bool = True) -> bool:
    """Legacy Colab compatibility shim."""
    return False
