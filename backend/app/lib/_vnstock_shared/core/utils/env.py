from __future__ import annotations

import os
import platform
import sys
from importlib.util import find_spec
from pathlib import Path


def get_vnstock_directory() -> Path:
    """Return the local vnstock data directory."""
    return Path.home() / ".vnstock"


def is_colab() -> bool:
    """Legacy Colab compatibility shim."""
    return False


def setup_colab_drive(auto_mount: bool = True) -> bool:
    """Legacy Colab compatibility shim."""
    return False


def get_colab_install_command() -> str:
    """Legacy Colab compatibility shim."""
    return ""


def show_colab_instructions() -> None:
    """No-op replacement for legacy notebook guidance."""
    return None


def get_vnstock_path() -> Path:
    """Return the local vnstock data directory."""
    return get_vnstock_directory()


def get_platform():
    """Get the name of the running operating system."""
    return platform.system()


def get_hosting_service():
    """Identify the current hosting environment in a lightweight way."""
    if "CODESPACE_NAME" in os.environ:
        return "Github Codespace"
    if "GITPOD_WORKSPACE_CLUSTER_HOST" in os.environ:
        return "Gitpod"
    if "REPLIT_USER" in os.environ:
        return "Replit"
    if "KAGGLE_CONTAINER_NAME" in os.environ:
        return "Kaggle"
    if ".hf.space" in os.environ.get("SPACE_HOST", ""):
        return "Hugging Face Spaces"
    return "Local or Unknown"


def get_package_path(package="vnstock"):
    """Get the import path of a Python package when available."""
    spec = find_spec(package)
    if spec and spec.origin:
        return spec.origin
    if spec and spec.submodule_search_locations:
        return spec.submodule_search_locations[0]
    return None


def id_valid():
    """Compatibility no-op for upstream license checks."""
    return True


def get_username():
    """Get the current system username when available."""
    try:
        return os.getlogin()
    except OSError:
        return None


def get_cwd():
    """Return the current working directory when available."""
    try:
        return os.getcwd()
    except OSError:
        return None


def get_path_delimiter():
    """Return the platform file separator."""
    return "\\" if os.name == "nt" else "/"


def detect_venv() -> dict:
    """Detect the current virtual environment details."""
    venv_path = None
    is_active = False
    venv_type = "system"
    python_exe = sys.executable

    if "VIRTUAL_ENV" in os.environ:
        venv_path = os.environ["VIRTUAL_ENV"]
        is_active = True
        venv_type = "conda" if "conda" in venv_path.lower() else "venv"
        if os.name == "nt":
            python_exe = os.path.join(venv_path, "Scripts", "python.exe")
        else:
            python_exe = os.path.join(venv_path, "bin", "python")
    elif hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix:
        venv_path = sys.prefix
        is_active = True
        venv_type = "venv"
    elif "CONDA_PREFIX" in os.environ:
        venv_path = os.environ["CONDA_PREFIX"]
        is_active = True
        venv_type = "conda"
        if os.name == "nt":
            python_exe = os.path.join(venv_path, "python.exe")
        else:
            python_exe = os.path.join(venv_path, "bin", "python")

    return {
        "path": venv_path,
        "is_active": is_active,
        "type": venv_type,
        "python_exe": python_exe,
    }


def get_python_executable() -> str:
    """Get the Python executable for the current environment."""
    return detect_venv()["python_exe"]


def get_python_version_string() -> str:
    """Get the major.minor Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def is_venv_active() -> bool:
    """Return whether a virtual environment is active."""
    return detect_venv().get("is_active", False)


def get_venv_type() -> str:
    """Return the current virtual environment type."""
    return detect_venv().get("type", "system")
