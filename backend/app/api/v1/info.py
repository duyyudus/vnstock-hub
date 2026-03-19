"""App info endpoint — returns version and build number."""
import subprocess
import tomllib
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter(prefix="/info", tags=["info"])


def _get_build_number() -> str:
    if settings.build_number:
        return settings.build_number
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _get_backend_version() -> str:
    try:
        return version("vnstock-hub-backend")
    except PackageNotFoundError:
        pass
    # Fallback: read directly from pyproject.toml (reliable in dev)
    try:
        toml_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
        with open(toml_path, "rb") as f:
            return tomllib.load(f)["project"]["version"]
    except Exception:
        return "unknown"


class AppInfoResponse(BaseModel):
    backend_version: str
    build_number: str


@router.get("", response_model=AppInfoResponse)
async def get_app_info():
    """Return backend version and build number."""
    return AppInfoResponse(
        backend_version=_get_backend_version(),
        build_number=_get_build_number(),
    )
