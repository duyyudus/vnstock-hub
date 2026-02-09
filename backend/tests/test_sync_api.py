from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core.deps import get_current_admin_user
from app.main import app
from app.services.vnstock_service import vnstock_service


@pytest.mark.asyncio
async def test_get_sync_status_includes_price_sync(client):
    response = await client.get("/api/v1/sync/status")
    assert response.status_code == 200

    payload = response.json()
    assert "fund_performance" in payload
    assert "price_sync" in payload
    assert "bootstrap" in payload["price_sync"]
    assert "incremental" in payload["price_sync"]
    assert "repair" in payload["price_sync"]


@pytest.mark.asyncio
async def test_start_price_bootstrap_requires_admin_auth(client):
    response = await client.post("/api/v1/sync/prices/bootstrap/start", json={"force_restart": False})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_start_price_bootstrap_with_admin_calls_service(client):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        with patch.object(
            vnstock_service,
            "start_price_bootstrap",
            AsyncMock(return_value={"started": True, "message": "Bootstrap started", "state": "running"}),
        ) as mock_start:
            response = await client.post("/api/v1/sync/prices/bootstrap/start", json={"force_restart": False})
    finally:
        app.dependency_overrides.pop(get_current_admin_user, None)

    assert response.status_code == 200
    data = response.json()
    assert data["started"] is True
    assert data["state"] == "running"
    mock_start.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_price_repair_validates_date_format(client):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        response = await client.post(
            "/api/v1/sync/prices/repair/run",
            json={
                "symbols": ["AAA"],
                "start_date": "2026/01/01",
                "end_date": "2026-01-10",
            },
        )
    finally:
        app.dependency_overrides.pop(get_current_admin_user, None)

    assert response.status_code == 400
    assert "YYYY-MM-DD" in response.json()["detail"]
