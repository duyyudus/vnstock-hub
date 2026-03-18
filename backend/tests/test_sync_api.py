from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core.deps import get_current_admin_user
from app.main import app
from app.services.vnstock_service import vnstock_service


@pytest.mark.asyncio
async def test_get_sync_status_includes_unified_history_sync(client):
    response = await client.get("/api/v1/sync/status")
    assert response.status_code == 200

    payload = response.json()
    assert "fund_performance" in payload
    assert "history_sync" in payload
    assert "sync" in payload["history_sync"]
    assert "audit" in payload["history_sync"]
    assert "repair" in payload["history_sync"]
    assert "finance_sync" in payload
    assert "company_sync" in payload
    assert "failed_tickers" in payload["history_sync"]["sync"]
    assert "failed_tickers" in payload["history_sync"]["audit"]
    assert "failed_tickers" in payload["history_sync"]["repair"]
    assert "failed_tickers" in payload["finance_sync"]
    assert "failed_tickers" in payload["company_sync"]


@pytest.mark.asyncio
async def test_run_history_sync_requires_admin_auth(client):
    response = await client.post("/api/v1/sync/history/run", json={"force_restart": False})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_run_history_sync_with_admin_calls_service(client):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        with patch.object(
            vnstock_service,
            "run_history_sync",
            AsyncMock(return_value={"started": True, "message": "Price sync started", "state": "running"}),
        ) as mock_run:
            response = await client.post(
                "/api/v1/sync/history/run",
                json={"force_restart": False, "index_symbol": "VN30"},
            )
    finally:
        app.dependency_overrides.pop(get_current_admin_user, None)

    assert response.status_code == 200
    data = response.json()
    assert data["started"] is True
    assert data["state"] == "running"
    mock_run.assert_awaited_once()
    called_kwargs = mock_run.await_args.kwargs
    assert called_kwargs["index_symbol"] == "VN30"


@pytest.mark.asyncio
async def test_run_history_sync_invalid_index_returns_400(client):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        with patch.object(
            vnstock_service,
            "run_history_sync",
            AsyncMock(side_effect=ValueError("Unsupported index symbol")),
        ):
            response = await client.post(
                "/api/v1/sync/history/run",
                json={"force_restart": False, "index_symbol": "INVALID"},
            )
    finally:
        app.dependency_overrides.pop(get_current_admin_user, None)

    assert response.status_code == 400
    assert "Unsupported index symbol" in response.json()["detail"]


@pytest.mark.asyncio
async def test_run_history_audit_validates_date_format(client):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        response = await client.post(
            "/api/v1/sync/history/audit/run",
            json={
                "symbols": ["AAA"],
                "start_date": "2026/01/01",
                "end_date": "2026-01-10",
                "auto_repair": False,
            },
        )
    finally:
        app.dependency_overrides.pop(get_current_admin_user, None)

    assert response.status_code == 400
    assert "YYYY-MM-DD" in response.json()["detail"]


@pytest.mark.asyncio
async def test_run_history_repair_validates_date_format(client):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        response = await client.post(
            "/api/v1/sync/history/repair/run",
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


@pytest.mark.asyncio
async def test_run_finance_sync_requires_admin_auth(client):
    response = await client.post("/api/v1/sync/finance/run", json={"force_restart": False})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_run_finance_sync_with_admin_calls_service(client):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        with patch.object(
            vnstock_service,
            "run_finance_sync",
            AsyncMock(return_value={"started": True, "message": "Finance sync started", "state": "running"}),
        ) as mock_run:
            response = await client.post(
                "/api/v1/sync/finance/run",
                json={"force_restart": False, "index_symbol": "VN30", "quick_sync": True},
            )
    finally:
        app.dependency_overrides.pop(get_current_admin_user, None)

    assert response.status_code == 200
    data = response.json()
    assert data["started"] is True
    assert data["state"] == "running"
    mock_run.assert_awaited_once()
    called_kwargs = mock_run.await_args.kwargs
    assert called_kwargs["index_symbol"] == "VN30"
    assert called_kwargs["quick_sync"] is True


@pytest.mark.asyncio
async def test_run_finance_sync_invalid_index_returns_400(client):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        with patch.object(
            vnstock_service,
            "run_finance_sync",
            AsyncMock(side_effect=ValueError("Unsupported index symbol")),
        ):
            response = await client.post(
                "/api/v1/sync/finance/run",
                json={"force_restart": False, "index_symbol": "INVALID"},
            )
    finally:
        app.dependency_overrides.pop(get_current_admin_user, None)

    assert response.status_code == 400
    assert "Unsupported index symbol" in response.json()["detail"]


@pytest.mark.asyncio
async def test_run_company_sync_requires_admin_auth(client):
    response = await client.post("/api/v1/sync/company/run", json={"force_restart": False})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_run_company_sync_with_admin_calls_service(client):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        with patch.object(
            vnstock_service,
            "run_company_sync",
            AsyncMock(return_value={"started": True, "message": "Company sync started", "state": "running"}),
        ) as mock_run:
            response = await client.post(
                "/api/v1/sync/company/run",
                json={"force_restart": False, "index_symbol": "VN30", "quick_sync": True},
            )
    finally:
        app.dependency_overrides.pop(get_current_admin_user, None)

    assert response.status_code == 200
    data = response.json()
    assert data["started"] is True
    assert data["state"] == "running"
    mock_run.assert_awaited_once()
    called_kwargs = mock_run.await_args.kwargs
    assert called_kwargs["index_symbol"] == "VN30"
    assert called_kwargs["quick_sync"] is True


@pytest.mark.asyncio
async def test_run_company_sync_invalid_index_returns_400(client):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        with patch.object(
            vnstock_service,
            "run_company_sync",
            AsyncMock(side_effect=ValueError("Unsupported index symbol")),
        ):
            response = await client.post(
                "/api/v1/sync/company/run",
                json={"force_restart": False, "index_symbol": "INVALID"},
            )
    finally:
        app.dependency_overrides.pop(get_current_admin_user, None)

    assert response.status_code == 400
    assert "Unsupported index symbol" in response.json()["detail"]
