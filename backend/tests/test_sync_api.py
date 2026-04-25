from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core.deps import get_current_admin_user
from app.db.models import ScheduledSyncJobRun
from app.main import app
from app.services.vnstock_service import vnstock_service
from app.services.vnstock_service.scheduler import ScheduledSyncRunStatus, utc_now


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
@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/sync/history/cancel",
        "/api/v1/sync/history/audit/cancel",
        "/api/v1/sync/history/repair/cancel",
        "/api/v1/sync/finance/cancel",
        "/api/v1/sync/company/cancel",
    ],
)
async def test_cancel_sync_requires_admin_auth(client, path):
    response = await client.post(path)
    assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "service_method", "message"),
    [
        ("/api/v1/sync/history/cancel", "cancel_history_sync", "History sync cancelled"),
        ("/api/v1/sync/history/audit/cancel", "cancel_history_audit_sync", "History audit cancelled"),
        ("/api/v1/sync/history/repair/cancel", "cancel_history_repair_sync", "History repair cancelled"),
        ("/api/v1/sync/finance/cancel", "cancel_finance_sync", "Finance sync cancelled"),
        ("/api/v1/sync/company/cancel", "cancel_company_sync", "Company sync cancelled"),
    ],
)
async def test_cancel_sync_with_admin_calls_matching_service_method(client, path, service_method, message):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        with patch.object(
            vnstock_service,
            service_method,
            AsyncMock(return_value={"started": False, "message": message, "state": "cancelled"}),
        ) as mock_cancel:
            response = await client.post(path)
    finally:
        app.dependency_overrides.pop(get_current_admin_user, None)

    assert response.status_code == 200
    data = response.json()
    assert data["started"] is False
    assert data["state"] == "cancelled"
    assert data["message"] == message
    mock_cancel.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "service_method", "message"),
    [
        ("/api/v1/sync/history/cancel", "cancel_history_sync", "History sync is not running"),
        ("/api/v1/sync/history/audit/cancel", "cancel_history_audit_sync", "History audit is not running"),
        ("/api/v1/sync/history/repair/cancel", "cancel_history_repair_sync", "History repair is not running"),
        ("/api/v1/sync/finance/cancel", "cancel_finance_sync", "Finance sync is not running"),
        ("/api/v1/sync/company/cancel", "cancel_company_sync", "Company sync is not running"),
    ],
)
async def test_cancel_sync_idle_returns_idle_state(client, path, service_method, message):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        with patch.object(
            vnstock_service,
            service_method,
            AsyncMock(return_value={"started": False, "message": message, "state": "idle"}),
        ):
            response = await client.post(path)
    finally:
        app.dependency_overrides.pop(get_current_admin_user, None)

    assert response.status_code == 200
    data = response.json()
    assert data["started"] is False
    assert data["state"] == "idle"
    assert data["message"] == message


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
async def test_run_history_repair_with_admin_calls_service(client):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        with patch.object(
            vnstock_service,
            "run_history_repair_sync",
            AsyncMock(return_value={"started": True, "message": "Repair sync started", "state": "running"}),
        ) as mock_run:
            response = await client.post(
                "/api/v1/sync/history/repair/run",
                json={"index_symbol": "VN30", "start_date": "2026-01-01", "end_date": "2026-01-10"},
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
    assert called_kwargs["symbols"] is None


@pytest.mark.asyncio
async def test_run_history_repair_invalid_index_returns_400(client):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        with patch.object(
            vnstock_service,
            "run_history_repair_sync",
            AsyncMock(side_effect=ValueError("Unsupported index symbol")),
        ):
            response = await client.post(
                "/api/v1/sync/history/repair/run",
                json={"index_symbol": "INVALID", "start_date": "2026-01-01", "end_date": "2026-01-10"},
            )
    finally:
        app.dependency_overrides.pop(get_current_admin_user, None)

    assert response.status_code == 400
    assert "Unsupported index symbol" in response.json()["detail"]


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
                json={
                    "force_restart": False,
                    "index_symbol": "VN30",
                    "quick_sync": True,
                    "force_refresh": True,
                },
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
    assert called_kwargs["force_refresh"] is True


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
                json={
                    "force_restart": False,
                    "index_symbol": "VN30",
                    "quick_sync": True,
                    "force_refresh": True,
                },
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
    assert called_kwargs["force_refresh"] is True


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


@pytest.mark.asyncio
async def test_scheduler_jobs_require_admin_auth(client):
    response = await client.get("/api/v1/sync/scheduler/jobs")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_create_scheduler_job_with_admin(client):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        response = await client.post(
            "/api/v1/sync/scheduler/jobs",
            json={
                "name": "Daily Finance Quick",
                "enabled": True,
                "sync_type": "finance",
                "sync_action": "quick",
                "symbols": ["VCB", "ACB"],
                "starts_at": "2026-03-27T09:00:00",
                "interval_value": 1,
                "interval_unit": "days",
                "timezone": "Asia/Ho_Chi_Minh",
                "max_retries": 2,
            },
        )
    finally:
        app.dependency_overrides.pop(get_current_admin_user, None)

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "Daily Finance Quick"
    assert payload["sync_type"] == "finance"
    assert payload["sync_action"] == "quick"
    assert payload["symbols"] == ["VCB", "ACB"]
    assert payload["partial_success_failure_threshold_percent"] == 10


@pytest.mark.asyncio
async def test_update_scheduler_job_validates_history_date_range(client):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    created_job = await vnstock_service.scheduler.create_job(
        {
            "name": "History Sync",
            "sync_type": "history",
            "sync_action": "sync",
            "starts_at": "2026-03-27T09:00:00",
            "interval_value": 1,
            "interval_unit": "days",
            "timezone": "Asia/Ho_Chi_Minh",
            "max_retries": 0,
            "partial_success_failure_threshold_percent": 10,
        }
    )

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        response = await client.patch(
            f"/api/v1/sync/scheduler/jobs/{created_job['id']}",
            json={
                "sync_action": "repair",
            },
        )
    finally:
        app.dependency_overrides.pop(get_current_admin_user, None)

    assert response.status_code == 400
    assert "date_from and date_to are required" in response.json()["detail"]


@pytest.mark.asyncio
async def test_list_scheduler_runs_returns_recent_attempts(client, db_session):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    job = await vnstock_service.scheduler.create_job(
        {
            "name": "Company Quick",
            "sync_type": "company",
            "sync_action": "quick",
            "starts_at": "2026-03-27T09:00:00",
            "interval_value": 1,
            "interval_unit": "days",
            "timezone": "Asia/Ho_Chi_Minh",
            "max_retries": 1,
        }
    )

    db_session.add(
        ScheduledSyncJobRun(
            job_id=job["id"],
            attempt_number=1,
            status=ScheduledSyncRunStatus.PARTIAL_SUCCEEDED.value,
            scheduled_for=utc_now(),
            finished_at=utc_now(),
            error=None,
            summary={"message": "partial"},
        )
    )
    await db_session.commit()

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        response = await client.get("/api/v1/sync/scheduler/runs?limit=10")
    finally:
        app.dependency_overrides.pop(get_current_admin_user, None)

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] >= 1
    assert payload["runs"][0]["job_name"] == "Company Quick"
    assert payload["runs"][0]["status"] == ScheduledSyncRunStatus.PARTIAL_SUCCEEDED.value


@pytest.mark.asyncio
async def test_update_scheduler_job_accepts_partial_success_threshold(client):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    created_job = await vnstock_service.scheduler.create_job(
        {
            "name": "Finance Sync",
            "sync_type": "finance",
            "sync_action": "quick",
            "starts_at": "2026-03-27T09:00:00",
            "interval_value": 1,
            "interval_unit": "days",
            "timezone": "Asia/Ho_Chi_Minh",
            "max_retries": 0,
        }
    )

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        response = await client.patch(
            f"/api/v1/sync/scheduler/jobs/{created_job['id']}",
            json={
                "partial_success_failure_threshold_percent": 25,
            },
        )
    finally:
        app.dependency_overrides.pop(get_current_admin_user, None)

    assert response.status_code == 200
    payload = response.json()
    assert payload["partial_success_failure_threshold_percent"] == 25


@pytest.mark.asyncio
async def test_delete_scheduler_job_returns_204(client):
    async def _admin_override():
        return SimpleNamespace(id=1, email="admin@example.com")

    job = await vnstock_service.scheduler.create_job(
        {
            "name": "Delete Me",
            "sync_type": "finance",
            "sync_action": "full",
            "starts_at": "2026-03-27T09:00:00",
            "interval_value": 1,
            "interval_unit": "days",
            "timezone": "Asia/Ho_Chi_Minh",
            "max_retries": 0,
        }
    )

    app.dependency_overrides[get_current_admin_user] = _admin_override
    try:
        response = await client.delete(f"/api/v1/sync/scheduler/jobs/{job['id']}")
    finally:
        app.dependency_overrides.pop(get_current_admin_user, None)

    assert response.status_code == 204
