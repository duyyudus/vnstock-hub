import pytest
from httpx import AsyncClient
import json


@pytest.fixture
async def auth_headers(client: AsyncClient):
    email = "trading_tester@example.com"
    password = "password123"

    await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password},
    )
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def _create_trading_position(client: AsyncClient, headers: dict, payload: dict):
    response = await client.post(
        "/api/v1/trading/positions",
        json=payload,
        headers=headers,
    )
    assert response.status_code == 201
    return response.json()


@pytest.mark.asyncio
async def test_trading_requires_auth(client: AsyncClient):
    response = await client.get("/api/v1/trading/positions")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_create_and_list_trading_positions(client: AsyncClient, auth_headers):
    payload = {
        "account_label": "Swing Account",
        "ticker": "SSI",
        "quantity": 200,
        "average_entry_cost": 25500,
        "opened_date": "2026-03-10",
        "target_price": 29000,
        "stop_loss": 24000,
        "notes": "Watch breakout confirmation",
    }

    response = await client.post(
        "/api/v1/trading/positions",
        json=payload,
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["account_label"] == "Swing Account"
    assert data["ticker"] == "SSI"
    assert data["average_entry_cost"] == 25500
    assert data["opened_date"] == "2026-03-10"

    list_response = await client.get("/api/v1/trading/positions", headers=auth_headers)
    assert list_response.status_code == 200
    list_data = list_response.json()
    assert list_data["count"] == 1
    assert list_data["positions"][0]["ticker"] == "SSI"


@pytest.mark.asyncio
async def test_duplicate_ticker_allowed_across_accounts(client: AsyncClient, auth_headers):
    payload = {
        "ticker": "FPT",
        "quantity": 50,
        "average_entry_cost": 121000,
        "opened_date": "2026-03-05",
    }

    first_response = await client.post(
        "/api/v1/trading/positions",
        json={**payload, "account_label": "Swing"},
        headers=auth_headers,
    )
    assert first_response.status_code == 201

    second_response = await client.post(
        "/api/v1/trading/positions",
        json={**payload, "account_label": "Scalp"},
        headers=auth_headers,
    )
    assert second_response.status_code == 201

    duplicate_response = await client.post(
        "/api/v1/trading/positions",
        json={**payload, "account_label": "swing"},
        headers=auth_headers,
    )
    assert duplicate_response.status_code == 409


@pytest.mark.asyncio
async def test_create_position_without_opened_date(client: AsyncClient, auth_headers):
    response = await client.post(
        "/api/v1/trading/positions",
        json={
            "account_label": "No Date Account",
            "ticker": "MBB",
            "quantity": 30,
            "average_entry_cost": 21500,
        },
        headers=auth_headers,
    )
    assert response.status_code == 201
    assert response.json()["opened_date"] is None


@pytest.mark.asyncio
async def test_update_and_delete_trading_position(client: AsyncClient, auth_headers):
    created = await _create_trading_position(
        client,
        auth_headers,
        {
            "account_label": "Momentum",
            "ticker": "VCI",
            "quantity": 120,
            "average_entry_cost": 36500,
            "opened_date": "2026-03-01",
        },
    )
    position_id = created["id"]

    update_response = await client.patch(
        f"/api/v1/trading/positions/{position_id}",
        json={
            "account_label": "Momentum Prime",
            "ticker": "VCI",
            "quantity": 140,
            "average_entry_cost": 37000,
            "opened_date": "2026-03-02",
            "target_price": 41000,
            "stop_loss": 35000,
            "notes": "Raised target after earnings",
        },
        headers=auth_headers,
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["account_label"] == "Momentum Prime"
    assert updated["quantity"] == 140
    assert updated["target_price"] == 41000
    assert updated["notes"] == "Raised target after earnings"

    clear_date_response = await client.patch(
        f"/api/v1/trading/positions/{position_id}",
        json={"opened_date": None},
        headers=auth_headers,
    )
    assert clear_date_response.status_code == 200
    assert clear_date_response.json()["opened_date"] is None

    delete_response = await client.delete(
        f"/api/v1/trading/positions/{position_id}",
        headers=auth_headers,
    )
    assert delete_response.status_code == 204

    list_response = await client.get("/api/v1/trading/positions", headers=auth_headers)
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 0


@pytest.mark.asyncio
async def test_trading_positions_do_not_affect_portfolio(client: AsyncClient, auth_headers):
    await _create_trading_position(
        client,
        auth_headers,
        {
            "account_label": "Trading Desk",
            "ticker": "TCB",
            "quantity": 80,
            "average_entry_cost": 31000,
            "opened_date": "2026-03-12",
        },
    )

    portfolio_response = await client.get("/api/v1/portfolio/positions", headers=auth_headers)
    assert portfolio_response.status_code == 200
    assert portfolio_response.json()["count"] == 0

    create_portfolio_response = await client.post(
        "/api/v1/portfolio/positions",
        json={
            "ticker": "TCB",
            "quantity": 20,
            "average_cost": 28000,
            "purchase_date": "2024-01-01",
        },
        headers=auth_headers,
    )
    assert create_portfolio_response.status_code == 201

    trading_response = await client.get("/api/v1/trading/positions", headers=auth_headers)
    assert trading_response.status_code == 200
    assert trading_response.json()["count"] == 1


@pytest.mark.asyncio
async def test_trading_import_requires_auth(client: AsyncClient):
    response = await client.post(
        "/api/v1/trading/import",
        files={"file": ("position.png", b"fake-image", "image/png")},
        data={
            "broker_id": "vpbanks",
            "account_label": "Trading",
        },
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_trading_import_upserts_positions_from_images(client: AsyncClient, auth_headers, monkeypatch):
    from app.core import config
    from app.api.v1 import trading as trading_api
    from app.services.llm.llm_client import ImagePositionItem

    async def fake_extract_positions_from_image(file, providers, timeout_seconds):
        return [
            ImagePositionItem(ticker="SSI", average_cost=25.5, quantity=100),
            ImagePositionItem(ticker="VCI", average_cost=38.2, quantity=40),
        ]

    monkeypatch.setattr(trading_api, "extract_positions_from_image", fake_extract_positions_from_image)
    monkeypatch.setattr(config.settings, "llm_providers", json.dumps([
        {"name": "test", "base_url": "http://example.com", "api_key": "test", "model": "test"}
    ]))

    existing = await client.post(
        "/api/v1/trading/positions",
        json={
            "account_label": "SSI Swing",
            "ticker": "SSI",
            "quantity": 50,
            "average_entry_cost": 24,
            "opened_date": "2026-03-01",
            "target_price": 30,
            "stop_loss": 22,
            "notes": "Keep notes",
        },
        headers=auth_headers,
    )
    assert existing.status_code == 201

    response = await client.post(
        "/api/v1/trading/import",
        files=[
            ("file", ("position-1.png", b"fake-image-1", "image/png")),
            ("file", ("position-2.png", b"fake-image-2", "image/png")),
        ],
        data={
            "broker_id": "vpbanks",
            "account_label": "SSI Swing",
            "opened_date": "2026-03-24",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["created_count"] == 1
    assert data["updated_count"] == 1
    assert len(data["imported_positions"]) == 2

    list_response = await client.get("/api/v1/trading/positions", headers=auth_headers)
    assert list_response.status_code == 200
    positions = {item["ticker"]: item for item in list_response.json()["positions"]}
    assert positions["SSI"]["quantity"] == 100
    assert positions["SSI"]["average_entry_cost"] == 25500
    assert positions["SSI"]["opened_date"] == "2026-03-01"
    assert positions["SSI"]["target_price"] == 30
    assert positions["SSI"]["notes"] == "Keep notes"
    assert positions["VCI"]["quantity"] == 40
    assert positions["VCI"]["average_entry_cost"] == 38200
    assert positions["VCI"]["opened_date"] == "2026-03-24"


@pytest.mark.asyncio
async def test_trading_import_allows_missing_opened_date_for_new_positions(client: AsyncClient, auth_headers, monkeypatch):
    from app.core import config
    from app.api.v1 import trading as trading_api
    from app.services.llm.llm_client import ImagePositionItem

    async def fake_extract_positions_from_image(file, providers, timeout_seconds):
        return [
            ImagePositionItem(ticker="DGC", average_cost=98.5, quantity=12),
        ]

    monkeypatch.setattr(trading_api, "extract_positions_from_image", fake_extract_positions_from_image)
    monkeypatch.setattr(config.settings, "llm_providers", json.dumps([
        {"name": "test", "base_url": "http://example.com", "api_key": "test", "model": "test"}
    ]))

    response = await client.post(
        "/api/v1/trading/import",
        files={"file": ("position.png", b"fake-image", "image/png")},
        data={
            "broker_id": "vpbanks",
            "account_label": "Quick Trade",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200

    list_response = await client.get("/api/v1/trading/positions", headers=auth_headers)
    assert list_response.status_code == 200
    positions = {item["ticker"]: item for item in list_response.json()["positions"]}
    assert positions["DGC"]["opened_date"] is None


@pytest.mark.asyncio
async def test_trading_import_skips_missing_quantity_for_new_positions(client: AsyncClient, auth_headers, monkeypatch):
    from app.core import config
    from app.api.v1 import trading as trading_api
    from app.services.llm.llm_client import ImagePositionItem

    async def fake_extract_positions_from_image(file, providers, timeout_seconds):
        return [
            ImagePositionItem(ticker="FPT", average_cost=120.5, quantity=None),
        ]

    monkeypatch.setattr(trading_api, "extract_positions_from_image", fake_extract_positions_from_image)
    monkeypatch.setattr(config.settings, "llm_providers", json.dumps([
        {"name": "test", "base_url": "http://example.com", "api_key": "test", "model": "test"}
    ]))

    response = await client.post(
        "/api/v1/trading/import",
        files={"file": ("position.png", b"fake-image", "image/png")},
        data={
            "broker_id": "vpbanks",
            "account_label": "Quick Trade",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["created_count"] == 0
    assert data["updated_count"] == 0
    assert data["skipped_count"] == 1

    list_response = await client.get("/api/v1/trading/positions", headers=auth_headers)
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 0
