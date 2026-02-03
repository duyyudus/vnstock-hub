import pytest
from httpx import AsyncClient


@pytest.fixture
async def auth_headers(client: AsyncClient):
    email = "portfolio_tester@example.com"
    password = "password123"

    await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password}
    )
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password}
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_portfolio_requires_auth(client: AsyncClient):
    response = await client.get("/api/v1/portfolio/positions")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_create_and_list_positions(client: AsyncClient, auth_headers):
    payload = {
        "ticker": "TCB",
        "quantity": 100,
        "average_cost": 42000,
        "purchase_date": "2024-01-15"
    }

    response = await client.post(
        "/api/v1/portfolio/positions",
        json=payload,
        headers=auth_headers
    )
    assert response.status_code == 201
    data = response.json()
    assert data["ticker"] == "TCB"
    assert data["quantity"] == 100
    assert data["average_cost"] == 42000
    assert data["purchase_date"] == "2024-01-15"

    list_response = await client.get("/api/v1/portfolio/positions", headers=auth_headers)
    assert list_response.status_code == 200
    list_data = list_response.json()
    assert list_data["count"] == 1
    assert list_data["positions"][0]["ticker"] == "TCB"


@pytest.mark.asyncio
async def test_create_position_without_purchase_date(client: AsyncClient, auth_headers):
    payload = {
        "ticker": "FPT",
        "quantity": 25,
        "average_cost": 95000
    }

    response = await client.post(
        "/api/v1/portfolio/positions",
        json=payload,
        headers=auth_headers
    )
    assert response.status_code == 201
    data = response.json()
    assert data["ticker"] == "FPT"
    assert data["purchase_date"] is None


@pytest.mark.asyncio
async def test_duplicate_position_rejected(client: AsyncClient, auth_headers):
    payload = {
        "ticker": "VCB",
        "quantity": 50,
        "average_cost": 90000,
        "purchase_date": "2024-02-01"
    }

    response = await client.post(
        "/api/v1/portfolio/positions",
        json=payload,
        headers=auth_headers
    )
    assert response.status_code == 201

    duplicate_response = await client.post(
        "/api/v1/portfolio/positions",
        json=payload,
        headers=auth_headers
    )
    assert duplicate_response.status_code == 409


@pytest.mark.asyncio
async def test_update_and_delete_position(client: AsyncClient, auth_headers):
    payload = {
        "ticker": "SSI",
        "quantity": 200,
        "average_cost": 25000,
        "purchase_date": "2024-01-10"
    }

    response = await client.post(
        "/api/v1/portfolio/positions",
        json=payload,
        headers=auth_headers
    )
    position_id = response.json()["id"]

    update_payload = {
        "quantity": 220,
        "average_cost": 26000,
        "purchase_date": "2024-02-10"
    }

    update_response = await client.patch(
        f"/api/v1/portfolio/positions/{position_id}",
        json=update_payload,
        headers=auth_headers
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["quantity"] == 220
    assert updated["average_cost"] == 26000
    assert updated["purchase_date"] == "2024-02-10"

    delete_response = await client.delete(
        f"/api/v1/portfolio/positions/{position_id}",
        headers=auth_headers
    )
    assert delete_response.status_code == 204

    list_response = await client.get("/api/v1/portfolio/positions", headers=auth_headers)
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 0
