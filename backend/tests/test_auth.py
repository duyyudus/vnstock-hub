import pytest
from httpx import AsyncClient


async def _login_and_get_auth_headers(client: AsyncClient, email: str, password: str = "password123"):
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_register_user(client: AsyncClient):
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "test_register@example.com", "password": "password123"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["user"]["email"] == "test_register@example.com"
    assert data["user"]["download_folder"] is None
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_register_existing_user(client: AsyncClient):
    # First registration
    await client.post(
        "/api/v1/auth/register",
        json={"email": "duplicate@example.com", "password": "password123"}
    )
    # Second registration
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "duplicate@example.com", "password": "password123"}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Email is already registered"


@pytest.mark.asyncio
async def test_login_user(client: AsyncClient):
    # Register first
    await client.post(
        "/api/v1/auth/register",
        json={"email": "login_user@example.com", "password": "password123"}
    )
    
    # Login success
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "login_user@example.com", "password": "password123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user"]["email"] == "login_user@example.com"
    assert data["user"]["download_folder"] is None
    assert "access_token" in data


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient):
    # Register first
    await client.post(
        "/api/v1/auth/register",
        json={"email": "wrong_pass@example.com", "password": "password123"}
    )
    
    # Login failure
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "wrong_pass@example.com", "password": "wrongpassword"}
    )
    assert response.status_code == 401
    assert "Incorrect email or password" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_auth_settings_requires_auth(client: AsyncClient):
    response = await client.get("/api/v1/auth/settings")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_patch_auth_settings_requires_auth(client: AsyncClient):
    response = await client.patch("/api/v1/auth/settings", json={"download_folder": "Reports"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_settings_update_and_get(client: AsyncClient):
    email = "settings_user@example.com"
    password = "password123"

    register_response = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password}
    )
    assert register_response.status_code == 201
    assert register_response.json()["user"]["download_folder"] is None

    headers = await _login_and_get_auth_headers(client, email, password)

    get_response = await client.get("/api/v1/auth/settings", headers=headers)
    assert get_response.status_code == 200
    assert get_response.json() == {"download_folder": None}

    update_response = await client.patch(
        "/api/v1/auth/settings",
        json={"download_folder": "Reports/Exports"},
        headers=headers,
    )
    assert update_response.status_code == 200
    assert update_response.json() == {"download_folder": "Reports/Exports"}

    get_after_update_response = await client.get("/api/v1/auth/settings", headers=headers)
    assert get_after_update_response.status_code == 200
    assert get_after_update_response.json() == {"download_folder": "Reports/Exports"}

    login_after_set = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password}
    )
    assert login_after_set.status_code == 200
    assert login_after_set.json()["user"]["download_folder"] == "Reports/Exports"

    clear_response = await client.patch(
        "/api/v1/auth/settings",
        json={"download_folder": "   "},
        headers=headers,
    )
    assert clear_response.status_code == 200
    assert clear_response.json() == {"download_folder": None}

    login_after_update = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password}
    )
    assert login_after_update.status_code == 200
    assert login_after_update.json()["user"]["download_folder"] is None
