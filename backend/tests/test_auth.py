import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_register_user(client: AsyncClient):
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "test_register@example.com", "password": "password123"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["user"]["email"] == "test_register@example.com"
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
