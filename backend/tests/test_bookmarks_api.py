import pytest
from httpx import AsyncClient
from unittest.mock import patch, MagicMock
from app.services.vnstock_service import vnstock_service
from app.services.vnstock_service.models import StockInfo

# Fixture to get auth headers for a test user
@pytest.fixture
async def auth_headers(client: AsyncClient):
    email = "bookmark_tester@example.com"
    password = "password123"
    
    # Register
    await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password}
    )
    
    # Login
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password}
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

@pytest.mark.asyncio
async def test_create_bookmark_group(client: AsyncClient, auth_headers):
    """Test creating a new bookmark group."""
    response = await client.post(
        "/api/v1/bookmarks/groups",
        json={"name": "My Watchlist"},
        headers=auth_headers
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "My Watchlist"
    assert data["tickers"] == []
    assert "id" in data

@pytest.mark.asyncio
async def test_create_duplicate_group(client: AsyncClient, auth_headers):
    """Test that creating a group with a duplicate name for the same user fails."""
    # Create first time
    await client.post(
        "/api/v1/bookmarks/groups",
        json={"name": "Duplicate Group"},
        headers=auth_headers
    )
    # Create second time
    response = await client.post(
        "/api/v1/bookmarks/groups",
        json={"name": "Duplicate Group"},
        headers=auth_headers
    )
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]

@pytest.mark.asyncio
async def test_create_group_invalid_name(client: AsyncClient, auth_headers):
    """Test validation for empty group name."""
    response = await client.post(
        "/api/v1/bookmarks/groups",
        json={"name": ""},
        headers=auth_headers
    )
    assert response.status_code == 422  # Pydantic validation error

@pytest.mark.asyncio
async def test_list_bookmark_groups(client: AsyncClient, auth_headers):
    """Test listing bookmark groups for the current user."""
    # Create some groups
    await client.post(
        "/api/v1/bookmarks/groups",
        json={"name": "Alpha Group"},
        headers=auth_headers
    )
    await client.post(
        "/api/v1/bookmarks/groups",
        json={"name": "Beta Group"},
        headers=auth_headers
    )
    
    response = await client.get("/api/v1/bookmarks/groups", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 2
    
    names = [g["name"] for g in data["groups"]]
    assert "Alpha Group" in names
    assert "Beta Group" in names

@pytest.mark.asyncio
async def test_add_stock_to_group(client: AsyncClient, auth_headers):
    """Test adding a stock to a bookmark group."""
    # Create group
    g_res = await client.post(
        "/api/v1/bookmarks/groups",
        json={"name": "Tech Stocks"},
        headers=auth_headers
    )
    group_id = g_res.json()["id"]
    
    # Add stock
    response = await client.post(
        f"/api/v1/bookmarks/groups/{group_id}/stocks",
        json={"ticker": "TCB"},
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "TCB" in data["tickers"]
    
    # Add same stock again (should be idempotent)
    response = await client.post(
        f"/api/v1/bookmarks/groups/{group_id}/stocks",
        json={"ticker": "TCB"},
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["tickers"].count("TCB") == 1

@pytest.mark.asyncio
async def test_add_stock_to_nonexistent_group(client: AsyncClient, auth_headers):
    """Test adding a stock to a non-existent group."""
    response = await client.post(
        "/api/v1/bookmarks/groups/99999/stocks",
        json={"ticker": "TCB"},
        headers=auth_headers
    )
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_remove_stock_from_group(client: AsyncClient, auth_headers):
    """Test removing a stock from a bookmark group."""
    # Create group and add stock
    g_res = await client.post(
        "/api/v1/bookmarks/groups",
        json={"name": "My Portfolio"},
        headers=auth_headers
    )
    group_id = g_res.json()["id"]
    await client.post(
        f"/api/v1/bookmarks/groups/{group_id}/stocks",
        json={"ticker": "VCB"},
        headers=auth_headers
    )
    
    # Remove stock
    response = await client.delete(
        f"/api/v1/bookmarks/groups/{group_id}/stocks/VCB",
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "VCB" not in data["tickers"]

    # Remove non-existent stock (should be idempotent/silent success or handled)
    # The API implementation just deletes if exists, returns group.
    response = await client.delete(
        f"/api/v1/bookmarks/groups/{group_id}/stocks/VCB",
        headers=auth_headers
    )
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_get_group_stocks_details(client: AsyncClient, auth_headers):
    """Test fetching enriched stock details for a group."""
    # Create group and add stock
    g_res = await client.post(
        "/api/v1/bookmarks/groups",
        json={"name": "Bank Stocks"},
        headers=auth_headers
    )
    group_id = g_res.json()["id"]
    await client.post(
        f"/api/v1/bookmarks/groups/{group_id}/stocks",
        json={"ticker": "TCB"},
        headers=auth_headers
    )
    
    # Mock vnstock service response
    mock_stock_info = StockInfo(
        ticker="TCB", 
        price=50000.0, 
        market_cap=100000.0, 
        company_name="Techcombank", 
        exchange="HOSE",
        charter_capital=0.0, 
        pe_ratio=None, 
        accumulated_value=None,
        foreign_buy_value=12.3,
        foreign_sell_value=9.8,
        price_change_24h=1.5, 
        price_change_1w=None,
        price_change_1m=None, 
        price_change_1y=None
    )
    
    with patch.object(vnstock_service, 'get_symbol_stocks', return_value=[mock_stock_info]):
        response = await client.get(
            f"/api/v1/bookmarks/groups/{group_id}/stocks",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["group_name"] == "Bank Stocks"
        assert len(data["stocks"]) == 1
        stock = data["stocks"][0]
        assert stock["ticker"] == "TCB"
        assert stock["company_name"] == "Techcombank"
        assert stock["price"] == 50000.0
        assert stock["foreign_buy_value"] == 12.3
        assert stock["foreign_sell_value"] == 9.8

@pytest.mark.asyncio
async def test_unauthorized_access(client: AsyncClient, auth_headers):
    """Test that a user cannot access or modify another user's bookmark groups."""
    # User A creates a group
    g_res = await client.post(
        "/api/v1/bookmarks/groups",
        json={"name": "User A Group"},
        headers=auth_headers
    )
    group_id = g_res.json()["id"]
    
    # User B logs in
    email_b = "user_b@example.com"
    password_b = "password123"
    await client.post("/api/v1/auth/register", json={"email": email_b, "password": password_b})
    login_res = await client.post("/api/v1/auth/login", json={"email": email_b, "password": password_b})
    token_b = login_res.json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}
    
    # User B tries to access User A's group
    response = await client.get(f"/api/v1/bookmarks/groups/{group_id}/stocks", headers=headers_b)
    assert response.status_code == 404 # Should be 404 because of _get_group_or_404 filter
    
    # User B tries to add stock to User A's group
    response = await client.post(
        f"/api/v1/bookmarks/groups/{group_id}/stocks",
        json={"ticker": "FPT"},
        headers=headers_b
    )
    assert response.status_code == 404
    
    # User B tries to remove stock from User A's group
    response = await client.delete(
        f"/api/v1/bookmarks/groups/{group_id}/stocks/TCB",
        headers=headers_b
    )
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_edge_case_long_name(client: AsyncClient, auth_headers):
    """Test group creation with a very long name."""
    long_name = "A" * 121
    response = await client.post(
        "/api/v1/bookmarks/groups",
        json={"name": long_name},
        headers=auth_headers
    )
    assert response.status_code == 422 # Max length is 120

@pytest.mark.asyncio
async def test_edge_case_special_chars(client: AsyncClient, auth_headers):
    """Test group creation with special characters."""
    special_name = "My Stocks! @# %^&*()"
    response = await client.post(
        "/api/v1/bookmarks/groups",
        json={"name": special_name},
        headers=auth_headers
    )
    assert response.status_code == 201
    assert response.json()["name"] == special_name
