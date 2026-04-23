import pytest
from httpx import AsyncClient
import csv
import io
import json
from datetime import date


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


async def _create_position(client: AsyncClient, headers: dict, payload: dict):
    response = await client.post(
        "/api/v1/portfolio/positions",
        json=payload,
        headers=headers,
    )
    assert response.status_code == 201
    return response.json()


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


@pytest.mark.asyncio
async def test_portfolio_export_csv_requires_auth(client: AsyncClient):
    response = await client.get("/api/v1/portfolio/export/csv")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_portfolio_export_csv_returns_expected_content(client: AsyncClient, auth_headers):
    await _create_position(
        client,
        auth_headers,
        {
            "ticker": "TCB",
            "quantity": 100,
            "average_cost": 42000,
            "purchase_date": "2024-01-15",
        },
    )
    await _create_position(
        client,
        auth_headers,
        {
            "ticker": "FPT",
            "quantity": 25,
        },
    )

    response = await client.get("/api/v1/portfolio/export/csv", headers=auth_headers)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    expected_filename = f"portfolio_tester_{date.today().isoformat()}.csv"
    assert f"attachment; filename=\"{expected_filename}\"" in response.headers["content-disposition"]

    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == 2
    assert list(rows[0].keys()) == ["ticker", "quantity", "average_cost", "purchase_date"]

    assert rows[0]["ticker"] == "TCB"
    assert float(rows[0]["quantity"]) == 100
    assert float(rows[0]["average_cost"]) == 42000
    assert rows[0]["purchase_date"] == "2024-01-15"

    assert rows[1]["ticker"] == "FPT"
    assert float(rows[1]["quantity"]) == 25
    assert rows[1]["average_cost"] == ""
    assert rows[1]["purchase_date"] == ""


@pytest.mark.asyncio
async def test_portfolio_fresh_import_requires_auth(client: AsyncClient):
    response = await client.post(
        "/api/v1/portfolio/import/fresh",
        files={"file": ("portfolio.csv", "ticker,quantity,average_cost,purchase_date\nTCB,1,,\n", "text/csv")},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_portfolio_fresh_import_missing_file(client: AsyncClient, auth_headers):
    response = await client.post("/api/v1/portfolio/import/fresh", headers=auth_headers)
    assert response.status_code == 400
    assert response.json()["detail"] == "File is required"


@pytest.mark.asyncio
async def test_portfolio_fresh_import_invalid_header_keeps_existing(client: AsyncClient, auth_headers):
    await _create_position(
        client,
        auth_headers,
        {
            "ticker": "SSI",
            "quantity": 200,
            "average_cost": 25000,
        },
    )

    response = await client.post(
        "/api/v1/portfolio/import/fresh",
        files={"file": ("portfolio.csv", "ticker,quantity,purchase_date\nTCB,100,2024-01-15\n", "text/csv")},
        headers=auth_headers,
    )
    assert response.status_code == 400

    list_response = await client.get("/api/v1/portfolio/positions", headers=auth_headers)
    assert list_response.status_code == 200
    list_data = list_response.json()
    assert list_data["count"] == 1
    assert list_data["positions"][0]["ticker"] == "SSI"


@pytest.mark.asyncio
async def test_portfolio_fresh_import_invalid_row_keeps_existing(client: AsyncClient, auth_headers):
    await _create_position(
        client,
        auth_headers,
        {
            "ticker": "SSI",
            "quantity": 200,
        },
    )

    csv_payload = (
        "ticker,quantity,average_cost,purchase_date\n"
        "TCB,abc,42000,2024-01-15\n"
    )
    response = await client.post(
        "/api/v1/portfolio/import/fresh",
        files={"file": ("portfolio.csv", csv_payload, "text/csv")},
        headers=auth_headers,
    )
    assert response.status_code == 400

    list_response = await client.get("/api/v1/portfolio/positions", headers=auth_headers)
    assert list_response.status_code == 200
    list_data = list_response.json()
    assert list_data["count"] == 1
    assert list_data["positions"][0]["ticker"] == "SSI"


@pytest.mark.asyncio
async def test_portfolio_fresh_import_duplicate_ticker_keeps_existing(client: AsyncClient, auth_headers):
    await _create_position(
        client,
        auth_headers,
        {
            "ticker": "SSI",
            "quantity": 200,
        },
    )

    csv_payload = (
        "ticker,quantity,average_cost,purchase_date\n"
        "TCB,100,42000,2024-01-15\n"
        "tcb,200,43000,2024-01-16\n"
    )
    response = await client.post(
        "/api/v1/portfolio/import/fresh",
        files={"file": ("portfolio.csv", csv_payload, "text/csv")},
        headers=auth_headers,
    )
    assert response.status_code == 400

    list_response = await client.get("/api/v1/portfolio/positions", headers=auth_headers)
    assert list_response.status_code == 200
    list_data = list_response.json()
    assert list_data["count"] == 1
    assert list_data["positions"][0]["ticker"] == "SSI"


@pytest.mark.asyncio
async def test_portfolio_fresh_import_empty_csv_rejected(client: AsyncClient, auth_headers):
    await _create_position(
        client,
        auth_headers,
        {
            "ticker": "SSI",
            "quantity": 200,
        },
    )

    response = await client.post(
        "/api/v1/portfolio/import/fresh",
        files={"file": ("portfolio.csv", "ticker,quantity,average_cost,purchase_date\n", "text/csv")},
        headers=auth_headers,
    )
    assert response.status_code == 400

    list_response = await client.get("/api/v1/portfolio/positions", headers=auth_headers)
    assert list_response.status_code == 200
    list_data = list_response.json()
    assert list_data["count"] == 1
    assert list_data["positions"][0]["ticker"] == "SSI"


@pytest.mark.asyncio
async def test_portfolio_fresh_import_replaces_positions(client: AsyncClient, auth_headers):
    await _create_position(
        client,
        auth_headers,
        {
            "ticker": "SSI",
            "quantity": 200,
            "average_cost": 25000,
            "purchase_date": "2024-01-10",
        },
    )
    await _create_position(
        client,
        auth_headers,
        {
            "ticker": "VCB",
            "quantity": 50,
            "average_cost": 90000,
            "purchase_date": "2024-02-01",
        },
    )

    csv_payload = (
        "ticker,quantity,average_cost,purchase_date\n"
        "TCB,100,42000,2024-01-15\n"
        "FPT,25,,\n"
    )
    response = await client.post(
        "/api/v1/portfolio/import/fresh",
        files={"file": ("portfolio.csv", csv_payload, "text/csv")},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["deleted_count"] == 2
    assert data["created_count"] == 2
    assert len(data["positions"]) == 2
    assert {item["ticker"] for item in data["positions"]} == {"TCB", "FPT"}

    list_response = await client.get("/api/v1/portfolio/positions", headers=auth_headers)
    assert list_response.status_code == 200
    list_data = list_response.json()
    assert list_data["count"] == 2
    assert {item["ticker"] for item in list_data["positions"]} == {"TCB", "FPT"}


@pytest.mark.asyncio
async def test_portfolio_import_brokers_returns_defaults(client: AsyncClient, auth_headers):
    response = await client.get("/api/v1/portfolio/import/brokers", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["id"] == "vpbanks"
    assert data[0]["top_left"] == "A9"


@pytest.mark.asyncio
async def test_portfolio_import_missing_inputs(client: AsyncClient, auth_headers):
    response = await client.post(
        "/api/v1/portfolio/import",
        data={"broker_id": "vpbanks"},
        headers=auth_headers
    )
    assert response.status_code == 400

    content = "header1,header2\nvalue1,value2"
    response = await client.post(
        "/api/v1/portfolio/import",
        files={"file": ("test.csv", content, "text/csv")},
        headers=auth_headers
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_portfolio_import_upserts_positions(client: AsyncClient, auth_headers, monkeypatch):
    from app.core import config
    from app.api.v1 import portfolio as portfolio_api
    from app.services.portfolio_import import import_service

    async def fake_extract_positions_from_rows(rows, providers, timeout_seconds):
        assert providers == [
            {
                "name": "table",
                "base_url": "http://table.example.com",
                "api_key": "table-key",
                "model": "table-cheap",
            }
        ]
        return [
            import_service.PositionItem(ticker="TCB", quantity=100),
            import_service.PositionItem(ticker="VCB", quantity=50),
        ]

    monkeypatch.setattr(portfolio_api, "extract_positions_from_rows", fake_extract_positions_from_rows)
    monkeypatch.setattr(
        config.settings,
        "llm_providers",
        json.dumps(
            [
                {"name": "vision", "base_url": "http://vision.example.com", "api_key": "vision-key", "model": "vision-default"},
                {"name": "table", "base_url": "http://table.example.com", "api_key": "table-key", "model": "table-default"},
            ]
        ),
    )
    monkeypatch.setattr(
        config.settings,
        "llm_task_config",
        json.dumps(
            {
                "position_table_extraction": [{"provider": "table", "model": "table-cheap"}],
                "position_image_extraction": [{"provider": "vision", "model": "vision-pro"}],
            }
        ),
    )

    rows = [["" for _ in range(5)] for _ in range(8)]
    rows.append(["TCB", "buy", "100", "2024-01-01", ""])
    rows.append(["VCB", "buy", "50", "2024-01-02", ""])
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)

    response = await client.post(
        "/api/v1/portfolio/import",
        files={"file": ("import.csv", buffer.getvalue(), "text/csv")},
        data={"broker_id": "vpbanks"},
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["created_count"] == 2
    assert data["updated_count"] == 0

    list_response = await client.get("/api/v1/portfolio/positions", headers=auth_headers)
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 2


@pytest.mark.asyncio
async def test_portfolio_image_import_uses_position_image_task_config(client: AsyncClient, auth_headers, monkeypatch):
    from app.core import config
    from app.api.v1 import portfolio as portfolio_api
    from app.services.llm.llm_client import ImagePositionItem

    async def fake_extract_positions_from_image(file, providers, timeout_seconds):
        assert providers == [
            {
                "name": "vision",
                "base_url": "http://vision.example.com",
                "api_key": "vision-key",
                "model": "vision-pro",
            }
        ]
        return [ImagePositionItem(ticker="SSI", average_cost=25.5, quantity=100)]

    monkeypatch.setattr(portfolio_api, "extract_positions_from_image", fake_extract_positions_from_image)
    monkeypatch.setattr(
        config.settings,
        "llm_providers",
        json.dumps(
            [
                {"name": "vision", "base_url": "http://vision.example.com", "api_key": "vision-key", "model": "vision-default"},
                {"name": "table", "base_url": "http://table.example.com", "api_key": "table-key", "model": "table-default"},
            ]
        ),
    )
    monkeypatch.setattr(
        config.settings,
        "llm_task_config",
        json.dumps(
            {
                "position_image_extraction": [{"provider": "vision", "model": "vision-pro"}],
                "position_table_extraction": [{"provider": "table", "model": "table-cheap"}],
            }
        ),
    )

    response = await client.post(
        "/api/v1/portfolio/import",
        files={"file": ("position.png", b"fake-image", "image/png")},
        data={"broker_id": "vpbanks"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["created_count"] == 1
    assert data["imported_positions"][0]["ticker"] == "SSI"
    assert data["imported_positions"][0]["average_cost"] == 25500


@pytest.mark.asyncio
async def test_portfolio_image_import_deletes_positions_missing_from_screenshots(client: AsyncClient, auth_headers, monkeypatch):
    from app.core import config
    from app.api.v1 import portfolio as portfolio_api
    from app.services.llm.llm_client import ImagePositionItem

    await _create_position(
        client,
        auth_headers,
        {"ticker": "SSI", "quantity": 50, "average_cost": 24000},
    )
    await _create_position(
        client,
        auth_headers,
        {"ticker": "HPG", "quantity": 80, "average_cost": 28000},
    )

    async def fake_extract_positions_from_image(file, providers, timeout_seconds):
        return [
            ImagePositionItem(ticker="SSI", average_cost=25.5, quantity=100),
            ImagePositionItem(ticker="VCI", average_cost=38.2, quantity=40),
        ]

    monkeypatch.setattr(portfolio_api, "extract_positions_from_image", fake_extract_positions_from_image)
    monkeypatch.setattr(config.settings, "llm_providers", json.dumps([
        {"name": "test", "base_url": "http://example.com", "api_key": "test", "model": "test"}
    ]))
    monkeypatch.setattr(config.settings, "llm_task_config", "{}")

    response = await client.post(
        "/api/v1/portfolio/import",
        files={"file": ("positions.png", b"fake-image", "image/png")},
        data={"broker_id": "vpbanks"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["created_count"] == 1
    assert data["updated_count"] == 1
    assert data["deleted_count"] == 1

    list_response = await client.get("/api/v1/portfolio/positions", headers=auth_headers)
    assert list_response.status_code == 200
    positions = {item["ticker"]: item for item in list_response.json()["positions"]}
    assert set(positions) == {"SSI", "VCI"}
    assert positions["SSI"]["quantity"] == 100
    assert positions["SSI"]["average_cost"] == 25500


@pytest.mark.asyncio
async def test_portfolio_spreadsheet_import_keeps_positions_missing_from_import(client: AsyncClient, auth_headers, monkeypatch):
    from app.core import config
    from app.api.v1 import portfolio as portfolio_api
    from app.services.portfolio_import import import_service

    await _create_position(
        client,
        auth_headers,
        {"ticker": "HPG", "quantity": 80, "average_cost": 28000},
    )

    async def fake_extract_positions_from_rows(rows, providers, timeout_seconds):
        return [
            import_service.PositionItem(ticker="TCB", quantity=100),
        ]

    monkeypatch.setattr(portfolio_api, "extract_positions_from_rows", fake_extract_positions_from_rows)
    monkeypatch.setattr(config.settings, "llm_providers", json.dumps([
        {"name": "test", "base_url": "http://example.com", "api_key": "test", "model": "test"}
    ]))
    monkeypatch.setattr(config.settings, "llm_task_config", "{}")

    rows = [["" for _ in range(5)] for _ in range(8)]
    rows.append(["TCB", "buy", "100", "2024-01-01", ""])
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)

    response = await client.post(
        "/api/v1/portfolio/import",
        files={"file": ("import.csv", buffer.getvalue(), "text/csv")},
        data={"broker_id": "vpbanks"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["created_count"] == 1
    assert data["deleted_count"] == 0

    list_response = await client.get("/api/v1/portfolio/positions", headers=auth_headers)
    assert list_response.status_code == 200
    assert {item["ticker"] for item in list_response.json()["positions"]} == {"HPG", "TCB"}


@pytest.mark.asyncio
async def test_portfolio_image_import_keeps_conflicted_ticker_from_stale_deletion(client: AsyncClient, auth_headers, monkeypatch):
    from app.core import config
    from app.api.v1 import portfolio as portfolio_api
    from app.services.llm.llm_client import ImagePositionItem

    await _create_position(
        client,
        auth_headers,
        {"ticker": "VCB", "quantity": 50, "average_cost": 90000},
    )
    await _create_position(
        client,
        auth_headers,
        {"ticker": "HPG", "quantity": 80, "average_cost": 28000},
    )

    async def fake_extract_positions_from_image(file, providers, timeout_seconds):
        if file.filename == "position-1.png":
            return [
                ImagePositionItem(ticker="SSI", average_cost=25.5, quantity=100),
                ImagePositionItem(ticker="VCB", average_cost=88.0, quantity=50),
            ]
        return [ImagePositionItem(ticker="VCB", average_cost=89.0, quantity=50)]

    monkeypatch.setattr(portfolio_api, "extract_positions_from_image", fake_extract_positions_from_image)
    monkeypatch.setattr(config.settings, "llm_providers", json.dumps([
        {"name": "test", "base_url": "http://example.com", "api_key": "test", "model": "test"}
    ]))
    monkeypatch.setattr(config.settings, "llm_task_config", "{}")

    response = await client.post(
        "/api/v1/portfolio/import",
        files=[
            ("file", ("position-1.png", b"fake-image-1", "image/png")),
            ("file", ("position-2.png", b"fake-image-2", "image/png")),
        ],
        data={"broker_id": "vpbanks"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["created_count"] == 1
    assert data["deleted_count"] == 1
    assert data["skipped_count"] == 1

    list_response = await client.get("/api/v1/portfolio/positions", headers=auth_headers)
    assert list_response.status_code == 200
    positions = {item["ticker"]: item for item in list_response.json()["positions"]}
    assert set(positions) == {"SSI", "VCB"}
    assert positions["VCB"]["average_cost"] == 90000
