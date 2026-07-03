# VNStock Hub Backend

FastAPI backend for VNStock Hub, providing market/fund data APIs, user features (auth, bookmarks, portfolio), and background synchronization jobs.

## Tech Stack

- FastAPI (Python 3.12+)
- uv (dependency + environment management)
- PostgreSQL + SQLAlchemy async
- Alembic migrations
- Pydantic v2
- JWT auth (`python-jose`) + password hashing
- `vnstock` data source wrapper (`app/services/vnstock_service`)

## Current Capabilities

- Stock endpoints for index values, index/industry constituents, quotes, finance datasets, company datasets, and historical price/volume.
- Fund endpoints for listing, aggregated performance, NAV report, top holdings, industry holdings, and asset holdings.
- Auth endpoints for register/login plus user export preference settings.
- Bookmark groups for authenticated users (group CRUD + per-group stocks).
- Portfolio position management (CRUD), CSV export, CSV overwrite import, and LLM-assisted import from file/image flows.
- Sync status + admin controls for price sync, price audit, repair, finance sync, and company sync.
- Startup lifecycle that syncs indices and starts/stops background workers.

## Project Structure

```text
backend/
├── alembic/
├── app/
│   ├── api/v1/            # auth, bookmarks, funds, portfolio, stocks, sync
│   ├── core/              # config, deps, security, exceptions, logging
│   ├── db/                # async engine/session + SQLAlchemy models
│   ├── services/
│   │   ├── vnstock_service/   # facade + domain services + sync workers
│   │   ├── portfolio_import/  # broker crop profiles + extraction helpers
│   │   └── llm/               # OpenAI-compatible LLM clients
│   └── main.py
├── tests/
├── settings.yaml
└── pyproject.toml
```

## Setup

### Prerequisites

- Python 3.12+
- `uv`
- PostgreSQL

### Install + migrate

```bash
cd backend
uv sync
uv run alembic upgrade head
```

Note: app startup also runs `Base.metadata.create_all()` for dev convenience, but Alembic remains the source of truth.

### Environment variables

Common secrets and deployment wiring in `backend/.env`:

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/vnstock_hub
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
SYNC_ADMIN_EMAILS=["admin@example.com"]
JWT_SECRET_KEY=change-me
LLM_PROVIDERS='[]'
```

Non-secret backend defaults live in `backend/settings.yaml`, including API
prefix, sync tuning, auth token lifetime, and LLM task routing. Deployment
policy like CORS origins and sync admin allowlists stays in env. Env vars still
override YAML when needed.

`LLM_PROVIDERS` is the shared provider catalog and stays in env because provider entries can contain API keys. `llm.task_config` in `settings.yaml` optionally maps task names such as `position_image_extraction` and `position_table_extraction` to ordered provider/model fallback chains. If a task is missing, the backend uses `default`; if `default` is also missing, it falls back to the legacy `LLM_PROVIDERS` list order.

### `settings.yaml`

Backend defaults and broker import presets are loaded from `backend/settings.yaml`:

```yaml
app:
  api_v1_prefix: /api/v1
sync:
  target_rpm: 150
  max_workers: 10
  chunk_days: 1825
brokers:
  - id: vpbanks
    name: VPBank Securities
    sheet: Sheet1
    top_left: A9
    bottom_right: E
```

## Run

```bash
cd backend
uv run uvicorn app.main:app --reload --port 8000
```

- Swagger UI: `http://localhost:8000/docs`
- Quick start from repo root: `./run-server`

## Testing

```bash
cd backend
uv run pytest
```

Test config is isolated via `backend/.env.test`.

## API Surface (`/api/v1`)

### Auth

- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/settings`
- `PATCH /auth/settings`

### Stocks

- `GET /stocks/index-values`
- `GET /stocks/indices`
- `GET /stocks/index/{index_symbol}`
- `GET /stocks/industries`
- `GET /stocks/industry/{industry_name}`
- `POST /stocks/quotes`
- `GET /stocks/finance/{symbol}/income-statement`
- `GET /stocks/finance/{symbol}/balance-sheet`
- `GET /stocks/finance/{symbol}/cash-flow`
- `GET /stocks/finance/{symbol}/ratios`
- `GET /stocks/company/{symbol}/overview`
- `GET /stocks/company/{symbol}/shareholders`
- `GET /stocks/company/{symbol}/officers`
- `GET /stocks/company/{symbol}/subsidiaries`
- `GET /stocks/history/{symbol}/volume`
- `GET /stocks/history/{symbol}/price`
- `POST /stocks/weekly-prices`

### Funds

- `GET /funds/listing`
- `GET /funds/performance`
- `GET /funds/{symbol}/nav-report`
- `GET /funds/{symbol}/top-holding`
- `GET /funds/{symbol}/industry-holding`
- `GET /funds/{symbol}/asset-holding`

### Portfolio (auth required)

- `GET /portfolio/positions`
- `POST /portfolio/positions`
- `PATCH /portfolio/positions/{position_id}`
- `DELETE /portfolio/positions/{position_id}`
- `GET /portfolio/export/csv`
- `POST /portfolio/import/fresh`
- `GET /portfolio/import/brokers`
- `POST /portfolio/import`

### Bookmarks (auth required)

- `GET /bookmarks/groups`
- `POST /bookmarks/groups`
- `PATCH /bookmarks/groups/{group_id}`
- `DELETE /bookmarks/groups/{group_id}`
- `GET /bookmarks/groups/{group_id}/stocks`
- `POST /bookmarks/groups/{group_id}/stocks`
- `DELETE /bookmarks/groups/{group_id}/stocks/{ticker}`

### Sync

- `GET /sync/status`
- `POST /sync/prices/run` (admin)
- `POST /sync/prices/audit/run` (admin)
- `POST /sync/prices/repair/run` (admin)
- `POST /sync/finance/run` (admin)
- `POST /sync/company/run` (admin)

### Non-versioned utilities

- `GET /`
- `GET /health`
