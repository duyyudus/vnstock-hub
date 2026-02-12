# VNStock Hub - Backend

This is the backend component of the VNStock Hub, a web application for tracking and analyzing the Vietnam stock market. It provides a RESTful API built with FastAPI, leveraging the `vnstock` library for data retrieval and PostgreSQL for data persistence.

## 🚀 Tech Stack

- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python 3.12+)
- **Package Management:** [uv](https://github.com/astral-sh/uv)
- **Database:** PostgreSQL with [SQLAlchemy](https://www.sqlalchemy.org/) ORM (Async)
- **Migrations:** [Alembic](https://alembic.sqlalchemy.org/)
- **Data Source:** [vnstock](https://github.com/thinh-vu/vnstock) library
- **Validation:** Pydantic v2
- **Security:** JWT Authentication (`python-jose`) + password hashing (`passlib`/bcrypt)

## 📦 Project Structure

```text
backend/
├── alembic/            # Database migrations
├── app/
│   ├── api/v1/         # API endpoints (Auth, Stocks, Funds, Portfolio, Sync)
│   ├── core/           # Config, auth deps, security, logging, circuit breaker
│   ├── db/             # Database session and SQLAlchemy models
│   ├── services/       # Business logic
│   │   ├── vnstock_service/     # Modular vnstock wrapper (indices/stocks/history/sync/funds/...)
│   │   ├── portfolio_import/    # Broker crop profiles + LLM-backed extraction helpers
│   │   └── llm/                 # OpenAI-compatible LLM client(s)
│   └── main.py         # Application entry point
├── tests/              # Pytest suite for APIs and Services
├── pyproject.toml      # Project dependencies and metadata
└── uv.lock             # Lockfile for reproducible builds
```

## 🛠️ Getting Started

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) installed
- PostgreSQL

### Installation

1. Clone the repository and navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Sync dependencies using `uv`:
   ```bash
   uv sync
   ```
3. Create a `.env` file (see `.env.example`).
4. Run migrations (recommended):
   ```bash
   uv run alembic upgrade head
   ```

Note: the app also calls `Base.metadata.create_all()` on startup to ensure tables exist in dev, but Alembic migrations are still the source of truth for schema changes.

### settings.yaml (Non-environmental config)

Broker import profiles live in `backend/settings.yaml`:
```yaml
brokers:
  - id: vpbanks
    name: VPBank Securities
    sheet: Sheet1
    top_left: A9
    bottom_right: E
```

### LLM Provider Configuration (Portfolio Import)

Set `LLM_PROVIDERS` in `.env` as a JSON array (ordered fallback):
```bash
LLM_PROVIDERS='[
  {"name":"gemini","base_url":"https://generativelanguage.googleapis.com/v1beta/openai","api_key":"YOUR_KEY","model":"gpt-4o-mini"},
  {"name":"openrouter","base_url":"https://openrouter.ai/api/v1","api_key":"YOUR_KEY","model":"openai/gpt-4o-mini"}
]'
LLM_REQUEST_TIMEOUT_SECONDS=30
```

### Sync Runtime Configuration

Background sync workers (price sync + finance sync) read these environment variables:

```bash
SYNC_TARGET_RPM=150
SYNC_MAX_WORKERS=10
SYNC_CHUNK_DAYS=1825
SYNC_RATE_LIMIT_FIXED_WAIT_SECONDS=30
SYNC_RATE_LIMIT_MAX_WAIT_SECONDS=1200
```

Admin-only sync endpoints are guarded by an allowlist:

```bash
SYNC_ADMIN_EMAILS='["you@example.com"]'
```

### Running the Application

Start the development server:
```bash
uv run uvicorn app.main:app --reload --port 8000
```
Interactive API docs: `http://localhost:8000/docs`

From repo root you can also use:
```bash
./run-server
```

## 🔌 API Endpoints (v1)

Base prefix: `/api/v1`

- **Auth**
   - `POST /auth/register`
   - `POST /auth/login`

- **Stocks**
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

- **Funds**
   - `GET /funds/listing`
   - `GET /funds/performance`
   - `GET /funds/{symbol}/nav-report`
   - `GET /funds/{symbol}/top-holding`
   - `GET /funds/{symbol}/industry-holding`
   - `GET /funds/{symbol}/asset-holding`

- **Portfolio** (requires `Authorization: Bearer <token>`)
   - `GET /portfolio/positions`
   - `POST /portfolio/positions`
   - `PATCH /portfolio/positions/{position_id}`
   - `DELETE /portfolio/positions/{position_id}`
   - `GET /portfolio/export/csv`
   - `POST /portfolio/import/fresh` (CSV overwrite import)
   - `GET /portfolio/import/brokers`
   - `POST /portfolio/import` (LLM-assisted import from `.csv` / `.xlsx` crop or from image uploads)

- **Bookmarks** (requires `Authorization: Bearer <token>`)
   - `GET /bookmarks/groups`
   - `POST /bookmarks/groups`
   - `PATCH /bookmarks/groups/{group_id}`
   - `DELETE /bookmarks/groups/{group_id}`
   - `GET /bookmarks/groups/{group_id}/stocks`
   - `POST /bookmarks/groups/{group_id}/stocks`
   - `DELETE /bookmarks/groups/{group_id}/stocks/{ticker}`

- **Sync**
   - `GET /sync/status`
   - `POST /sync/prices/run` (admin)
   - `POST /sync/prices/audit/run` (admin)
   - `POST /sync/prices/repair/run` (admin)
   - `POST /sync/finance/run` (admin)

Non-versioned utility endpoints:

- `GET /` (API info)
- `GET /health`

## 🧪 Features

- **Market Data APIs:** Indices, index constituents, industry classification, quotes, and historical price/volume.
- **Company & Finance Data:** Income statement, balance sheet, cash flow, ratios, and company profile datasets.
- **Fund Analytics:** Fund listing, cached performance comparison, NAV report, and holdings breakdowns.
- **User Features:** JWT auth, portfolio positions (CRUD + CSV export/import), and bookmark groups.
- **Background Sync:** Startup index sync + background workers for price sync and finance dataset sync (admin-controlled).
