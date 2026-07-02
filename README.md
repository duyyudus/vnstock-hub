# VNStock Hub

VNStock Hub is a full-stack application for tracking and analyzing Vietnamese stock market data, mutual fund performance, personal portfolios, and curated market news.

## What the App Includes

- Market dashboard for indices, industries, and ticker-level data.
- Screeners for exploring index constituents and symbol-level metrics.
- Fund analytics with performance, allocation, and risk-return visualizations.
- News dashboard with public/default sources, private RSS or crawl sources, blocked-topic filtering, and on-demand LLM summaries.
- User accounts with JWT authentication.
- Bookmark groups for favorite stocks.
- Portfolio tracking with CRUD, CSV workflows, and LLM-assisted import.
- Trading views with position tracking and P&L details for signed-in users.
- Admin control panel for running and monitoring sync jobs, scheduler settings, and news ingestion.

<details>
<summary><h2>Product Tour</h2></summary>

The dashboard opens with live Vietnamese market index cards, index constituent analytics, and multiple chart modes for comparing price change, traded value, liquidity, and foreign flow.

![Stocks dashboard with VN30 market overview](docs/screenshots/vnstock-hub-stocks.png)

Screeners rank index universes with valuation, margin trend, liquidity/volatility, and financial health models. Rows are computed from the selected benchmark and industry filters, with fallback indicators shown when data is partial.

![Valuation screener for the VN100 universe](docs/screenshots/vnstock-hub-screeners.png)

News combines the public source pack with semantic summaries, timeframe filters, article clustering, and private source/topic controls for signed-in users.

![News dashboard with quick-glance digest](docs/screenshots/vnstock-hub-news.png)

Fund analytics compare stock, bond, and balanced funds against VN-Index with growth, risk-return, and yearly heatmap views.

![Fund performance comparison chart](docs/screenshots/vnstock-hub-funds.png)

Signed-in users get portfolio tracking with current prices, market value, CSV export/import workflows, and per-position P&L.

![Authenticated portfolio positions](docs/screenshots/vnstock-hub-portfolio.png)

Trading views keep short-term positions separate from the long-term portfolio, with broker account filtering and open P&L.

![Authenticated trading positions](docs/screenshots/vnstock-hub-trading.png)

The admin surface is protected by authentication and hosts sync jobs, scheduler controls, runtime settings, and news ingestion monitoring.

![Authenticated admin control panel](docs/screenshots/vnstock-hub-admin.png)

</details>

## Repository Layout

```text
vnstock-hub/
├── backend/     # FastAPI API service + async DB + sync workers
├── docs/        # subsystem and implementation notes
├── frontend/    # React dashboard (Vite + TypeScript)
├── run-server   # helper script to start backend
├── run-ui       # helper script to start frontend
└── README.md
```

## Architecture Overview

### Backend (`backend/`)

- FastAPI app with versioned APIs under `/api/v1`.
- Async SQLAlchemy + PostgreSQL with Alembic migrations.
- Modular service layer in `app/services/vnstock_service`:
  - stocks, funds, finance, company, history
  - sync workers for price, finance, and company data
  - rate-limit aware workflows and status reporting
- Dedicated news subsystem in `app/services/news`:
  - public source seeding from `backend/news_sources.yaml`
  - RSS discovery, crawl-source validation, and background ingestion
  - semantic classification, per-user topic blocking, and stored summaries
- Additional domains:
  - auth (`/auth`), bookmarks (`/bookmarks`), portfolio (`/portfolio`), trading (`/trading`), app info (`/info`), sync admin (`/sync`), news (`/news`)
- App startup initializes both market sync workers and the news ingestion worker.

### Frontend (`frontend/`)

- React 19 + TypeScript + Vite.
- Feature-based structure (`src/features`).
- Dashboard tabs:
  - Indices (table/growth/comparison/risk-return views)
  - Screeners (filterable symbol exploration)
  - News (public feed, private source setup, semantic filtering, summaries)
  - Funds (performance + holdings + heatmaps)
  - Portfolio (authenticated users)
  - Trading (authenticated users)
- Admin page at `/admin` for sync operations, scheduler controls, settings, and news monitoring.
- Central API client in `src/api/stockApi.ts`.

## Quick Start

### 1) Backend

```bash
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000
```

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

Or from repository root:

```bash
./run-server
./run-ui
```

## Environment Configuration

### Backend (`backend/.env`)

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/vnstock_hub
API_V1_PREFIX=/api/v1
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
SETTINGS_YAML_PATH=./settings.yaml
NEWS_SOURCES_YAML_PATH=./news_sources.yaml
JWT_SECRET_KEY=change-me
LLM_PROVIDERS=[]
LLM_REQUEST_TIMEOUT_SECONDS=30
SYNC_ADMIN_EMAILS=["admin@example.com"]
NEWS_INGESTION_ENABLED=true
NEWS_POLL_INTERVAL_SECONDS=120
NEWS_DEFAULT_POLL_INTERVAL_MINUTES=30
NEWS_INGESTION_BATCH_SIZE=5
```

`LLM_PROVIDERS` is shared by portfolio import and the news semantic/summarization flows. `NEWS_SOURCES_YAML_PATH` points to the seeded public news source pack loaded by the backend worker.

### Frontend (`frontend/.env`)

```env
VITE_API_BASE_URL=http://localhost:8000
```

## Testing

Backend tests:

```bash
cd backend
uv run pytest
```

Frontend currently uses linting as the main static check:

```bash
cd frontend
npm run lint
npm run build
```

Focused backend coverage for the new news subsystem lives in:

```bash
cd backend
uv run pytest tests/test_news_api.py tests/test_news_service.py tests/test_news_semantics.py
```

## API Docs

Run backend and open:

- `http://localhost:8000/docs`

## Vendored API Docs

The vendored `vnstock_alt` and `vnstock_data_alt` packages also have a local,
generated docs site under `backend/docs/generated`.

Generate the source-derived docs:

```bash
cd backend
uv run python scripts/generate_vnstock_api_docs.py
```

Refresh live samples, then regenerate docs:

```bash
cd backend
uv run python scripts/capture_vnstock_api_samples.py
uv run python scripts/generate_vnstock_api_docs.py
```

Browse the local docs site:

```bash
cd backend
uv run mkdocs serve -f mkdocs.yml
```

Build static HTML output:

```bash
cd backend
uv run mkdocs build -f mkdocs.yml
```

## Module Docs

- News subsystem v1: `docs/news_subsystem.md`
- Backend details: `backend/README.md`
- Frontend details: `frontend/README.md`
