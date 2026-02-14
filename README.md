# VNStock Hub

VNStock Hub is a full-stack application for tracking and analyzing Vietnamese stock market data, mutual fund performance, and personal portfolios.

## What the App Includes

- Market dashboard for indices, industries, and ticker-level data.
- Fund analytics with performance, allocation, and risk-return visualizations.
- User accounts with JWT authentication.
- Bookmark groups for favorite stocks.
- Portfolio tracking with CRUD, CSV workflows, and LLM-assisted import.
- Admin control panel for running and monitoring background sync jobs.

## Repository Layout

```text
vnstock-hub/
├── backend/     # FastAPI API service + async DB + sync workers
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
- Additional domains:
  - auth (`/auth`), bookmarks (`/bookmarks`), portfolio (`/portfolio`), sync admin (`/sync`)

### Frontend (`frontend/`)

- React 19 + TypeScript + Vite.
- Feature-based structure (`src/features`).
- Dashboard tabs:
  - Indices (table/growth/comparison/risk-return views)
  - Funds (performance + holdings + heatmaps)
  - Portfolio (authenticated users)
- Admin page at `/admin` for sync operations.
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
VNSTOCK_API_KEY=
JWT_SECRET_KEY=change-me
SYNC_ADMIN_EMAILS=["admin@example.com"]
```

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
```

## API Docs

Run backend and open:

- `http://localhost:8000/docs`

## Module Docs

- Backend details: `backend/README.md`
- Frontend details: `frontend/README.md`
