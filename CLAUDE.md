# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VNStock Hub is a full-stack web application for tracking and analyzing the Vietnamese stock market. It provides real-time dashboards for market indices (VN-100, VN-30, etc.), stock data, financial statements, and company information.

**Tech Stack:**
- **Backend:** FastAPI (Python 3.12+), PostgreSQL with SQLAlchemy (async), Alembic migrations, `vnstock` library
- **Frontend:** React 19, TypeScript, Vite, TailwindCSS, DaisyUI, Axios
- **Package Management:** Backend uses `uv`, Frontend uses `npm`

## Development Commands

### Quick Start
```bash
# Run backend server (from project root)
./run-server

# Run frontend UI (from project root)
./run-ui
```

### Backend

**Setup:**
```bash
cd backend
uv sync                                    # Install dependencies
uv run alembic upgrade head                # Run database migrations
```

**Running:**
```bash
cd backend
uv run uvicorn app.main:app --reload --port 8000
```
API docs available at http://localhost:8000/docs

**Testing:**
```bash
cd backend
uv run python tests/test_vn100_fetch.py   # Run standalone test scripts
uv run pytest                              # Run all tests
```

**Database Migrations:**
```bash
cd backend
uv run alembic revision --autogenerate -m "description"  # Create migration
uv run alembic upgrade head                              # Apply migrations
uv run alembic downgrade -1                              # Rollback one migration
```

### Frontend

**Setup:**
```bash
cd frontend
npm install
```

**Running:**
```bash
cd frontend
npm run dev      # Start dev server (http://localhost:5173)
npm run build    # Build for production
npm run preview  # Preview production build
npm run lint     # Run ESLint
```

## Architecture

### Backend Architecture

**Entry Point:** `backend/app/main.py`
- FastAPI app initialization with CORS middleware
- Lifespan events handle database table creation and index synchronization on startup
- All API routes are versioned under `/api/v1`

**Layered Structure:**
- `app/api/v1/` - API endpoints with Pydantic request/response models
- `app/services/` - Business logic layer (e.g., `vnstock_service.py` handles all stock data fetching)
- `app/db/` - Database layer with SQLAlchemy models and async session management
- `app/core/` - Configuration (`config.py` uses Pydantic settings from `.env`)

**Key Services:**
- `vnstock_service.py` - Singleton service that wraps the `vnstock` library, handles data fetching, transformation, and caching for indices, stocks, financial statements, and company data
- Database uses async SQLAlchemy with `asyncpg` driver

**Data Models (app/db/models.py):**
- `StockCompany` - Company metadata (name, charter capital, PE ratio)
- `StockIndex` - Market indices metadata
- `StockDailyPrice` - Historical OHLCV data

**API Patterns:**
- FastAPI dependency injection is used throughout (see `app/core/deps.py`)
- All endpoints return structured Pydantic models
- The `vnstock_service` singleton is accessed directly in endpoints (not via DI)

### Frontend Architecture

**Entry Point:** `frontend/src/main.tsx` → `App.tsx`

**Feature-Based Structure:**
- `src/features/dashboard/` - Complete dashboard feature with all components
  - `Dashboard.tsx` - Main dashboard container
  - `IndexTable.tsx` - Stock table with sorting, searching, and row click handlers
  - `CompanyFinancialPopup.tsx` - Draggable popup for displaying financial data
  - `IndexBanners.tsx` - Market index summary banners
  - `IndexSelector.tsx` & `IndustrySelector.tsx` - Dropdown selectors
  - `indexConfig.ts` - Configuration for available indices

**API Layer:**
- `src/api/stockApi.ts` - Centralized API client with Axios
- All API calls go through this module
- TypeScript interfaces define API request/response shapes

**Component Patterns:**
- Feature-based: Components related to a specific feature live together in `src/features/[feature_name]`
- Reusable UI components in `src/components/` (e.g., `TabNavigation.tsx`)
- DaisyUI components are preferred for consistent styling

**State Management:**
- React hooks (`useState`, `useEffect`) for local state
- No global state management library (Redux, Zustand, etc.)
- API data is fetched and managed within feature components

## Environment Configuration

### Backend `.env`
Required variables (defaults in `backend/app/core/config.py`):
```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/vnstock_hub
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
VN100_LIMIT=100
VN30_LIMIT=30
```

### Frontend `.env`
```env
VITE_API_BASE_URL=http://localhost:8000
```

## Key Conventions

### Backend
- All package management uses `uv` (not pip or poetry)
- Database operations are async throughout
- API versioning is strict: all endpoints under `/api/v1`
- The `vnstock` library is wrapped in a service layer to centralize data fetching logic
- On startup, the app syncs available indices from vnstock into the database

### Frontend
- Feature-based organization: keep related components, hooks, and logic together
- All API interactions through `src/api/stockApi.ts`
- TailwindCSS + DaisyUI for styling
- TypeScript strict mode enabled

## Data Flow

1. **Startup:** Backend fetches and caches available indices from vnstock into PostgreSQL
2. **Frontend Request:** User selects an index → Frontend calls `stockApi.getIndexStocks(symbol)`
3. **Backend Processing:** `vnstock_service` fetches live data from vnstock library, enriches with company metadata from DB
4. **Response:** Structured data returned with stock prices, market caps, PE ratios, price changes
5. **Frontend Display:** `IndexTable` component renders data with sorting, filtering, and interactive popups for detailed financial views
