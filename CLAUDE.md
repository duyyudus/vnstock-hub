# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VNStock Hub is a full-stack web application for tracking and analyzing the Vietnamese stock market. It provides real-time dashboards for market indices (VN-100, VN-30, etc.), mutual fund performance analysis, stock data, financial statements, and comprehensive company information with interactive visualizations.

**Tech Stack:**

- **Backend:** FastAPI (Python 3.12+), PostgreSQL with SQLAlchemy (async), Alembic migrations, `vnstock` library (with API key support)
- **Frontend:** React 19, TypeScript, Vite, TailwindCSS, DaisyUI, Axios, Recharts
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

API docs available at <http://localhost:8000/docs>

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
- Three routers registered: stocks, funds, sync

**Layered Structure:**

- `app/api/v1/` - API endpoints with Pydantic request/response models
  - `stocks.py` - 19 endpoints for indices, industries, stock data, financial statements, company info, historical data
  - `funds.py` - 6 endpoints for fund listings, performance, NAV reports, and holdings
  - `sync.py` - 1 endpoint for background sync status tracking
- `app/services/` - Business logic layer
  - `vnstock_service/` - Modular service package (~3800 lines) with facade pattern for all data fetching operations
    - `__init__.py` - VnstockService facade that composes sub-services
    - `core.py` - Shared utilities, retry logic, circuit breaker integration, thread pool executors
    - `models.py` - Data models (IndexValue, StockInfo, etc.)
    - `indices.py` - IndicesService for market indices operations
    - `stocks.py` - StocksService for stock data and industry operations
    - `stock_metadata.py` - StockMetadataService for stock metadata operations
    - `history.py` - HistoryService for historical price and volume data
    - `finance.py` - FinanceService for financial statements and ratios
    - `company.py` - CompanyService for company overview, shareholders, officers, subsidiaries
    - `funds.py` - FundsService for mutual fund listings, performance, NAV, holdings
    - `symbols.py` - Symbol constants and validation
  - `sync_status.py` - Global sync status tracking for background operations
- `app/db/` - Database layer with SQLAlchemy models and async session management
- `app/core/` - Core infrastructure
  - `config.py` - Configuration using Pydantic settings from `.env`
  - `circuit_breaker.py` - Thread-safe circuit breaker for API rate limit management
  - `logging_config.py` - Logging configuration with separate loggers for main and background tasks

**Key Services:**

- `vnstock_service/` package - Modular service wrapping the `vnstock` library using facade pattern
  - **VnstockService** (facade) - Composes sub-services and provides unified interface
  - **Sub-services** - Each handles a specific domain (indices, stocks, funds, finance, company, history, metadata)
  - **Core utilities** - Retry logic with exponential backoff, circuit breaker integration
  - **Thread pool executors** - Separate pools for frontend (user-facing) and background (sync) operations
  - **Rate limit management** - Circuit breaker prevents cascading failures, fail-fast on rate limits
  - **Caching** - In-memory caching for indices, industries, and fund data with TTL
  - **Database integration** - Company enrichment using async SQLAlchemy
  - **Background sync** - Fund performance data synchronization (6-hour intervals)
- `sync_status.py` - Global singleton for tracking background sync operations and rate limit status
- `circuit_breaker.py` - Thread-safe circuit breaker for API rate limit management with CLOSED/OPEN/HALF_OPEN states
- Database uses async SQLAlchemy with `asyncpg` driver

**Data Models (app/db/models.py):**

- `StockCompany` - Company metadata (name, charter capital, PE ratio)
- `StockIndex` - Market indices metadata (symbol, name, description, group)
- `StockDailyPrice` - Historical OHLCV data with unique constraint on (symbol, date)
- `FundNav` - Historical NAV data for mutual funds with unique constraint on (symbol, date)

**API Patterns:**

- FastAPI dependency injection is used throughout (see `app/core/deps.py`)
- All endpoints return structured Pydantic models
- The `vnstock_service` singleton is accessed directly in endpoints (not via DI)
- Rate limiting is handled globally with retry logic and status tracking

### Frontend Architecture

**Entry Point:** `frontend/src/main.tsx` → `App.tsx`

**Feature-Based Structure:**

- `src/features/dashboard/` - Complete dashboard feature organized into subdirectories
  - **`Dashboard.tsx`** - Main container with tab navigation (Indices/Funds), global popup state management, z-index coordination

  - **`indices/`** - Stock indices and industry tracking
    - `IndicesTab.tsx` - Container managing index/industry selection, table/chart view toggle
    - `IndexSelector.tsx` & `IndustrySelector.tsx` - Dropdown selectors
    - `StocksTable.tsx` - Sortable, searchable stock table with click handlers for popups
    - `StocksGrowthChart.tsx` - Weekly price growth visualization with benchmark comparison
    - `indexConfig.ts` - Index configuration definitions

  - **`funds/`** - Mutual fund performance tracking and analysis
    - `FundsTab.tsx` - Main container with aggregate performance charts and fund selector
    - `FundSelector.tsx` - Individual fund selection dropdown
    - `FundInfoCard.tsx` - Fund metadata and key metrics display
    - `NavReportChart.tsx` - NAV history chart
    - `TopHoldingChart.tsx` - Top stock holdings visualization
    - `IndustryHoldingChart.tsx` - Industry allocation breakdown
    - `AssetHoldingChart.tsx` - Asset type allocation
    - `CumulativeGrowthChart.tsx` - Cumulative NAV growth with benchmark
    - `RiskReturnScatterPlot.tsx` - Risk vs return scatter with Capital Market Line
    - `PeriodicReturnHeatmap.tsx` - Returns heatmap by period and year

  - **`banner/`** - Market overview
    - `IndexBanners.tsx` - Major index values (VNINDEX, HNXINDEX, UPCOMINDEX, VN30, HNX30) with 5-minute refresh

  - **`components/`** - Shared dashboard components
    - `CompanyFinancialPopup.tsx` - Draggable popup for financial statements, ratios, shareholders, officers, subsidiaries
    - `VolumeChartPopup.tsx` - Draggable popup for volume history chart

**API Layer:**

- `src/api/stockApi.ts` - Centralized API client with Axios (395 lines)
- 25+ TypeScript interfaces for API request/response types
- Complete methods for stocks, indices, industries, financials, company data, funds, historical data
- TypeScript interfaces define API request/response shapes

**Shared Components:**

- `src/components/TabNavigation.tsx` - Vertical tab navigation using DaisyUI
- `src/components/SyncIndicator.tsx` - Loading spinner for background sync operations

**Component Patterns:**

- Feature-based with subdirectory organization: `src/features/dashboard/{indices|funds|banner|components}/`
- Window-based event delegation for popup triggers (global coordination)
- Recharts library for all data visualizations
- DaisyUI components are preferred for consistent styling

**State Management:**

- React hooks (`useState`, `useEffect`) for local state
- No global state management library (Redux, Zustand, etc.)
- API data is fetched and managed within feature components
- Global popup state managed in Dashboard.tsx and passed down

## Environment Configuration

### Backend `.env`

Required variables (defaults in `backend/app/core/config.py`):

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/vnstock_hub
API_V1_PREFIX=/api/v1
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
VNSTOCK_API_KEY=your_vnstock_api_key_here  # For vnstock data source access
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
- The `vnstock` library is wrapped in a modular service layer using facade pattern
  - Sub-services organized by domain (indices, stocks, funds, finance, company, history)
  - VnstockService facade provides unified interface
- On startup, the app syncs available indices from vnstock into the database
- Circuit breaker pattern for API rate limit management (CLOSED/OPEN/HALF_OPEN states)
  - Prevents cascading failures by failing fast when rate limited
  - Automatic recovery with configurable timeout
- Retry logic with exponential backoff for background sync operations
- Separate thread pool executors for frontend (user-facing) and background (sync) operations
- Background sync operations for fund performance data (6-hour intervals to reduce API calls)
- Global sync status tracking via `sync_status.py` singleton
- In-memory caching for indices, industries, and fund data with TTL management

### Frontend

- Feature-based organization with subdirectory structure: `features/dashboard/{indices|funds|banner|components}/`
- Tab-based navigation: separate views for Indices and Funds
- All API interactions through `src/api/stockApi.ts`
- Recharts library for all data visualizations (growth charts, scatter plots, heatmaps)
- Global event delegation via window object for popup coordination
- Automatic retry on rate limit errors with user feedback
- TailwindCSS + DaisyUI for styling
- TypeScript strict mode enabled

## Data Flow

### Stock Indices Flow

1. **Startup:** Backend fetches and caches available indices from vnstock into PostgreSQL
2. **Frontend Request:** User selects an index → Frontend calls `stockApi.getIndexStocks(symbol)`
3. **Circuit Breaker Check:** System checks if API is rate limited before proceeding
4. **Backend Processing:** `vnstock_service` fetches live data from vnstock library (fail-fast on rate limit), enriches with company metadata from DB
5. **Response:** Structured data returned with stock prices, market caps, PE ratios, price changes
6. **Frontend Display:** `StocksTable` or `StocksGrowthChart` component renders data with sorting, filtering, and interactive popups for financial views

### Mutual Funds Flow

1. **Background Sync:** `vnstock_service` syncs fund performance data every 6 hours to reduce API calls (uses async retry with backoff)
2. **Frontend Request:** User navigates to Funds tab → Frontend calls `stockApi.getFundPerformance()`
3. **Backend Processing:** Returns cached/synced fund performance data with benchmarks
4. **Frontend Display:** Multiple visualizations (growth charts, risk-return scatter, heatmaps, holdings)
5. **Rate Limit Handling:** Circuit breaker opens on rate limit, preventing further API calls. `sync_status` tracks state and frontend auto-retries after circuit breaker recovery timeout

### Company Details Flow

1. **User Action:** Click on stock ticker in table
2. **Popup Trigger:** Window event triggers `CompanyFinancialPopup` with symbol
3. **Circuit Breaker Check:** Each API request checks if circuit breaker allows the call
4. **Multiple API Calls:** Frontend fetches income statement, balance sheet, cash flow, ratios, shareholders, officers, subsidiaries in parallel
5. **Popup Display:** Draggable popup with tabbed financial data and company information

### Volume Chart Flow

1. **User Action:** Click on volume icon in stock table
2. **Popup Trigger:** Window event triggers `VolumeChartPopup` with symbol
3. **Circuit Breaker Check:** System checks if API is rate limited before proceeding
4. **API Call:** Frontend fetches volume history data
5. **Chart Display:** Draggable popup with Recharts volume visualization
