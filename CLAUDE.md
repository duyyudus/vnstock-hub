# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VNStock Hub is a full-stack web application for tracking and analyzing the Vietnamese stock market. It provides real-time dashboards for market indices (VN-100, VN-30, etc.), mutual fund performance analysis, stock data, financial statements, comprehensive company information, stock screeners, and portfolio management with interactive visualizations.

**Tech Stack:**

- **Backend:** FastAPI (Python 3.12+), PostgreSQL with SQLAlchemy (async), Alembic migrations, `vnstock` library (with API key support), JWT authentication
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

**Docs (vendored API reference):**

```bash
cd backend
uv run mkdocs serve   # Starts at http://127.0.0.1:8001
```

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

### Docker

```bash
# Development (backend :8020, frontend :3030)
docker compose up

# Build and push production images
./docker-build-push-prod           # Linux
./docker-build-push-prod-mac       # macOS (uses socat relay for registry access)
```

## Architecture

### Backend Architecture

**Entry Point:** `backend/app/main.py`

- FastAPI app initialization with CORS middleware
- Lifespan events handle database table creation and index synchronization on startup
- All API routes are versioned under `/api/v1`
- Six routers registered: stocks, funds, sync, auth, bookmarks, portfolio

**Layered Structure:**

- `app/api/v1/` - API endpoints with Pydantic request/response models
  - `stocks.py` - 19 endpoints for indices, industries, stock data, financial statements, company info, historical data
  - `funds.py` - 6 endpoints for fund listings, performance, NAV reports, and holdings
  - `sync.py` - 6 endpoints for sync status and triggering history/finance/company sync jobs (admin-only)
  - `auth.py` - 4 endpoints for login, register, token refresh, and user profile
  - `bookmarks.py` - 7 endpoints for bookmark group and stock management
  - `portfolio.py` - 8 endpoints for portfolio positions, CSV import/export, and broker import
- `app/services/` - Business logic layer
  - `vnstock_service/` - Modular service package (~7700 lines) with facade pattern for all data fetching operations
    - `__init__.py` - VnstockService facade that composes sub-services
    - `core.py` - Shared utilities, retry logic, circuit breaker integration, thread pool executors
    - `models.py` - Data models (IndexValue, StockInfo, etc.)
    - `indices.py` - IndicesService for market indices operations
    - `stocks.py` - StocksService for stock data and industry operations
    - `stock_metadata.py` - StockMetadataService for stock metadata operations
    - `history.py` - HistoryService for historical price and volume data (on-demand sync trigger)
    - `history_sync.py` - HistorySyncService for full/incremental/audit/repair sync jobs
    - `finance.py` - FinanceService for financial statements and ratios (cache-aside pattern)
    - `finance_sync.py` - FinanceDataSyncService for background financial data sync
    - `company.py` - CompanyService for company overview, shareholders, officers, subsidiaries
    - `company_sync.py` - CompanyDataSyncService for background company data sync
    - `funds.py` - FundsService for mutual fund listings, performance, NAV, holdings
    - `symbols.py` - Symbol constants and validation
    - `rate_limit_pause.py` - Rate limit pause/resume utility
  - `auth_service.py` - User authentication helpers
  - `sync_status.py` - Global singleton tracking 4 background sync jobs (history, finance, company, funds) with per-symbol progress
  - `llm/` - LLM client integration (OpenAI-compatible) for portfolio import parsing
  - `portfolio_import/` - Broker-specific portfolio import parsers
- `app/db/` - Database layer with SQLAlchemy models and async session management
- `app/core/` - Core infrastructure
  - `config.py` - Configuration using Pydantic settings from `.env`
  - `circuit_breaker.py` - Thread-safe circuit breaker for API rate limit management
  - `logging_config.py` - Logging configuration with separate loggers for main and background tasks
  - `security.py` - JWT token encoding/decoding
  - `exceptions.py` - Global exception handlers (RateLimitError, CircuitOpenError middleware)
  - `deps.py` - FastAPI dependency injection (get_current_user, get_current_admin_user, get_db)
- `app/lib/` - Vendored vnstock runtime bindings
  - `vnstock_runtime.py` - Dynamic selection between vnstock versions
  - `vnstock_alt/`, `vnstock_data_alt/` - Alternative vnstock implementations

**Key Services:**

- `vnstock_service/` package - Modular service wrapping the `vnstock` library using facade pattern
  - **VnstockService** (facade) - Composes sub-services and provides unified interface
  - **Sub-services** - Each handles a specific domain (indices, stocks, funds, finance, company, history, metadata)
  - **Sync services** - Dedicated background sync classes per domain (history_sync, finance_sync, company_sync)
  - **Core utilities** - Retry logic with exponential backoff, circuit breaker integration
  - **Thread pool executors** - Separate pools for frontend (user-facing) and background (sync) operations
  - **Rate limit management** - Circuit breaker prevents cascading failures, fail-fast on rate limits
  - **Caching** - In-memory caching for indices, industries, and fund data with TTL; DB-backed cache for finance and company data
  - **Background sync** - 4 independent sync jobs: history (full/incremental/audit/repair), finance, company, funds
- `sync_status.py` - Global singleton for tracking all background sync operations, per-symbol progress, and rate limit status
- `circuit_breaker.py` - Thread-safe circuit breaker with CLOSED/OPEN/HALF_OPEN states
- Database uses async SQLAlchemy with `asyncpg` driver

**Data Models (`app/db/models.py`):**

- `StockCompany` - Company metadata (name, exchange, charter capital, PE ratio)
- `StockIndex` - Market indices metadata (symbol, name, description, group)
- `StockDailyHistory` - Historical OHLCV + buy/sell flow data; unique on (symbol, date)
- `StockHistorySyncState` - Per-symbol sync tracking (listing date, sync status, date ranges, retry count)
- `StockFinancialDataCache` - Cached financial statements; unique on (symbol, data_type, period, lang)
- `StockCompanyDataCache` - Cached company profiles; unique on (symbol, data_type)
- `FundNav` - Historical NAV data; unique on (symbol, date)
- `FundDetailCache` - Cached fund holdings (top_holding, industry_holding, asset_holding)
- `FundListing` - Fund metadata (name, type, owner)
- `User` - Application users (email, password_hash, download preferences, export categories)
- `BookmarkGroup` / `BookmarkStock` - User bookmark groups and their stocks
- `PortfolioPosition` - User portfolio positions (ticker, quantity, average cost, purchase date)

**API Patterns:**

- FastAPI dependency injection is used throughout (see `app/core/deps.py`)
- All endpoints return structured Pydantic models
- The `vnstock_service` singleton is accessed directly in endpoints (not via DI)
- Auth-protected endpoints use `get_current_user`; admin-only endpoints use `get_current_admin_user`
- Rate limiting is handled globally with retry logic and status tracking

### Frontend Architecture

**Entry Point:** `frontend/src/main.tsx` → `App.tsx`

- Routes `/admin` → `AdminPage` (requires authentication)
- Default route → `Dashboard`

**Feature-Based Structure:**

- `src/features/dashboard/` - Complete dashboard feature organized into subdirectories
  - **`Dashboard.tsx`** - Main container with tab navigation (Indices/Screeners/Funds/Portfolio), global popup state management, z-index coordination, AuthWidget integration

  - **`indices/`** - Stock indices and industry tracking
    - `IndicesTab.tsx` - Container managing index/industry/bookmark selection, table/chart view toggle
    - `IndexSelector.tsx`, `IndustrySelector.tsx`, `BookmarkSelector.tsx` - Dropdown selectors
    - `StocksTable.tsx` - Sortable, searchable stock table with click handlers for popups; CSV export
    - `StocksGrowthChart.tsx` - Weekly price growth visualization with benchmark comparison
    - `StocksComparisonChart.tsx` - Comparative performance chart across selected stocks
    - `StocksRiskReturnScatterPlot.tsx` - Risk vs return scatter plot for stock selection
    - `StocksVolumeChart.tsx` - Volume trading chart
    - `indexConfig.ts` - Index configuration definitions
    - `indexIndustryScope.ts` - Industry filtering by index utility
    - `stockExport.ts` - CSV export utilities for stock data

  - **`screeners/`** - Stock screening and filtering
    - `ScreenersTab.tsx` - Main screeners container with index/industry/bookmark scope selector
    - `ValuationScreener.tsx` + `valuationEngine.ts` - P/E, price-to-book, market cap analysis
    - `MarginTrendScreener.tsx` + `marginTrendEngine.ts` - Profit margin trend analysis
    - `FinancialHealthScreener.tsx` + `financialHealthEngine.ts` - Financial stability metrics
    - `LiquidityRiskScreener.tsx` + `liquidityRiskEngine.ts` - Liquidity and risk assessment
    - `valuationBenchmarks.ts` - Benchmark stock comparison definitions

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

  - **`portfolio/`** - User portfolio management
    - `PortfolioTab.tsx` - Position CRUD, CSV import/export, broker templates, real-time P/L, stock/industry allocation charts

  - **`banner/`** - Market overview
    - `IndexBanners.tsx` - Major index values (VNINDEX, HNXINDEX, UPCOMINDEX, VN30, HNX30) with 5-minute refresh

  - **`components/`** - Shared dashboard components
    - `CompanyFinancialPopup.tsx` - Draggable popup for financial statements, ratios, shareholders, officers, subsidiaries
    - `VolumeChartPopup.tsx` - Draggable popup for volume history chart
    - `PriceChartPopup.tsx` - Draggable popup for price history chart
    - `StockAllocationChart.tsx` - Stock/industry allocation visualization (used in portfolio)
    - `IndustryHoldingChart.tsx` - Industry holding breakdown chart

- `src/features/auth/` - Authentication
  - `AuthWidget.tsx` - Login/Register modal in Dashboard header
  - `useAuthUser.ts` - React hook for reactive auth state; JWT in localStorage, auto-logout on 401

- `src/features/admin/` - Admin dashboard (`/admin` route)
  - `AdminPage.tsx` - Tabbed admin panel (Settings, History Sync, Finance Sync, Company Sync)
  - `tabs/HistorySyncTab.tsx`, `FinanceSyncTab.tsx`, `CompanySyncTab.tsx`, `SettingsTab.tsx`
  - `components/FailedTickerList.tsx` - Failed symbol list display

**API Layer:**

- `src/api/stockApi.ts` - Centralized API client with Axios (~1140 lines)
- 60+ TypeScript interfaces for API request/response types
- 52 methods covering stocks, indices, industries, financials, company data, funds, historical data, portfolio, bookmarks, auth, and sync operations
- Axios interceptor injects Bearer token; auto-logout on 401

**Shared Components:**

- `src/components/TabNavigation.tsx` - Vertical tab navigation using DaisyUI
- `src/components/SyncIndicator.tsx` - Loading spinner for background sync operations

**Utilities (`src/utils/`):**

- `downloadFile.ts` - File download with user preference handling
- `downloadFolderPreference.ts` - User download folder preferences
- `exportCsv.ts` - CSV export utilities

**Component Patterns:**

- Feature-based with subdirectory organization: `src/features/dashboard/{indices|screeners|funds|portfolio|banner|components}/`
- Tab-based navigation: Indices, Screeners, Funds, Portfolio (Portfolio only when authenticated)
- Window-based event delegation for popup triggers (global coordination)
- Dashboard manages 3 popup types: CompanyFinancial, VolumeChart, PriceChart
- Recharts library for all data visualizations
- DaisyUI components are preferred for consistent styling

**State Management:**

- React hooks (`useState`, `useEffect`, `useCallback`, `useMemo`) for local state
- No global state management library (Redux, Zustand, etc.)
- API data is fetched and managed within feature components
- Global popup state managed in Dashboard.tsx and passed down
- Auth state via `useAuthUser` custom hook

## Environment Configuration

### Backend `.env`

Required variables (defaults and full list in `backend/app/core/config.py`; see `backend/.env.example`):

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/vnstock_hub
API_V1_PREFIX=/api/v1
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
VNSTOCK_API_KEY=your_vnstock_api_key_here  # For vnstock data source access
USE_VNSTOCK_ALT=false                       # Use vendored alternative vnstock implementation
USE_VNSTOCK_DATA_ALT=false                  # Use vendored alternative data provider
SETTINGS_YAML_PATH=./backend/settings.yaml

# Authentication
JWT_SECRET_KEY=your_jwt_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Sync configuration
SYNC_ADMIN_EMAILS=["admin@example.com"]     # Emails allowed to trigger sync jobs
SYNC_TARGET_RPM=150                         # Target API requests per minute
SYNC_MAX_WORKERS=10                         # Max background sync workers
SYNC_CHUNK_DAYS=1825                        # Days per sync chunk
SYNC_RATE_LIMIT_FIXED_WAIT_SECONDS=30
SYNC_RATE_LIMIT_MAX_WAIT_SECONDS=1200

# LLM (for portfolio import parsing)
LLM_PROVIDERS=[...]                         # OpenAI-compatible provider configs
LLM_REQUEST_TIMEOUT_SECONDS=30
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
  - Dedicated sync services per domain (history_sync, finance_sync, company_sync)
  - VnstockService facade provides unified interface
- On startup, the app syncs available indices from vnstock into the database
- Circuit breaker pattern for API rate limit management (CLOSED/OPEN/HALF_OPEN states)
  - Prevents cascading failures by failing fast when rate limited
  - Automatic recovery with configurable timeout
- Retry logic with exponential backoff for background sync operations
- Separate thread pool executors for frontend (user-facing) and background (sync) operations
- Four background sync jobs (history, finance, company, funds) with per-symbol progress tracking
- Global sync status tracking via `sync_status.py` singleton
- In-memory caching for indices, industries, and fund data with TTL management
- DB-backed cache for financial statements and company profiles
- JWT-based authentication; admin actions gated by email allowlist (`SYNC_ADMIN_EMAILS`)

### Frontend

- Feature-based organization: `features/dashboard/{indices|screeners|funds|portfolio|banner|components}/`, `features/auth/`, `features/admin/`
- Tab-based navigation: Indices, Screeners, Funds, Portfolio
- All API interactions through `src/api/stockApi.ts`
- Recharts library for all data visualizations (growth charts, scatter plots, heatmaps, volume)
- Global event delegation via window object for popup coordination (3 popup types)
- Automatic retry on rate limit errors with user feedback
- TailwindCSS + DaisyUI for styling
- TypeScript strict mode enabled
- Auth state managed via `useAuthUser` hook; Bearer token auto-injected by Axios interceptor

## Data Flow

### Stock Indices Flow

1. **Startup:** Backend fetches and caches available indices from vnstock into PostgreSQL
2. **Frontend Request:** User selects an index → Frontend calls `stockApi.getIndexStocks(symbol)`
3. **Circuit Breaker Check:** System checks if API is rate limited before proceeding
4. **Backend Processing:** `vnstock_service` fetches live data from vnstock library (fail-fast on rate limit), enriches with company metadata from DB
5. **Response:** Structured data returned with stock prices, market caps, PE ratios, price changes
6. **Frontend Display:** `StocksTable` or chart components render data with sorting, filtering, and interactive popups

### Mutual Funds Flow

1. **Background Sync:** `vnstock_service` syncs fund performance data periodically to reduce API calls
2. **Frontend Request:** User navigates to Funds tab → Frontend calls `stockApi.getFundPerformance()`
3. **Backend Processing:** Returns cached/synced fund performance data with benchmarks
4. **Frontend Display:** Multiple visualizations (growth charts, risk-return scatter, heatmaps, holdings)
5. **Rate Limit Handling:** Circuit breaker opens on rate limit; `sync_status` tracks state; frontend auto-retries after recovery

### Company Details Flow

1. **User Action:** Click on stock ticker in table
2. **Popup Trigger:** Window event triggers `CompanyFinancialPopup` with symbol
3. **Circuit Breaker Check:** Each API request checks if circuit breaker allows the call
4. **Multiple API Calls:** Frontend fetches income statement, balance sheet, cash flow, ratios, shareholders, officers, subsidiaries in parallel
5. **Popup Display:** Draggable popup with tabbed financial data and company information

### Volume / Price Chart Flow

1. **User Action:** Click volume or price icon in stock table
2. **Popup Trigger:** Window event triggers `VolumeChartPopup` or `PriceChartPopup` with symbol
3. **Circuit Breaker Check:** System checks if API is rate limited before proceeding
4. **API Call:** Frontend fetches history data
5. **Chart Display:** Draggable popup with Recharts visualization

### Portfolio Flow

1. **Auth Required:** User must be logged in; Portfolio tab only shown when authenticated
2. **Data Fetch:** `stockApi.getPortfolioPositions()` loads saved positions
3. **Real-Time Quotes:** `stockApi.getStockQuotes(symbols)` fetches current prices for P/L calculation
4. **CRUD:** Positions can be added, edited, deleted, imported from broker CSV, or exported
5. **Visualizations:** Stock allocation and industry allocation charts update with position changes

### Screeners Flow

1. **Scope Selection:** User selects an index, industry, or bookmark group as the stock universe
2. **Financial Data Fetch:** Screener fetches financial statements/ratios for all symbols in scope
3. **Engine Processing:** Client-side engine calculates metrics (valuation, margin trends, financial health, liquidity)
4. **Display:** Ranked/filtered stock table with metric columns; supports search and sorting
