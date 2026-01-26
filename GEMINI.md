# VNStock Hub

VNStock Hub is a full-stack web application designed for tracking and analyzing the Vietnamese stock market. It provides real-time dashboards for market indices (VN-100, VN-30, etc.), mutual fund performance tracking, stock financial statements, and company information.

## Tech Stack

### Backend

* **Language:** Python 3.12+
* **Framework:** FastAPI
* **Database:** PostgreSQL (with `asyncpg` driver)
* **ORM:** SQLAlchemy (Async)
* **Migrations:** Alembic
* **Package Manager:** `uv`
* **Key Libraries:** `vnstock` (market data), `pydantic` (validation), `pandas` (data analysis)

### Frontend

* **Framework:** React 19
* **Language:** TypeScript
* **Build Tool:** Vite
* **Styling:** TailwindCSS, DaisyUI
* **HTTP Client:** Axios
* **Charts:** Recharts (performance, growth, and holding visualizations)
* **Package Manager:** `npm`

## Architecture

### Backend (`/backend`)

The backend follows a layered architecture:

* **Entry Point:** `app/main.py`
* **API Layer:** `app/api/v1/` - Defines endpoints for stocks, funds, and data synchronization.
  * `stocks.py`: Endpoints for market indices, industry filtering, and company financial/volume data.
  * `funds.py`: Endpoints for fund listings, NAV performance history, and asset/industry/top holdings.
* **Service Layer:** `app/services/` - Contains business logic.
  * `vnstock_service.py`: Core service wrapping the `vnstock` library with caching, retry logic, and background synchronization for historical data.
* **Database Layer:** `app/db/` - Contains SQLAlchemy models.
  * `models.py`: Defines schemas for `StockCompany`, `StockIndex`, `StockDailyPrice`, and `FundNav`.
* **Core:** `app/core/` - Configuration settings and dependencies.

### Frontend (`/frontend`)

The frontend is structured by features:

* **Entry Point:** `src/main.tsx` -> `src/App.tsx`
* **Features:** `src/features/` - Contains self-contained feature modules.
  * `dashboard/`: Main tracking hub with tabbed navigation:
    * **Indices Tab**: Real-time index tracking (`IndexBanners`), stock growth charts, and industry/index stock tables.
    * **Funds Tab**: Mutual fund analysis with Nav report charts, risk-return scatter plots, and allocation breakdowns (Top Holding, Industry, Asset).
    * **Popups**: Draggable popups for `CompanyFinancialPopup` and `VolumeChartPopup`.
* **API:** `src/api/` - Centralized API client configuration (`stockApi.ts`).
* **Components:** `src/components/` - Shared UI components like `TabNavigation` and `SyncIndicator`.

## Setup and Development

### Quick Start (Root)

* **Run Backend:** `./run-server`
* **Run Frontend:** `./run-ui`

### Backend Setup

1. Navigate to `backend/`.
2. Install dependencies: `uv sync`
3. Run migrations: `uv run alembic upgrade head`
4. Run server: `uv run uvicorn app.main:app --reload --port 8000`

### Frontend Setup

1. Navigate to `frontend/`.
2. Install dependencies: `npm install`
3. Run dev server: `npm run dev` (Access at `http://localhost:5173`)

## Conventions

* **Package Management:** Always use `uv` for backend python dependencies and `npm` for frontend.
* **Async/Await:** The backend uses `async/await` extensively, especially for database interactions and external API calls.
* **Styling:** Use TailwindCSS utility classes and DaisyUI components for UI development.
* **Versioning:** API endpoints are versioned (e.g., `/api/v1/`).
* **Feature Folders:** Keep frontend components, hooks, and logic related to a specific feature within its `features/<feature-name>` directory.
* **Version Control:** Please implement the changes, but leave them for my review, do not commit nor stage them.
