# VNStock Hub - Backend

This is the backend component of the VNStock Hub, a web application for tracking and analyzing the Vietnam stock market. It provides a RESTful API built with FastAPI, leveraging the `vnstock` library for data retrieval and PostgreSQL for data persistence.

## 🚀 Tech Stack

- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python 3.12+)
- **Package Management:** [uv](https://github.com/astral-sh/uv)
- **Database:** PostgreSQL with [SQLAlchemy](https://www.sqlalchemy.org/) ORM (Async)
- **Migrations:** [Alembic](https://alembic.sqlalchemy.org/)
- **Data Source:** [vnstock](https://github.com/thinh-vu/vnstock) library
- **Validation:** Pydantic v2
- **Security:** JWT Authentication with `passlib` (bcrypt)

## 📦 Project Structure

```text
backend/
├── alembic/            # Database migrations
├── app/
│   ├── api/v1/         # API endpoints (Auth, Stocks, Funds, Portfolio, Sync)
│   ├── core/           # Config, Security, Logging, Circuit Breaker
│   ├── db/             # Database session and SQLAlchemy models
│   ├── services/       # Business logic
│   │   └── vnstock_service/ # Modular vnstock wrapper (History, Finance, Funds, etc.)
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
4. Run migrations:
   ```bash
   uv run alembic upgrade head
   ```

### Running the Application

Start the development server:
```bash
uv run uvicorn app.main:app --reload --port 8000
```
Interactive API docs: `http://localhost:8000/docs`

## 🔌 API Endpoints (v1)

- **Auth:** `/auth/register`, `/auth/token`, `/auth/me`
- **Stocks:** `/stocks/index/{symbol}`, `/stocks/industry/{industry_name}`, `/stocks/{symbol}/financials`, `/stocks/{symbol}/volume`
- **Funds:** `/funds/list`, `/funds/{symbol}/nav`, `/funds/{symbol}/holdings`
- **Portfolio:** `/portfolio/positions`, `/portfolio/summary`
- **Bookmarks:** `/bookmarks/groups`, `/bookmarks/stocks`
- **Sync:** `/sync/status`, `/sync/trigger`

## 🧪 Features

- **Real-time Market Data:** Integration with `vnstock` for indices, industries, and stock details.
- **Mutual Fund Analysis:** Tracking NAV history, asset allocation, and top holdings.
- **Portfolio Management:** Track positions, gains/losses, and historical performance.
- **User Personalization:** JWT-based auth with customizable bookmark groups.
- **Robust Infrastructure:** Async database operations, circuit breakers for external APIs, and background data synchronization.