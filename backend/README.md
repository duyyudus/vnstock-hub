# VNStock Hub - Backend

This is the backend component of the VNStock Hub, a web application for tracking and analyzing the Vietnam stock market. It provides a RESTful API built with FastAPI, leveraging the `vnstock` library for data retrieval and PostgreSQL for data persistence.

## 🚀 Tech Stack

- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python 3.12+)
- **Package Management:** [uv](https://github.com/astral-sh/uv)
- **Database:** PostgreSQL with [SQLAlchemy](https://www.sqlalchemy.org/) ORM
- **Migrations:** [Alembic](https://alembic.sqlalchemy.org/)
- **Data Source:** [vnstock](https://github.com/thinh-vu/vnstock) library
- **Validation:** Pydantic v2

## 📦 Project Structure

```text
backend/
├── alembic/            # Database migrations
├── app/
│   ├── api/            # API routing and endpoints (v1)
│   ├── core/           # Configuration and global settings
│   ├── db/             # Database session and models
│   ├── services/       # Business logic and external service integrations
│   └── main.py         # Application entry point
├── pyproject.toml      # Project dependencies and metadata
└── uv.lock             # Lockfile for reproducible builds
```

## 🛠️ Getting Started

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) installed
- PostgreSQL (optional for local development if using mock data, but required for full functionality)

### Installation

1. Clone the repository and navigate to the backend directory:

   ```bash
   cd backend
   ```

2. Sync dependencies using `uv`:

   ```bash
   uv sync
   ```

3. Create a `.env` file based on the environment variables needed (see `app/core/config.py` for reference).

### Running the Application

Start the development server:

```bash
uv run uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. You can access the Interactive API documentation at `http://localhost:8000/docs`.

## 🔌 API Endpoints

### Stocks

- `GET /api/v1/stocks/index/{index_symbol}`: Returns stocks for a given index symbol (e.g. `VN100`, `VN30`) with ticker, price, and market capitalization.

## 🧪 Features

- **Index Dashboard Data:** Real-time (or cached) fetching of index constituents in the Vietnamese market.
- **Market Cap Conversion:** Data is processed to ensure consistent units (VND) for display.
- **Ticker Formatting:** Standardized 3-character ticker symbols.
