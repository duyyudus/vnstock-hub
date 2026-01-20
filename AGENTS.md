# VNStock Hub

VNStock Hub is a full-stack web application for tracking and analyzing the Vietnamese stock market. It provides a real-time dashboard of the VN-100 index, featuring price tracking, market capitalization analysis, and a modern user interface.

## 🚀 Tech Stack

### Backend
- **Framework:** FastAPI (Python 3.12+)
- **Package Management:** `uv`
- **Database:** PostgreSQL with SQLAlchemy ORM
- **Migrations:** Alembic
- **Data Source:** `vnstock` library for fetching Vietnamese market data.

### Frontend
- **Framework:** React 19 (Vite)
- **Language:** TypeScript
- **Styling:** TailwindCSS & DaisyUI
- **State Management/API:** Axios for API requests.

---

## 🛠️ Building and Running

### Backend

1. **Install Dependencies:**
   ```bash
   cd backend
   uv sync
   ```
2. **Setup Environment:**
   Create `backend/.env` (refer to `backend/app/core/config.py`).
3. **Run Migrations:**
   ```bash
   cd backend
   uv run alembic upgrade head
   ```
4. **Start Server:**
   ```bash
   cd backend
   uv run uvicorn app.main:app --reload --port 8000
   ```
   API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Frontend

1. **Install Dependencies:**
   ```bash
   cd frontend
   npm install
   ```
2. **Setup Environment:**
   Create `frontend/.env` with `VITE_API_URL=http://localhost:8000/api/v1`.
3. **Start Development Server:**
   ```bash
   cd frontend
   npm run dev
   ```
   App: [http://localhost:5173](http://localhost:5173)

---

## 🧪 Testing

### Backend
Run standalone test scripts:
```bash
cd backend
uv run python tests/test_vn100_fetch.py
```

### Frontend
Linting:
```bash
cd frontend
npm run lint
```

---

## 📂 Project Structure

- `backend/`: FastAPI application.
  - `app/api/`: API endpoints (v1).
  - `app/core/`: Configuration and settings.
  - `app/db/`: Database models and connection.
  - `app/services/`: Business logic (stock data fetching).
- `frontend/`: React application.
  - `src/api/`: API client definitions.
  - `src/features/`: Feature-based modules (e.g., `dashboard`).
  - `src/components/`: Reusable UI components.

---

## 📝 Development Conventions

- **Backend:**
  - Use `uv` for all dependency and environment management.
  - Follow FastAPI's dependency injection pattern (`app/core/deps.py`).
  - API versioning is strictly enforced under `/api/v1`.
- **Frontend:**
  - Feature-based architecture: Keep components, hooks, and logic related to a specific feature within `src/features/[feature_name]`.
  - Use DaisyUI for styling components to maintain visual consistency.
  - All API interactions should go through `src/api/stockApi.ts`.
