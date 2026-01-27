# Repository Guidelines

## Project Structure & Module Organization

- `backend/`: FastAPI service. Entry point `backend/app/main.py`; versioned routes in `backend/app/api/v1`; business logic in `backend/app/services`; database models/session in `backend/app/db`; settings in `backend/app/core`.
- `backend/alembic/`: database migrations.
- `backend/tests/`: pytest-based backend tests.
- `frontend/`: React app. Entry `frontend/src/main.tsx` -> `frontend/src/App.tsx`; feature modules in `frontend/src/features`; shared UI in `frontend/src/components`; API client in `frontend/src/api/stockApi.ts`.
- Root helpers: `run-server` and `run-ui` wrap the dev commands.

## Build, Test, and Development Commands

Backend:

```bash
cd backend
uv sync                               # install dependencies
uv run alembic upgrade head           # run migrations
uv run uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev                           # http://localhost:5173
npm run build                         # production build
npm run preview                       # preview build
npm run lint                          # ESLint
```

Quick start from repo root:

```bash
./run-server
./run-ui
```

## Coding Style & Naming Conventions

- Python uses FastAPI with async/await; keep new endpoints under `/api/v1` and follow dependency patterns in `backend/app/core/deps.py`.
- Frontend follows a feature-based layout (`src/features/<feature>`). Route all API calls through `src/api/stockApi.ts`.
- Use TailwindCSS + DaisyUI for UI consistency.
- Follow existing formatting: 4-space indentation and semicolons in `.ts/.tsx`. No auto-formatter is configured, so keep changes aligned with nearby code.

## Testing Guidelines

- Backend: `uv run pytest` for the full suite; standalone scripts live in `backend/tests/` (e.g., `uv run python tests/test_vn100_fetch.py`).
- Frontend: no automated tests are currently configured; run `npm run lint` for static checks.

## Commit & Pull Request Guidelines

- Commit history follows Conventional Commits (e.g., `feat:`, `fix:`, `refactor:`, `perf:`, `style:`, `chore:`, `build:`). Keep messages short and scoped.
- PRs should include a concise summary, test commands run, linked issues (if any), and screenshots/GIFs for UI changes. Note any required env or migration steps.

## Configuration & Secrets

- Backend configuration is loaded from `backend/.env` (see `backend/app/core/config.py` for variables).
- Frontend expects `frontend/.env` with `VITE_API_URL=http://localhost:8000/api/v1`.
- Do not commit secrets or local overrides.
