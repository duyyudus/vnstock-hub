# Agent Playbook

## Purpose

This file is the root operational guide for coding agents working in this repo.
Keep changes targeted, verify repo facts before editing, and prefer the smallest
useful validation step for the area you touched.

## Repo Map

- `backend/`: FastAPI service.
  - Entry point: `backend/app/main.py`
  - API routers: `backend/app/api/v1/`
  - Core config/deps/security/logging: `backend/app/core/`
  - DB models/session: `backend/app/db/`
  - Main business logic: `backend/app/services/`
  - Vendored vnstock code: `backend/app/lib/`
  - Migrations: `backend/alembic/`
  - Tests: `backend/tests/`
- `frontend/`: React 19 + Vite app.
  - Entry points: `frontend/src/main.tsx`, `frontend/src/App.tsx`
  - API client: `frontend/src/api/stockApi.ts`
  - Shared UI: `frontend/src/components/`
  - Feature modules: `frontend/src/features/`
- Root helpers:
  - `./run-server`
  - `./run-ui`

## Working Rules

- Keep backend API additions under `/api/v1` and follow existing dependency
  patterns in `backend/app/core/deps.py`.
- Route frontend HTTP changes through `frontend/src/api/stockApi.ts` rather than
  scattering ad hoc `fetch`/`axios` calls across features.
- Treat Alembic as the schema source of truth. `Base.metadata.create_all()` in
  app startup exists for dev convenience, not as a migration substitute.
- Avoid casual edits in `backend/app/lib/*`. That tree contains vendored
  compatibility code for `vnstock` and `vnstock_data`; read
  `backend/app/lib/VNSTOCK_VENDORING_GUIDE.md` and relevant tests before
  changing it.
- Preserve existing local style. Python uses 4-space indentation; frontend
  files use semicolons and the existing Tailwind/DaisyUI patterns.

## Commands

Backend setup and run:

```bash
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000
```

Frontend setup and run:

```bash
cd frontend
npm install
npm run dev
npm run build
npm run lint
```

Root shortcuts:

```bash
./run-server
./run-ui
```

## Execution Boundaries

- Be careful with `backend/app/main.py`: app startup creates tables, syncs
  indices, and starts background workers. Do not run the backend casually when
  a targeted test or static check will answer the question.
- Prefer narrow validation for the surface you changed.
  - Backend logic: targeted `uv run pytest ...`
  - Frontend changes: `npm run lint` and, when relevant, `npm run build`
- Use full-app runs only when the task actually needs runtime verification.

## Testing Defaults

- Backend default suite:

```bash
cd backend
uv run pytest
```

- Frontend default checks:

```bash
cd frontend
npm run lint
npm run build
```

- Live/networked vnstock comparison suites are opt-in, not default validation.
  Run them only when the task touches the vendored/runtime vnstock surface or
  specifically needs live-source verification.
  - `RUN_VNSTOCK_LIVE_DIFF=1 uv run pytest tests/test_alt_package_differential_live.py`
  - `RUN_VNSTOCK_SERVICE_SHADOW=1 uv run pytest tests/test_vnstock_service_shadow_live.py`
  - `RUN_VNSTOCK_EXTENDED_LIVE_DIFF=1 uv run pytest tests/test_alt_package_extended_live.py`

## Configuration Notes

- Backend settings load from `backend/.env` by default via `APP_ENV_FILE`.
- Frontend uses `frontend/.env` with:

```env
VITE_API_BASE_URL=http://localhost:8000
```

- The frontend client appends `/api/v1` itself in
  `frontend/src/api/stockApi.ts`. Do not duplicate that suffix in the env var.

## High-Signal Gotchas

- Admin and dashboard UI are split by path in `frontend/src/App.tsx`; `/admin`
  renders separately from the default dashboard flow.
- Backend sync behavior spans multiple services under
  `backend/app/services/vnstock_service/`; changes there can affect startup
  workers, sync status, and cached data behavior.
- This repo may contain live-data and rate-limit-sensitive tests. Failures in
  opt-in live suites are not automatically equivalent to local regressions.
