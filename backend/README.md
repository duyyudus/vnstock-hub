# VNStock Hub Backend

FastAPI backend for VNStock Hub, providing market/fund data APIs, user features (auth, bookmarks, portfolio), and background synchronization jobs.

## Tech Stack

- FastAPI (Python 3.12+)
- uv (dependency + environment management)
- PostgreSQL + SQLAlchemy async
- Alembic migrations
- Pydantic v2
- JWT auth (`python-jose`) + password hashing
- `vnstock` data source wrapper (`app/services/vnstock_service`)

## Current Capabilities

- Stock endpoints for index values, index/industry constituents, quotes, finance datasets, company datasets, and historical price/volume.
- Fund endpoints for listing, aggregated performance, NAV report, top holdings, industry holdings, and asset holdings.
- Auth endpoints for register/login plus user export preference settings.
- Bookmark groups for authenticated users (group CRUD + per-group stocks).
- Portfolio position management (CRUD), CSV export, CSV overwrite import, and LLM-assisted import from file/image flows.
- Sync status + admin controls for price sync, price audit, repair, finance sync, and company sync.
- Startup lifecycle that syncs indices and starts/stops background workers.

## Project Structure

```text
backend/
├── alembic/
├── app/
│   ├── api/v1/            # auth, bookmarks, funds, portfolio, stocks, sync
│   ├── core/              # config, deps, security, exceptions, logging
│   ├── db/                # async engine/session + SQLAlchemy models
│   ├── services/
│   │   ├── vnstock_service/   # facade + domain services + sync workers
│   │   ├── portfolio_import/  # broker crop profiles + extraction helpers
│   │   └── llm/               # OpenAI-compatible LLM clients
│   └── main.py
├── tests/
├── settings.yaml
└── pyproject.toml
```

## Setup

### Prerequisites

- Python 3.12+
- `uv`
- PostgreSQL

### Install + migrate

```bash
cd backend
uv sync
uv run alembic upgrade head
```

Note: app startup also runs `Base.metadata.create_all()` for dev convenience, but Alembic remains the source of truth.

### Environment variables

Common settings in `backend/.env`:

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/vnstock_hub
API_V1_PREFIX=/api/v1
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]

JWT_SECRET_KEY=change-me
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

SYNC_TARGET_RPM=150
SYNC_MAX_WORKERS=6
SYNC_CHUNK_DAYS=1095
SYNC_RATE_LIMIT_FIXED_WAIT_SECONDS=30
SYNC_RATE_LIMIT_MAX_WAIT_SECONDS=1200
SYNC_ADMIN_EMAILS=["admin@example.com"]

LLM_PROVIDERS='[]'
LLM_TASK_CONFIG='{}'
LLM_REQUEST_TIMEOUT_SECONDS=30
```

`LLM_PROVIDERS` is the shared provider catalog. `LLM_TASK_CONFIG` optionally maps task names such as `position_image_extraction` and `position_table_extraction` to ordered provider/model fallback chains. If a task is missing, the backend uses `default`; if `default` is also missing, it falls back to the legacy `LLM_PROVIDERS` list order.

### `settings.yaml`

Broker import presets are loaded from `backend/settings.yaml`:

```yaml
brokers:
  - id: vpbanks
    name: VPBank Securities
    sheet: Sheet1
    top_left: A9
    bottom_right: E
```

## Run

```bash
cd backend
uv run uvicorn app.main:app --reload --port 8000
```

- Swagger UI: `http://localhost:8000/docs`
- Quick start from repo root: `./run-server`

## Standalone CAFEF E1VFVN30 PDF Scraper

This repository includes a standalone CLI scraper at `tools/cafef_e1vfvn30_scraper.py` for:

- `E1VFVN30: Thông báo về danh mục chứng khoán cơ cấu hoán đổi`
- `E1VFVN30: Kết thúc giao dịch hoán đổi`

It stores PDFs on disk and metadata/state in SQLite (`catalog.sqlite`) under your output folder.

### Usage

```bash
cd backend
uv run python tools/cafef_e1vfvn30_scraper.py --output-dir /data/e1vfvn30_docs backfill
uv run python tools/cafef_e1vfvn30_scraper.py --output-dir /data/e1vfvn30_docs incremental --lookback-days 14
uv run python tools/cafef_e1vfvn30_scraper.py --output-dir /data/e1vfvn30_docs audit --from-date 2014-01-01 --to-date 2026-02-16
uv run python tools/cafef_e1vfvn30_scraper.py --output-dir /data/e1vfvn30_docs retry-failed --from-date 2026-01-01 --to-date 2026-02-16
```

Discovery checkpoints are resumable by date-range scope. Re-running the same backfill command resumes discovery without re-crawling completed discovery pages. Use `--reset-discovery-in-resume` to reset checkpoints and perform a full source refresh.

Discovery strategy defaults to `balanced`, which pre-filters non-target article URLs before full article fetch. Use `--discovery-strategy exhaustive` when you need maximum recall and accept slower runs.

Backfill/incremental support explicit pipeline modes:

- `--mode full`: discovery + consume (default)
- `--mode discover-sources`: event-feed/sitemap discovery only
- `--mode discover-idscan`: id-scan discovery only
- `--mode consume-only`: fetch queued discovered article URLs and download extracted PDFs only

For legacy gaps (especially 2014-2015), you can add optional ID-range discovery:

```bash
uv run python tools/cafef_e1vfvn30_scraper.py --output-dir /data/e1vfvn30_docs \
  backfill --from-date 2014-01-01 --to-date 2015-12-31 \
  --id-range 840000:950000 --id-scan-coarse-step 250 --id-scan-window 800
```

ID-scan discovery has its own resumable checkpoint per scope + ID range, and is only enabled when `--id-range` is provided.

ID-scan numeric args are validated (fail fast): `--id-scan-coarse-step >= 1`, `--id-scan-coarse-offsets >= 1`, `--id-scan-window >= 0`, `--id-scan-probe-max-retries >= 1`, `--id-scan-probe-timeout-seconds >= 1.0`.

Fine probing is auto-skipped when coarse scan already has full coverage across the range (for example `--id-scan-coarse-step 1` or `--id-scan-coarse-step N --id-scan-coarse-offsets N`).

`--id-scan-window` is used only when fine probing runs. Setting `--id-scan-window 0` means exact hit IDs only in fine phase.

ID-scan probing now uses `--max-concurrency` as the parallel worker cap while still honoring global `--rate-limit-rps`.

You can tune probe-stage behavior without changing other stages via global flags:

- `--id-scan-probe-max-retries`
- `--id-scan-probe-timeout-seconds`

If you resume the same scope + ID range with different ID-scan params (`step/offsets/window`), the run fails and asks you to re-run with `--reset-discovery-in-resume`.

For brute-force coarse scan coverage:

```bash
uv run python tools/cafef_e1vfvn30_scraper.py --output-dir /data/e1vfvn30_docs \
  backfill --from-date 2014-01-01 --to-date 2015-12-31 \
  --id-range 840000:950000 --id-scan-coarse-step 1 --id-scan-coarse-offsets 1 --id-scan-window 0
```

To quickly map historical ID windows before expensive fine scanning, use coarse-only mode:

```bash
uv run python tools/cafef_e1vfvn30_scraper.py --output-dir /data/e1vfvn30_docs \
  backfill --from-date 2017-01-01 --to-date 2021-12-31 \
  --max-event-pages 0 \
  --id-range 900000:1600000 --id-scan-coarse-step 1500 --id-scan-coarse-offsets 4 --id-scan-window 120 \
  --id-scan-coarse-only
```

The run summary includes `id_scan_suggested_windows` (for follow-up fine scans).

When resuming from coarse-only and you only want to process ID-scan discoveries, skip other discovery sources:

```bash
uv run python tools/cafef_e1vfvn30_scraper.py --output-dir /data/e1vfvn30_docs \
  backfill --from-date 2016-01-01 --to-date 2022-12-31 \
  --id-range 900000:1800000 --id-scan-coarse-step 1500 --id-scan-coarse-offsets 4 --id-scan-window 120 \
  --no-event-feed --no-sitemap
```

To consume already discovered URLs without any new discovery:

```bash
uv run python tools/cafef_e1vfvn30_scraper.py --output-dir /data/e1vfvn30_docs \
  backfill --from-date 2016-01-01 --to-date 2022-12-31 \
  --mode consume-only --no-event-feed --no-sitemap --no-retry-failed-docs
```

URL guessing fallback is removed. The scraper now only processes PDF URLs discovered from CAFEF article pages.

Backfill/incremental retry previously failed document URLs by default. Disable with `--no-retry-failed-docs` for focused test runs.

Rate limiting defaults to `--rate-limit-rps 1.5` with automatic HTTP retry/backoff (up to 5 attempts for request errors and `429/5xx` statuses). For long runs, you can enable adaptive throttling:

```bash
uv run python tools/cafef_e1vfvn30_scraper.py --output-dir /data/e1vfvn30_docs \
  --rate-limit-rps 4.0 --adaptive-rate-limit --adaptive-min-rps 0.8 \
  backfill --from-date 2014-01-01 --to-date 2015-12-31
```

If CAFEF starts timing out/rate-limiting during ID scans, use conservative adaptive settings (slower recovery + cooldown):

```bash
uv run python tools/cafef_e1vfvn30_scraper.py --output-dir /data/e1vfvn30_docs \
  --rate-limit-rps 1.0 --adaptive-rate-limit --adaptive-min-rps 0.2 \
  --adaptive-recovery-multiplier 1.02 --adaptive-cooldown-seconds 45 --adaptive-cooldown-streak 2 \
  --timeout-seconds 12 --max-retries 4 --id-scan-probe-timeout-seconds 8 --id-scan-probe-max-retries 2 \
  backfill --from-date 2014-01-01 --to-date 2015-12-31 --id-range 840000:950000
```

Use `--help` to view all options:

```bash
uv run python tools/cafef_e1vfvn30_scraper.py --help
```

## Testing

```bash
cd backend
uv run pytest
```

Test config is isolated via `backend/.env.test`.

## API Surface (`/api/v1`)

### Auth

- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/settings`
- `PATCH /auth/settings`

### Stocks

- `GET /stocks/index-values`
- `GET /stocks/indices`
- `GET /stocks/index/{index_symbol}`
- `GET /stocks/industries`
- `GET /stocks/industry/{industry_name}`
- `POST /stocks/quotes`
- `GET /stocks/finance/{symbol}/income-statement`
- `GET /stocks/finance/{symbol}/balance-sheet`
- `GET /stocks/finance/{symbol}/cash-flow`
- `GET /stocks/finance/{symbol}/ratios`
- `GET /stocks/company/{symbol}/overview`
- `GET /stocks/company/{symbol}/shareholders`
- `GET /stocks/company/{symbol}/officers`
- `GET /stocks/company/{symbol}/subsidiaries`
- `GET /stocks/history/{symbol}/volume`
- `GET /stocks/history/{symbol}/price`
- `POST /stocks/weekly-prices`

### Funds

- `GET /funds/listing`
- `GET /funds/performance`
- `GET /funds/{symbol}/nav-report`
- `GET /funds/{symbol}/top-holding`
- `GET /funds/{symbol}/industry-holding`
- `GET /funds/{symbol}/asset-holding`

### Portfolio (auth required)

- `GET /portfolio/positions`
- `POST /portfolio/positions`
- `PATCH /portfolio/positions/{position_id}`
- `DELETE /portfolio/positions/{position_id}`
- `GET /portfolio/export/csv`
- `POST /portfolio/import/fresh`
- `GET /portfolio/import/brokers`
- `POST /portfolio/import`

### Bookmarks (auth required)

- `GET /bookmarks/groups`
- `POST /bookmarks/groups`
- `PATCH /bookmarks/groups/{group_id}`
- `DELETE /bookmarks/groups/{group_id}`
- `GET /bookmarks/groups/{group_id}/stocks`
- `POST /bookmarks/groups/{group_id}/stocks`
- `DELETE /bookmarks/groups/{group_id}/stocks/{ticker}`

### Sync

- `GET /sync/status`
- `POST /sync/prices/run` (admin)
- `POST /sync/prices/audit/run` (admin)
- `POST /sync/prices/repair/run` (admin)
- `POST /sync/finance/run` (admin)
- `POST /sync/company/run` (admin)

### Non-versioned utilities

- `GET /`
- `GET /health`
