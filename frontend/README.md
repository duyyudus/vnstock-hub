# VNStock Hub Frontend

React + TypeScript dashboard for VNStock Hub, covering market analytics, fund exploration, personal portfolio tracking, bookmarks, auth, and admin sync controls.

## Tech Stack

- React 19 + TypeScript
- Vite
- TailwindCSS + DaisyUI
- Recharts
- Axios

## App Structure

```text
frontend/
├── src/
│   ├── api/                 # stockApi.ts (all backend calls + auth storage)
│   ├── components/          # shared components (tab nav, sync indicator)
│   ├── features/
│   │   ├── auth/            # AuthWidget, useAuthUser
│   │   ├── admin/           # /admin page + sync control tabs
│   │   └── dashboard/
│   │       ├── banner/      # index banners
│   │       ├── components/  # financial/volume/price popups, shared charts
│   │       ├── funds/       # fund analysis views
│   │       ├── indices/     # index/industry/bookmark views + table/charts/export
│   │       └── portfolio/   # portfolio management and imports/exports
│   ├── App.tsx              # path switch: dashboard vs /admin
│   └── main.tsx
└── package.json
```

## Key Features

- Dashboard tabs: `Stocks`, `Funds`, and `Portfolio` (portfolio tab shown when logged in).
- Stocks workflows: search, index/industry/bookmark filters, table + growth/comparison/risk-return chart modes.
- Stock tooling: draggable financial popup, volume history popup, and price history popup.
- Bookmark workflows: group CRUD and per-group ticker management from the stock table.
- Batch export flow from stocks tab for company + finance CSV datasets.
- Funds analytics: NAV charts, holdings breakdowns, cumulative growth, risk-return scatter, periodic return heatmap.
- Portfolio workflows: position CRUD, quote refresh, CSV export, fresh CSV import, and LLM-assisted import using broker presets.
- Admin page (`/admin`): live sync status polling and trigger actions for price, audit, repair, finance sync, and company sync.
- Auth/session handling: register/login, JWT persistence, auto-logout by token expiry, and user settings for export preferences.

## Setup

### Prerequisites

- Node.js 18+
- npm

### Install

```bash
cd frontend
npm install
```

### Environment

Create `frontend/.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

## Run

```bash
cd frontend
npm run dev
```

- App URL: `http://localhost:5173`
- Quick start from repo root: `./run-ui`

## Scripts

- `npm run dev` — start dev server
- `npm run build` — type-check + production build
- `npm run preview` — preview built app
- `npm run lint` — ESLint
