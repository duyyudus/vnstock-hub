# Vendored Vnstock Packages

This directory contains the vendored compatibility replacements for the
upstream `vnstock` and `vnstock_data` packages used by the backend.

## What lives here

- `vnstock_alt`
  - Vendored replacement for the public-data parts of upstream `vnstock`
- `vnstock_data_alt`
  - Vendored replacement for the public-data parts of upstream `vnstock_data`
- `_vnstock_shared`
  - Shared helpers, constants, parsers, registry logic, transport helpers,
    and compatibility shims used by both vendored packages
- `vnstock_runtime.py`
  - Runtime aliasing helper that lets the backend keep importing `vnstock`
    and `vnstock_data` while always resolving them to vendored packages

## Design goals

- Preserve the retained upstream public API shape for backend compatibility
- Remove auth, API-key, and user-tier mechanics from the vendored packages
- Keep implementation self-contained inside `app.lib`
- Keep legacy public import names without depending on upstream packages

## Runtime aliases

The aliasing logic in `vnstock_runtime.py` always maps public imports to the
vendored packages:

- `vnstock` resolves to `app.lib.vnstock_alt`
- `vnstock_data` resolves to `app.lib.vnstock_data_alt`

## Intentional differences from upstream

- No auth or API-key registration helpers in the vendored public surface
- No local API-limit behavior copied from upstream
- No startup upgrade/auth notices in `vnstock_alt`
- Charting helpers are intentionally disabled and replaced with clear stubs

## Testing map

The main compatibility coverage lives in `backend/tests/`:

- `test_alt_package_compat.py`
  - import smoke, self-containment, retained public method contracts
- `test_alt_package_extended_live.py`
  - opt-in alt-only live coverage for retained public surfaces
- `test_vnstock_runtime.py`
  - permanent public alias behavior

## Before changing these packages

Read `VNSTOCK_VENDORING_GUIDE.md` in this folder. It documents:

- what was intentionally kept or removed
- how to back-port features from future upstream releases
- what tests to run before trusting a change

For a package-level usage guide, read `VNSTOCK_API_GUIDE.md`.

## Local API docs

The vendored packages now ship with a generated local docs workflow that
combines source-derived API reference with optional live-observed samples.

From `backend/`:

Generate docs from source:

```bash
uv run python scripts/generate_vnstock_api_docs.py
```

Capture live samples, then regenerate docs:

```bash
uv run python scripts/capture_vnstock_api_samples.py
uv run python scripts/generate_vnstock_api_docs.py
```

Browse the local site:

```bash
uv run mkdocs serve -f mkdocs.yml
```

The MkDocs dev server is configured to run at `http://127.0.0.1:8001/` so it does not conflict with the backend API on port `8000`.

Build static HTML:

```bash
uv run mkdocs build -f mkdocs.yml
```

Key paths:

- generated docs source: `backend/docs/generated/`
- live probe manifest: `backend/docs/live_probe_manifest.json`
- live snapshot outputs: `backend/docs/generated/live/`
