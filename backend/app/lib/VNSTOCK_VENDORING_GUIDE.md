# Vnstock Vendoring Guide

This note records the current vendored architecture and the rules we followed
while re-implementing `vnstock` and `vnstock_data` inside the backend.

It is meant for future maintenance, especially when back-porting useful public
features from a newer upstream release.

## Current structure

- `app.lib.vnstock_alt`
  - Vendored replacement for the retained public-data surface of `vnstock`
- `app.lib.vnstock_data_alt`
  - Vendored replacement for the retained public-data surface of
    `vnstock_data`
- `app.lib._vnstock_shared`
  - Common internals shared by both vendored packages

The guiding rule is:

- `vnstock_alt` and `vnstock_data_alt` should import vendored modules directly
- they should not depend on `vnstock`, `vnstock_data`, or `vnai` namespace
  imports for their own internal correctness

`backend/tests/test_alt_package_compat.py` enforces this with an AST-based
self-containment check.

## Why these packages were vendored

The upstream packages were useful as compatibility targets but not as direct
runtime dependencies for the backend:

- they include auth and API-key flows that are not wanted here
- some limit mechanics are local-package behavior rather than true server-side
  enforcement
- the upstream surface is broader than what the backend actually needs

The vendored versions keep the useful public-data behavior while removing the
parts that do not belong in this backend.

## Intentional omissions

These are intentionally out of scope unless requirements change:

- user registration helpers
- API-key change/status helpers
- auth/license setup behavior
- local user-tier or local API-limit logic
- bundled charting support
- upgrade-notice, notebook, and Google Colab helper flows

If a future upstream release changes any of those areas, do not back-port them
unless the backend explicitly decides to support them.

## Current compatibility posture

We aim to preserve:

- public class and method names for retained surfaces
- method signatures where retained
- normalized output types and schemas
- backend-facing behavior for the currently used service paths

We do not promise byte-for-byte permanence for all live responses, because the
public upstream data sources themselves can change.

## Runtime integration model

The backend still imports public names like `vnstock` in service code.

`app.lib.vnstock_runtime` maps those public names at startup:

- `vnstock` resolves to `app.lib.vnstock_alt`
- `vnstock_data` resolves to `app.lib.vnstock_data_alt`

This aliasing exists for backend integration convenience only. The vendored
packages should remain internally self-contained.

## Shared-layer rules

When adding or back-porting code:

- prefer placing reusable helpers in `_vnstock_shared`
- keep provider-independent constants and transforms in the shared layer
- avoid duplicating utility logic in both alt packages
- keep package-specific facade behavior in `vnstock_alt` or `vnstock_data_alt`

Good shared-layer candidates:

- constants
- parser helpers
- dataframe transforms
- request/client helpers
- registry/provider plumbing
- standardized index metadata

## Charting policy

Chart helpers were intentionally removed from active use.

Current behavior:

- `common.__init__` no longer auto-imports `viz`
- `common.viz` still exists as a stub
- importing `Chart` or `get_chart` raises a clear `ImportError`

If charting is needed in the future, build it in a separate visualization
layer instead of re-expanding the vendored data packages by default.

## Upgrade and notebook policy

Legacy upstream helpers for package upgrade checks, notebook display, and
Google Colab setup were reduced to local-only no-op shims.

Current behavior:

- no PyPI upgrade checks from the vendored packages
- no Jupyter/IPython-specific upgrade messaging
- no Google Drive mount or Colab installation logic
- `.vnstock` path resolution always falls back to a normal local directory

## Back-port workflow

When bringing in a useful feature from future upstream releases:

1. Identify the exact upstream modules involved.
2. Decide whether the feature belongs to:
   - `vnstock_alt`
   - `vnstock_data_alt`
   - `_vnstock_shared`
3. Copy the smallest useful slice of code rather than bulk-replacing files.
4. Rewrite imports so the code uses vendored modules only.
5. Remove or bypass any new auth, tier, license, or startup side effects.
6. Check whether the feature pulls in new optional dependencies.
7. Add or extend compatibility tests before trusting the change.

## Back-port review checklist

Before merging a vendored back-port, verify:

- no internal imports from `vnstock`, `vnstock_data`, or `vnai`
- no auth/API-key logic leaked back into the retained surface
- no charting or notebook side effects were reintroduced accidentally
- retained public method contracts still match backend expectations
- output schemas match expected backend contracts
- optional live tests are updated for the new or changed methods

## Test strategy

The strongest validation is focused contract testing against the vendored
packages plus opt-in live smoke checks for retained public-data surfaces.

### Core tests

- `tests/test_alt_package_compat.py`
  - import smoke
  - self-containment
  - retained public method contracts
  - explicit vendored behavior checks
- `tests/test_vnstock_runtime.py`
  - permanent public aliasing behavior

### Live tests

- `tests/test_alt_package_extended_live.py`
  - broader retained-surface live coverage against vendored packages

### Useful commands

From `backend/`:

```bash
uv run pytest tests/test_vnstock_runtime.py tests/test_alt_package_compat.py
RUN_VNSTOCK_EXTENDED_LIVE_DIFF=1 uv run pytest tests/test_alt_package_extended_live.py
```

## Dependency policy

Prefer the smallest dependency set that supports the retained public-data
surface.

Current direction:

- keep `vnstock` installed while upstream fallback mode and differential tests
  still rely on it
- do not reintroduce `vnai` or `vnii` as declared backend dependencies unless a
  real supported use case requires them
- keep optional/legacy connector dependencies isolated when possible

## Known sharp edges

- upstream imports may still emit notices during tests that intentionally load
  the real installed packages for comparison
- live public APIs can rate-limit or change response shapes over time
- some copied legacy modules still exist even when they are outside the
  retained supported surface

If a future cleanup removes more dead upstream baggage, update this document so
the next maintainer knows which divergences were intentional.
