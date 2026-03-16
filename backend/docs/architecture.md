# Vendored Vnstock Package Architecture

This document covers the architecture of the vendored packages under
`backend/app/lib` only.

It is not a map of the FastAPI application, service layer, database, sync
pipelines, auth flows, or the overall backend system. Backend integration is
mentioned only where it helps explain how the vendored packages are loaded or
consumed.

## Why This Page Exists

The generated docs under `generated/` explain exported APIs, normalized output
schemas, and live-observed samples. They do not explain how the vendored
packages are structured internally, how they relate to each other, or where the
current coupling points live.

Use this page when you need to answer package-level questions such as:

- which vendored package owns a given public surface
- how a method call flows from root export to provider implementation
- what is shared through `_vnstock_shared`
- where `vnstock_alt` and `vnstock_data_alt` currently depend on each other

## Package Inventory

| Package | Role |
| --- | --- |
| `app.lib.vnstock_alt` | Vendored replacement for the retained `vnstock` public-data surface. |
| `app.lib.vnstock_data_alt` | Vendored replacement for the retained `vnstock_data` public-data surface, including routed UI namespaces and schema normalization. |
| `app.lib._vnstock_shared` | Shared internals reused by the vendored packages: common client entrypoints, registries, transforms, parser helpers, logging, and compatibility utilities. |
| `app.lib.vnstock_runtime` | Runtime aliasing helper that maps public imports like `vnstock` and `vnstock_data` to vendored modules when feature flags are enabled. |

## Ownership Boundaries

### `vnstock_alt`

Owns the compatibility-first `vnstock` surface that the backend already expects.

- Root exports such as `Vnstock`, `Quote`, `Company`, `Finance`, `Listing`,
  `Trading`, and `Fund`
- Adapter-style public APIs under `api/`
- Provider implementations under `explorer/` and `connector/`
- A compatibility-oriented call path built around provider lookup and direct
  delegation

### `vnstock_data_alt`

Owns the broader `vnstock_data`-style data-product surface.

- Root exports such as `Quote`, `Company`, `Finance`, `Listing`, `Trading`,
  `CommodityPrice`, `TopStock`, `Fund`
- Routed UI namespaces exposed lazily from the package root:
  `Reference`, `Market`, `Insights`, `Fundamental`, `Macro`, `Analytics`
- Schema-aware dispatch and output normalization through `ui/`
- Provider implementations under `explorer/` and `connector/`

### `_vnstock_shared`

Owns reusable internals that should not be duplicated across the vendored
packages.

- Shared registries and typed/provider infrastructure
- Common entrypoint objects such as the vendored `Vnstock` client internals
- Transforms, parsers, logging, compatibility shims, and market metadata

### `vnstock_runtime`

Owns runtime import aliasing only.

- It decides whether public imports resolve to upstream packages or vendored
  packages
- It is part of backend integration context, not part of either package's data
  model or provider pipeline

## Internal Layer Maps

### `vnstock_alt`

`vnstock_alt` is adapter-first and compatibility-oriented. Its public classes
mostly wrap provider implementations chosen by source name.

```text
vnstock_alt package root
  -> root exports / lazy helpers
  -> api/* adapter classes
  -> BaseAdapter + ProviderRegistry lookup
  -> explorer/* or connector/* provider class
  -> remote source / scraped payload

Shared support used along the way:
  -> _vnstock_shared.common.*
  -> _vnstock_shared.core.*
```

Key traits:

- The root module exports a lazy `Vnstock` proxy that resolves to
  `_vnstock_shared.common.client.Vnstock`
- Public `api/*` classes use `BaseAdapter` to choose a provider by
  `(module_name, source)`
- Providers self-register into a registry and are then called directly
- The flow is relatively thin once a provider has been selected

Representative call flow:

```text
from vnstock import Quote
  -> vnstock_runtime may alias `vnstock` to `app.lib.vnstock_alt`
  -> vnstock_alt.Quote(source="kbs", symbol="VCI")
  -> vnstock_alt.base.BaseAdapter
  -> registry lookup for ("quote", "kbs")
  -> vnstock_alt.explorer.kbs.quote.Quote
  -> provider method such as history()
```

### `vnstock_data_alt`

`vnstock_data_alt` has two layers:

- an adapter layer under `api/`, similar in shape to `vnstock_alt`
- a routed UI layer under `ui/`, which maps namespace methods to providers and
  standardizes returned columns

```text
vnstock_data_alt package root
  -> root exports / lazy UI namespace loading
  -> either:
     a) api/* adapter classes
        -> BaseAdapter + ProviderRegistry
        -> explorer/* or connector/* provider
     b) ui/* domain objects
        -> ui._registry route table
        -> ui.config overrides
        -> ProviderRegistry lookup
        -> provider call
        -> ui.schemas column standardization
```

Key traits:

- The package root lazily exposes high-level namespaces such as `Market` and
  `Reference`
- `ui._registry` is the package-local routing table that maps domain methods to
  `(source, provider_type, class_name, method)`
- `ui.config` allows route overrides without rebuilding the whole provider tree
- `ui.schemas` and related helpers normalize columns after provider execution

Representative UI flow:

```text
from vnstock_data_alt import Market
  -> Market().equity(symbol).ohlcv(...)
  -> ui domain object in ui/domains/market/*
  -> ui._registry route selection
  -> ProviderRegistry lookup
  -> explorer provider method
  -> standardized output columns via ui.schemas
```

## Relationship Between The Vendored Packages

This section is a dependency map among the vendored packages only. It is not a
full dependency graph of the backend application.

### High-Level Relationship

```text
                   +----------------------+
                   |   _vnstock_shared    |
                   | shared internals     |
                   +----------+-----------+
                              ^
                              |
                +-------------+-------------+
                |                           |
                |                           |
        +-------+--------+         +--------+---------+
        |   vnstock_alt  |         | vnstock_data_alt |
        | compat facade  |         | routed data UI   |
        +----------------+         +------------------+
                ^                           |
                |                           |
                +----------- some current --+
                            direct reuse
```

### Current Coupling Points

Both packages depend on `_vnstock_shared`, but they are not fully isolated from
each other today.

Current direct reuse from `vnstock_data_alt` into `vnstock_alt` includes:

- MSN providers registered into `vnstock_data_alt.ui._registry` from
  `vnstock_alt.explorer.msn.*`
- shared constant reuse from `vnstock_alt.explorer.vci.const`
- shared FMarket constants reused from `vnstock_alt.explorer.fmarket.const`
- a market-events fallback path in `ui/domains/reference/events.py` that looks
  into `vnstock_alt` if shared loading paths fail

What this means in practice:

- Conceptually the packages are sibling vendored packages
- Operationally `vnstock_data_alt` still borrows selected implementation pieces
  from `vnstock_alt`
- That reuse is intentional in the current state, but it creates maintenance
  coupling that contributors should treat as fragile

## Backend Entry Context

This page is not backend architecture documentation, but two backend touchpoints
matter for understanding how the vendored packages are reached.

```text
backend app startup / vnstock_service imports
  -> app.lib.vnstock_runtime.install_vnstock_aliases()
  -> public import name `vnstock` and optionally `vnstock_data`
  -> vendored module selected by feature flags
```

Only the following backend context is relevant here:

- `app.lib.vnstock_runtime` controls import aliasing
- `app.services.vnstock_service` continues importing public names like
  `from vnstock import Vnstock`
- those imports can resolve to vendored packages without changing the service
  call sites

## Maintenance Notes And Sharp Edges

- `vnstock_alt` is the more compatibility-shaped facade for existing backend
  usage and upstream-style imports
- `vnstock_data_alt` adds a richer routed UI layer and normalized schema layer,
  so changes there often have both routing and output-shape implications
- The two packages have separate registry systems in practice, even though both
  also rely on shared internals from `_vnstock_shared`
- Lazy loading is used in several places to avoid circular import deadlocks, so
  entrypoint changes should be checked carefully
- Some vendored modules are intentionally retained for compatibility even if the
  backend does not actively use every exported surface

## Safe Extension Rules

- Put provider-agnostic helpers in `_vnstock_shared` when both vendored
  packages would otherwise duplicate the same logic
- Keep compatibility-facing facade behavior in `vnstock_alt`
- Keep routed namespace behavior, route tables, and schema normalization in
  `vnstock_data_alt`
- If `vnstock_data_alt` must reuse `vnstock_alt` internals, document the reason
  in code and update this page so the coupling remains explicit
- Treat new package-to-package imports as an architectural decision, not an
  incidental convenience

## Out Of Scope

This document does not describe:

- FastAPI routes or request handling
- service orchestration in the backend
- database models, caching tables, or persistence flows
- auth, users, or admin features
- sync jobs and background workers
- frontend consumers or UI behavior

For exported APIs, schema contracts, and live-observed samples, use the
generated docs linked from the docs homepage.
