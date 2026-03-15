# Vnstock Alt API Guide

This document is a practical usage guide for the vendored packages:

- `app.lib.vnstock_alt`
- `app.lib.vnstock_data_alt`

It focuses on the retained root public API that we intentionally support.

For exact signature parity and compatibility rules, treat these as the source
of truth:

- class docstrings in the vendored modules
- `backend/tests/test_alt_package_compat.py`
- the live comparison suites in `backend/tests/`

## Which package to use

Use `vnstock_alt` when you want the classic `vnstock` facade:

```python
from app.lib.vnstock_alt import Vnstock, Listing, Trading, Company, Finance, Fund
```

Use `vnstock_data_alt` when you want the newer `vnstock_data` style adapters
and UI/domain entrypoints:

```python
from app.lib.vnstock_data_alt import (
    Quote, Company, Finance, Listing, Trading,
    CommodityPrice, TopStock, Fund,
    Reference, Market, Insights, Fundamental, Macro, Analytics,
    show_api, show_doc,
)
```

## Common patterns

### Classic `vnstock_alt` facade

```python
from app.lib.vnstock_alt import Vnstock

stock = Vnstock().stock(symbol="VCB", source="VCI")

price_df = stock.quote.history(start="2024-01-01", end="2024-12-31")
income_df = stock.finance.income_statement(period="quarter")
company_df = stock.company.overview()
```

### Direct adapter usage

```python
from app.lib.vnstock_alt import Listing, Trading, Company, Finance

symbols = Listing(source="VCI").all_symbols()
board = Trading(source="VCI").price_board(["VCB", "FPT"])
overview = Company(symbol="VCB", source="VCI").overview()
ratios = Finance(symbol="VCB", source="VCI").ratio(period="year")
```

### `vnstock_data_alt` adapter usage

```python
from app.lib.vnstock_data_alt import Listing, Quote, Trading

indices = Listing().all_indices()
history = Quote(symbol="VCB", source="vci").history(start="2024-01-01", end="2024-12-31")
board = Trading(source="vci").price_board(["VCB", "FPT"])
```

### `vnstock_data_alt` discovery helpers

```python
from app.lib.vnstock_data_alt import show_api, show_doc, Reference

show_api()
show_doc(Reference)
index_df = Reference().index.list()
```

## `vnstock_alt` root API

Root exports:

- `Vnstock`
- `Quote`
- `Company`
- `Finance`
- `Listing`
- `Trading`
- `Fund`
- constants from `constants.py`

### `Vnstock`

Primary chaining entrypoint that keeps the classic upstream usage style:

```python
from app.lib.vnstock_alt import Vnstock

stock = Vnstock().stock(symbol="VCB", source="VCI")
stock.quote.history(...)
stock.company.overview()
stock.finance.balance_sheet(...)
stock.trading.price_board([...])
```

### `Quote`

Methods:

- `history`
- `intraday`
- `price_depth`

Typical use:

```python
from app.lib.vnstock_alt import Quote

df = Quote(symbol="VCB", source="VCI").history(start="2024-01-01", end="2024-12-31")
```

### `Company`

Methods:

- `affiliate`
- `events`
- `history`
- `news`
- `officers`
- `overview`
- `shareholders`
- `subsidiaries`

Typical use:

```python
from app.lib.vnstock_alt import Company

company = Company(symbol="VCB", source="VCI")
overview = company.overview()
officers = company.officers()
```

### `Finance`

Methods:

- `balance_sheet`
- `cash_flow`
- `history`
- `income_statement`
- `ratio`

Typical use:

```python
from app.lib.vnstock_alt import Finance

finance = Finance(symbol="VCB", source="VCI")
income = finance.income_statement(period="quarter")
ratio = finance.ratio(period="year")
```

### `Listing`

Methods:

- `all_bonds`
- `all_covered_warrant`
- `all_future_indices`
- `all_government_bonds`
- `all_symbols`
- `history`
- `industries_icb`
- `symbols_by_exchange`
- `symbols_by_group`
- `symbols_by_industries`

Typical use:

```python
from app.lib.vnstock_alt import Listing

listing = Listing(source="VCI")
symbols = listing.all_symbols()
vn30 = listing.symbols_by_group(group="VN30")
industries = listing.industries_icb()
```

### `Trading`

Methods:

- `foreign_trade`
- `history`
- `insider_deal`
- `order_stats`
- `price_board`
- `price_history`
- `prop_trade`
- `side_stats`
- `trading_stats`

Typical use:

```python
from app.lib.vnstock_alt import Trading

trading = Trading(source="VCI")
board = trading.price_board(["VCB", "FPT"])
```

### `Fund`

Methods:

- `asset_holding`
- `filter`
- `industry_holding`
- `listing`
- `nav_report`
- `top_holding`

Typical use:

```python
from app.lib.vnstock_alt import Fund

fund = Fund()
funds = fund.listing()
nav = fund.nav_report(symbol="SSISCA")
```

## `vnstock_data_alt` root API

Root exports:

- `Quote`
- `Company`
- `Finance`
- `Listing`
- `Trading`
- `CommodityPrice`
- `TopStock`
- `Fund`
- `IndexGroup`
- lazy UI entrypoints:
  - `Reference`
  - `Market`
  - `Insights`
  - `Fundamental`
  - `Macro`
  - `Analytics`
- helper functions:
  - `show_api`
  - `show_doc`

### `Quote`

Methods:

- `history`
- `intraday`
- `price_depth`

### `Company`

Methods:

- `affiliate`
- `capital_history`
- `events`
- `history`
- `insider_trading`
- `news`
- `officers`
- `overview`
- `shareholders`
- `subsidiaries`

### `Finance`

Methods:

- `balance_sheet`
- `cash_flow`
- `history`
- `income_statement`
- `note`
- `ratio`

### `Listing`

Methods:

- `all_bonds`
- `all_covered_warrant`
- `all_etf`
- `all_future_indices`
- `all_government_bonds`
- `all_indices`
- `all_symbols`
- `history`
- `indices_by_group`
- `industries_icb`
- `symbols_by_exchange`
- `symbols_by_group`
- `symbols_by_industries`

### `Trading`

Methods:

- `foreign_trade`
- `history`
- `insider_deal`
- `matched_by_price`
- `odd_lot`
- `order_stats`
- `price_board`
- `price_history`
- `prop_trade`
- `put_through`
- `side_stats`
- `trade_history`
- `trading_stats`

### `CommodityPrice`

Methods:

- `coke`
- `corn`
- `fertilizer_ure`
- `gas_natural`
- `gas_vn`
- `gold_global`
- `gold_vn`
- `history`
- `iron_ore`
- `oil_crude`
- `pork_china`
- `pork_north_vn`
- `soybean`
- `steel_d10`
- `steel_hrc`
- `sugar`

### `TopStock`

Methods:

- `deal`
- `foreign_buy`
- `foreign_sell`
- `gainer`
- `history`
- `loser`
- `value`
- `volume`

### `Fund`

Methods:

- `asset_holding`
- `filter`
- `industry_holding`
- `listing`
- `nav_report`
- `top_holding`

## UI/domain entrypoints in `vnstock_data_alt`

These are lazily loaded from the root package:

- `Reference`
- `Market`
- `Insights`
- `Fundamental`
- `Macro`
- `Analytics`

Helper functions:

- `show_api()`
- `show_doc(obj)`

Useful discovery pattern:

```python
from app.lib.vnstock_data_alt import show_api, Reference, Market

show_api()

ref = Reference()
market = Market()
```

## Source and provider notes

The retained public API keeps upstream-style provider routing where applicable.

Common source examples:

- `vci`
- `kbs`
- `msn`
- `vnd`
- `vds`
- `tvs`
- `mas`
- `cafef`
- `spl`
- `mbk`
- `fmarket`

Not every class supports every provider. Follow the class docstrings and the
existing live compatibility tests when adding new usage patterns.

## Important intentional differences

- auth/API-key helpers are not part of the vendored public surface
- local package-side API-limit mechanics were not carried over
- bundled charting is intentionally disabled
- `vnstock_alt` suppresses upstream upgrade/auth chatter

## Recommended way to keep this doc accurate

Treat this guide as a curated map, not the only source of truth.

When the retained public surface changes:

1. update the compatibility tests first
2. update this guide's method inventory
3. add a short example only when the new surface is expected to be used often

If richer docs are needed later, the best next step is generating API reference
pages from docstrings with a tool like `pdoc`, `mkdocs`, or `Sphinx`, while
keeping this file as the human-oriented quick-start guide.
