# VNStock Usage Snapshot (Backend)

## Purpose
This document is a migration-focused snapshot of how backend code currently uses `vnstock`. It records the exact classes/methods in use, the backend-consumed response shape (especially DataFrame columns), and minimal examples.

Primary goal: if `source='VCI'` (or any `vnstock` surface) changes or becomes unavailable, this file lets us replace data providers with minimal guesswork.

Maintenance guidance:
- Treat this file as a contract of what backend needs, not full upstream API docs.
- Update it when backend usage changes (new methods, removed methods, new required columns, changed fallback mapping).

## Snapshot Metadata
- Snapshot date: `2026-02-14`
- Code snapshot (HEAD short SHA): `c75e1db`
- Backend dependency intent: `vnstock>=3.4.2` in `backend/pyproject.toml:27`
- Locked version observed: `vnstock-3.4.2` in `backend/uv.lock`

## Current Dependency Context
- Backend wraps `vnstock` in `backend/app/services/vnstock_service`.
- Most stock/index/company/finance calls use `source='VCI'`.
- Fund calls use `Fund()` from Fmarket path (still part of `vnstock` dependency surface).
- Backend normalizes many payloads (flatten columns, alias mapping, NaN cleanup), so replacement providers must satisfy backend-consumed fields.

## Usage Inventory (Class/Method Matrix)

| # | Method | Backend service usage (example call site) |
|---|---|---|
| 1 | `vnstock.change_api_key(api_key)` | `backend/app/services/vnstock_service/__init__.py:33` |
| 2 | `Listing(source='VCI').all_indices()` | `backend/app/services/vnstock_service/indices.py:141` |
| 3 | `Listing(source='VCI').industries_icb()` | `backend/app/services/vnstock_service/stocks.py:100` |
| 4 | `Listing(source='VCI').symbols_by_industries()` | `backend/app/services/vnstock_service/stocks.py:136` |
| 5 | `Listing(source='VCI').symbols_by_group(group_code)` | `backend/app/services/vnstock_service/stocks.py:185` |
| 6 | `Listing(source='VCI').all_symbols()` | `backend/app/services/vnstock_service/stock_metadata.py:275` |
| 7 | `Trading(source='VCI').price_board(symbols)` | `backend/app/services/vnstock_service/stocks.py:274` |
| 8 | `Vnstock(...).stock(...).quote.history(...)` | `backend/app/services/vnstock_service/history.py:290` |
| 9 | `Vnstock(...).stock(...).finance.income_statement(period, lang)` | `backend/app/services/vnstock_service/finance.py:321` |
| 10 | `Vnstock(...).stock(...).finance.balance_sheet(period, lang)` | `backend/app/services/vnstock_service/finance.py:342` |
| 11 | `Vnstock(...).stock(...).finance.cash_flow(period, lang)` | `backend/app/services/vnstock_service/finance.py:363` |
| 12 | `Vnstock(...).stock(...).finance.ratio(period, lang)` | `backend/app/services/vnstock_service/finance.py:384` |
| 13 | `Company(symbol, source='VCI').overview()` | `backend/app/services/vnstock_service/company.py:290` |
| 14 | `Company(symbol, source='VCI').shareholders()` | `backend/app/services/vnstock_service/company.py:309` |
| 15 | `Company(symbol, source='VCI').officers()` | `backend/app/services/vnstock_service/company.py:328` |
| 16 | `Company(symbol, source='VCI').subsidiaries()` | `backend/app/services/vnstock_service/company.py:371` |
| 17 | `Fund().listing()` | `backend/app/services/vnstock_service/funds.py:253` |
| 18 | `Fund().details.nav_report(symbol)` | `backend/app/services/vnstock_service/funds.py:397` |
| 19 | `Fund().details.top_holding(symbol)` | `backend/app/services/vnstock_service/funds.py:555` |
| 20 | `Fund().details.industry_holding(symbol)` | `backend/app/services/vnstock_service/funds.py:602` |
| 21 | `Fund().details.asset_holding(symbol)` | `backend/app/services/vnstock_service/funds.py:648` |

## Detailed Method Contracts

### 1. `vnstock.change_api_key(api_key)`
- Method signature (as used): `vnstock.change_api_key(settings.vnstock_api_key)`
- Backend usage: optional one-time API key setup in service bootstrap.
- Call sites:
  - `backend/app/services/vnstock_service/__init__.py:33`
- Return/DataFrame contract:
  - Required: boolean success (`True`/`False`).
  - Optional/fallback: none.
- Backend normalization:
  - No transformation; result only affects logging flow.
- Minimal example:
```python
import vnstock
ok = vnstock.change_api_key("your_api_key")
```
- Compact example output: `True`

### 2. `Listing(source='VCI').all_indices()`
- Method signature (as used): `Listing(source='VCI').all_indices()`
- Backend usage: sync supported indices into DB.
- Call sites:
  - `backend/app/services/vnstock_service/indices.py:141`
- Return/DataFrame contract:
  - Required columns used: `symbol`.
  - Optional/fallback columns used: `full_name` (preferred display name), `name`, `index_name`, `group`, `description`.
- Backend normalization:
  - Name priority: `full_name` -> `name` -> `index_name` -> `symbol`.
  - Filters to supported groups via `get_group_code_for_index` mapping.
- Minimal example:
```python
from vnstock import Listing
df = Listing(source='VCI').all_indices()
```
- Compact example output:

| symbol | full_name | group | description |
|---|---|---|---|
| VN30 | VN30 Index | HOSE | Top 30 HOSE stocks |

### 3. `Listing(source='VCI').industries_icb()`
- Method signature (as used): `Listing(source='VCI').industries_icb()`
- Backend usage: build industry list endpoint.
- Call sites:
  - `backend/app/services/vnstock_service/stocks.py:100`
- Return/DataFrame contract:
  - Required columns used: `level`, `icb_name`, `en_icb_name`, `icb_code`.
  - Optional/fallback: none.
- Backend normalization:
  - Filters `level == 2`, returns selected columns only.
- Minimal example:
```python
from vnstock import Listing
df = Listing(source='VCI').industries_icb()
```
- Compact example output:

| icb_name | en_icb_name | icb_code | level |
|---|---|---|---|
| Ngan hang | Banks | 8300 | 2 |

### 4. `Listing(source='VCI').symbols_by_industries()`
- Method signature (as used): `Listing(source='VCI').symbols_by_industries()`
- Backend usage: industry stock filtering + symbol-to-industry cache map.
- Call sites:
  - `backend/app/services/vnstock_service/stocks.py:136`
  - `backend/app/services/vnstock_service/stocks.py:225`
- Return/DataFrame contract:
  - Required columns used: `symbol`.
  - Optional/fallback columns used: `icb_name2`, `icb_name3`, `icb_name4`.
- Backend normalization:
  - Match industry name against level 2/3/4 columns.
  - Build cache map: `symbol.upper() -> icb_name2`.
- Minimal example:
```python
from vnstock import Listing
df = Listing(source='VCI').symbols_by_industries()
```
- Compact example output:

| symbol | icb_name2 | icb_name3 | icb_name4 |
|---|---|---|---|
| TCB | Banks | Commercial Banks | Large Banks |

### 5. `Listing(source='VCI').symbols_by_group(group_code)`
- Method signature (as used): `Listing(source='VCI').symbols_by_group(group_code)`
- Backend usage: resolve index/group universe for stock/sync flows.
- Call sites:
  - `backend/app/services/vnstock_service/stocks.py:185`
  - `backend/app/services/vnstock_service/price_sync.py:931`
  - `backend/app/services/vnstock_service/finance_sync.py:382`
  - `backend/app/services/vnstock_service/company_sync.py:380`
- Return/DataFrame contract:
  - Required: symbol series convertible by `.tolist()`.
  - Optional/fallback: none.
- Backend normalization:
  - Group code is mapped/validated first.
  - Symbols normalized to uppercase and deduplicated downstream.
- Minimal example:
```python
from vnstock import Listing
symbols = Listing(source='VCI').symbols_by_group('VN30').tolist()
```
- Compact example output: `['ACB', 'FPT']`

### 6. `Listing(source='VCI').all_symbols()`
- Method signature (as used): `Listing(source='VCI').all_symbols()`
- Backend usage: symbol universe for metadata/company/finance/price sync.
- Call sites:
  - `backend/app/services/vnstock_service/stock_metadata.py:275`
  - `backend/app/services/vnstock_service/price_sync.py:1045`
  - `backend/app/services/vnstock_service/finance_sync.py:407`
  - `backend/app/services/vnstock_service/company_sync.py:405`
- Return/DataFrame contract:
  - Required columns used: `symbol`.
  - Optional/fallback columns used: `organ_name`, and listing-date candidates (`listing_date`, `listed_date`, `listing_first_trade_date`, `first_trade_date`, `ipo_date`, `trading_date`).
- Backend normalization:
  - Metadata enricher maps `symbol -> organ_name`.
  - Price sync scans candidate listing-date columns and parses first valid date.
- Minimal example:
```python
from vnstock import Listing
df = Listing(source='VCI').all_symbols()
```
- Compact example output:

| symbol | organ_name | listing_date |
|---|---|---|
| VCB | Vietcombank | 2009-06-30 |

### 7. `Trading(source='VCI').price_board(symbols)`
- Method signature (as used): `Trading(source='VCI').price_board(batch_symbols)`
- Backend usage: live stock board data for index/industry/symbol endpoints.
- Call sites:
  - `backend/app/services/vnstock_service/stocks.py:274`
- Return/DataFrame contract:
  - Required columns after flattening: `listing_symbol`, `match_match_price`.
  - Optional/fallback columns used: `listing_listed_share`, `listing_charter_capital`, `financial_pe`, `match_accumulated_value`, `match_foreign_buy_value`, `match_foreign_sell_value`, `match_current_room`, `match_total_room`, `match_price_change_ratio`, `listing_ref_price`, `listing_exchange`, `listing_organ_name`.
- Backend normalization:
  - Flattens multi-index columns with `_flatten_columns`.
  - Converts units (e.g., VND -> billion VND for selected fields).
  - Computes fallback `% change` from `listing_ref_price` if ratio missing.
- Minimal example:
```python
from vnstock import Trading
df = Trading(source='VCI').price_board(['VCB', 'TCB'])
```
- Compact example output (flattened):

| listing_symbol | match_match_price | listing_listed_share | financial_pe |
|---|---:|---:|---:|
| VCB | 91500 | 5589137101 | 13.2 |

### 8. `Vnstock(...).stock(...).quote.history(...)`
- Method signature (as used):
  - `Vnstock().stock(symbol=s, source='VCI').quote.history(start, end, interval='1D')`
  - `Vnstock(symbol=i, source='VCI').stock().quote.history(start, end)`
- Backend usage: price sync, benchmark/index values, fund benchmark history.
- Call sites:
  - `backend/app/services/vnstock_service/history.py:290`
  - `backend/app/services/vnstock_service/history.py:864`
  - `backend/app/services/vnstock_service/indices.py:196`
  - `backend/app/services/vnstock_service/price_sync.py:963`
  - `backend/app/services/vnstock_service/funds.py:1315`
- Return/DataFrame contract:
  - Required columns across backend: `time`, `close`.
  - Additional required in some paths: `open`, `high`, `low`, `volume`.
- Backend normalization:
  - Parses `time` to `date` and deduplicates by date.
  - Resamples weekly where needed.
  - Converts numeric fields for DB upsert.
- Minimal example:
```python
from vnstock import Vnstock
df = Vnstock().stock(symbol='VCB', source='VCI').quote.history(
    start='2026-01-01', end='2026-02-14', interval='1D'
)
```
- Compact example output:

| time | open | high | low | close | volume |
|---|---:|---:|---:|---:|---:|
| 2026-02-13 | 91000 | 92000 | 90500 | 91500 | 3210000 |

### 9. `Vnstock(...).stock(...).finance.income_statement(period, lang)`
- Method signature (as used): `s.finance.income_statement(period='quarter', lang='en')`
- Backend usage: finance dataset cache refresh.
- Call sites:
  - `backend/app/services/vnstock_service/finance.py:321`
- Return/DataFrame contract:
  - Required: DataFrame convertible to records.
  - Optional/fallback: dynamic report columns (schema may vary by symbol/time).
- Backend normalization:
  - Flattens multi-index columns.
  - Serializes JSON-safe records and stores as generic list of dicts.
- Minimal example:
```python
from vnstock import Vnstock
df = Vnstock().stock(symbol='VCB', source='VCI').finance.income_statement(period='quarter', lang='en')
```
- Compact example output:

| ticker | yearReport | lengthReport | Revenue |
|---|---:|---:|---:|
| VCB | 2025 | 4 | 15000000000000 |

### 10. `Vnstock(...).stock(...).finance.balance_sheet(period, lang)`
- Method signature (as used): `s.finance.balance_sheet(period='quarter', lang='en')`
- Backend usage: finance dataset cache refresh.
- Call sites:
  - `backend/app/services/vnstock_service/finance.py:342`
- Return/DataFrame contract:
  - Required: DataFrame convertible to records.
  - Optional/fallback: dynamic columns.
- Backend normalization:
  - Same as income statement normalization path.
- Minimal example:
```python
from vnstock import Vnstock
df = Vnstock().stock(symbol='VCB', source='VCI').finance.balance_sheet(period='quarter', lang='en')
```
- Compact example output:

| ticker | yearReport | lengthReport | Total Assets |
|---|---:|---:|---:|
| VCB | 2025 | 4 | 2200000000000000 |

### 11. `Vnstock(...).stock(...).finance.cash_flow(period, lang)`
- Method signature (as used): `s.finance.cash_flow(period='quarter', lang='en')`
- Backend usage: finance dataset cache refresh.
- Call sites:
  - `backend/app/services/vnstock_service/finance.py:363`
- Return/DataFrame contract:
  - Required: DataFrame convertible to records.
  - Optional/fallback: dynamic columns.
- Backend normalization:
  - Same flatten + JSON-safe record flow.
- Minimal example:
```python
from vnstock import Vnstock
df = Vnstock().stock(symbol='VCB', source='VCI').finance.cash_flow(period='quarter', lang='en')
```
- Compact example output:

| ticker | yearReport | lengthReport | Operating Cash Flow |
|---|---:|---:|---:|
| VCB | 2025 | 4 | 31000000000000 |

### 12. `Vnstock(...).stock(...).finance.ratio(period, lang)`
- Method signature (as used): `s.finance.ratio(period='quarter', lang='en')`
- Backend usage: finance dataset cache + metadata PE extraction.
- Call sites:
  - `backend/app/services/vnstock_service/finance.py:384`
- Return/DataFrame contract:
  - Required: DataFrame records containing at least one P/E-like key for PE extraction path.
  - Optional/fallback keys accepted by extractor: `P/E`, `PE`, `P_E`, `*_PE`.
- Backend normalization:
  - Flattened records persisted; PE parsed from first record by tolerant key matching.
- Minimal example:
```python
from vnstock import Vnstock
df = Vnstock().stock(symbol='VCB', source='VCI').finance.ratio(period='quarter', lang='en')
```
- Compact example output:

| ticker | yearReport | lengthReport | P/E |
|---|---:|---:|---:|
| VCB | 2025 | 4 | 13.2 |

### 13. `Company(symbol, source='VCI').overview()`
- Method signature (as used): `Company(symbol=s, source='VCI').overview()`
- Backend usage: cached company overview API.
- Call sites:
  - `backend/app/services/vnstock_service/company.py:290`
- Return/DataFrame contract:
  - Required: DataFrame/dict/list payload convertible into records.
  - Optional/fallback: dynamic company fields.
- Backend normalization:
  - Converts dict/list to DataFrame if needed.
  - Flattens columns; serializes records with NaN/date normalization.
- Minimal example:
```python
from vnstock import Company
df = Company(symbol='VCB', source='VCI').overview()
```
- Compact example output:

| symbol | organ_name | charter_capital |
|---|---|---:|
| VCB | Vietcombank | 55891371010000 |

### 14. `Company(symbol, source='VCI').shareholders()`
- Method signature (as used): `Company(symbol=s, source='VCI').shareholders()`
- Backend usage: cached shareholders API.
- Call sites:
  - `backend/app/services/vnstock_service/company.py:309`
- Return/DataFrame contract:
  - Required: payload convertible to records.
  - Optional/fallback: shareholder columns vary.
- Backend normalization:
  - Generic flatten + JSON-safe serialization.
- Minimal example:
```python
from vnstock import Company
df = Company(symbol='VCB', source='VCI').shareholders()
```
- Compact example output:

| share_holder | share_own_percent | update_date |
|---|---:|---|
| State Bank | 74.8 | 2026-01-15 |

### 15. `Company(symbol, source='VCI').officers()`
- Method signature (as used): `Company(symbol=s, source='VCI').officers()`
- Backend usage: cached officers API.
- Call sites:
  - `backend/app/services/vnstock_service/company.py:328`
- Return/DataFrame contract:
  - Required: payload convertible to records.
  - Optional/fallback: officer fields vary.
- Backend normalization:
  - Generic flatten + JSON-safe serialization.
- Minimal example:
```python
from vnstock import Company
df = Company(symbol='VCB', source='VCI').officers()
```
- Compact example output:

| officer_name | officer_position | officer_own_percent |
|---|---|---:|
| Nguyen Van A | CEO | 0.02 |

### 16. `Company(symbol, source='VCI').subsidiaries()`
- Method signature (as used): `Company(symbol=s, source='VCI').subsidiaries()`
- Backend usage: cached subsidiaries API.
- Call sites:
  - `backend/app/services/vnstock_service/company.py:371`
- Return/DataFrame contract:
  - Required: payload convertible to records.
  - Optional/fallback: variable subsidiary fields.
- Backend normalization:
  - Known upstream edge case: missing `organ_code` can be treated as "no data" and return `[]`.
- Minimal example:
```python
from vnstock import Company
df = Company(symbol='VCB', source='VCI').subsidiaries()
```
- Compact example output:

| sub_organ_code | organ_name | ownership_percent |
|---|---|---:|
| BIZ123 | Subsidiary Co | 51.0 |

### 17. `Fund().listing()`
- Method signature (as used): `fund = Fund(); fund.listing()`
- Backend usage: fund listing endpoint + sync bootstrap.
- Call sites:
  - `backend/app/services/vnstock_service/funds.py:253`
  - `backend/app/services/vnstock_service/funds.py:861`
- Return/DataFrame contract:
  - Required columns used after normalization: one of `short_name|fund_code|symbol` for symbol identity.
  - Optional/fallback columns used: `name`, `fund_type`, `fund_owner_name|management_company`.
- Backend normalization:
  - Flattens columns; maps aliases into canonical keys: `symbol`, `name`, `fund_owner`.
- Minimal example:
```python
from vnstock import Fund
df = Fund().listing()
```
- Compact example output:

| short_name | name | fund_type | fund_owner_name |
|---|---|---|---|
| SSISCA | SSI-SCA | STOCK | SSIAM |

### 18. `Fund().details.nav_report(symbol)`
- Method signature (as used): `Fund().details.nav_report(symbol=symbol)`
- Backend usage: NAV history endpoint + background NAV sync.
- Call sites:
  - `backend/app/services/vnstock_service/funds.py:397`
  - `backend/app/services/vnstock_service/funds.py:476`
  - `backend/app/services/vnstock_service/funds.py:932`
- Return/DataFrame contract:
  - Required via tolerant alias: date field from `date|nav_date`.
  - Required via tolerant alias: NAV field from `nav|nav_per_unit|value`.
- Backend normalization:
  - Parses date to `date`, NAV to float, dedup by date against DB.
  - Weekly resampling used for response payload.
- Minimal example:
```python
from vnstock import Fund
df = Fund().details.nav_report(symbol='SSISCA')
```
- Compact example output:

| date | nav_per_unit |
|---|---:|
| 2026-02-13 | 15432.1 |

### 19. `Fund().details.top_holding(symbol)`
- Method signature (as used): `Fund().details.top_holding(symbol=symbol)`
- Backend usage: fund top-holdings endpoint.
- Call sites:
  - `backend/app/services/vnstock_service/funds.py:555`
- Return/DataFrame contract:
  - Required via tolerant alias: ticker from `stock_code|ticker|symbol`.
  - Required via tolerant alias: allocation from `net_asset_percent|allocation|weight|percentage`.
- Backend normalization:
  - Flattens columns; maps aliases; replaces NaN with `None`.
- Minimal example:
```python
from vnstock import Fund
df = Fund().details.top_holding(symbol='SSISCA')
```
- Compact example output:

| stock_code | net_asset_percent |
|---|---:|
| FPT | 8.5 |

### 20. `Fund().details.industry_holding(symbol)`
- Method signature (as used): `Fund().details.industry_holding(symbol=symbol)`
- Backend usage: fund industry-allocation endpoint.
- Call sites:
  - `backend/app/services/vnstock_service/funds.py:602`
- Return/DataFrame contract:
  - Required: industry descriptor column(s) and allocation alias.
  - Optional/fallback allocation alias: `net_asset_percent|allocation|weight|percentage`.
- Backend normalization:
  - Flattens columns; maps allocation alias; NaN -> `None`.
- Minimal example:
```python
from vnstock import Fund
df = Fund().details.industry_holding(symbol='SSISCA')
```
- Compact example output:

| industry | net_asset_percent |
|---|---:|
| Banking | 27.4 |

### 21. `Fund().details.asset_holding(symbol)`
- Method signature (as used): `Fund().details.asset_holding(symbol=symbol)`
- Backend usage: fund asset-allocation endpoint.
- Call sites:
  - `backend/app/services/vnstock_service/funds.py:648`
- Return/DataFrame contract:
  - Required: asset type descriptor and allocation alias.
  - Optional/fallback allocation alias: `asset_percent|allocation|weight|percentage`.
- Backend normalization:
  - Flattens columns; maps allocation alias; NaN -> `None`.
- Minimal example:
```python
from vnstock import Fund
df = Fund().details.asset_holding(symbol='SSISCA')
```
- Compact example output:

| asset_type | asset_percent |
|---|---:|
| Equity | 92.1 |

## Migration Notes
- `all_indices()` is a provider-level extension exposed via VCI provider path; treat this as a high-risk dependency when replacing source.
- Finance/company payload schemas are dynamic; backend persists mostly generic flattened records, so downstream code should not assume stable full schemas.
- Fund data is fetched through `Fund()` (Fmarket path), not `source='VCI'`, but it is still part of backend `vnstock` dependency risk.
- Backend depends on tolerant fallback mapping in multiple places (industry names, listing names, NAV aliases, allocation aliases, price-board fallbacks). Preserve this tolerance in replacement adapters.
- Assumptions/defaults for this snapshot:
  - Scope is backend runtime usage only.
  - Documented format is backend-consumed contract, not full upstream schema.
  - Includes both request-path and background sync usage.
  - Includes fund methods although non-VCI.

## Update Procedure
1. Re-run usage scan for `from vnstock import` and method call sites under `backend/app/services/vnstock_service`.
2. Confirm method signatures against installed `vnstock` package in backend virtual env.
3. Re-validate required and fallback columns against actual backend consumption logic.
4. Update snapshot metadata date and commit SHA.
5. Add newly introduced methods and remove deprecated/unused methods.

Validation scenarios:
- Coverage validation: all 21 methods above must appear once in matrix and once in detailed contracts.
- Call-site validation: each method section must contain at least one concrete backend path+line reference.
- Schema validation: every DataFrame column accessed by backend from `vnstock` payloads must be documented in required/optional fields.
- File validation: `backend/docs/vnstock_usage_snapshot.md` exists and headings match this structure.
