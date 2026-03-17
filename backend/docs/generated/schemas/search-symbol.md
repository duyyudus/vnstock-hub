# search.symbol

- Class: `SearchReference`
- Method: `symbol`
- Signature: `(query, locale=None, limit=10)`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `msn`
- Default route source: `msn`
- Default provider: `listing.Listing.search_symbol_id`

Retrieves a list of symbols from the market matching the query.

## Purpose

Retrieves a list of symbols from the market matching the query.
Backed by MSN Autosuggest to find global and local financial instruments.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `query` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `str` | `sample` |  | Search keyword to find the symbol. |
| `locale` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `str, optional` | `omitted in live probe` | `vi-vn`, `en-us` | Target language/locale to filter results (e.g., 'vi-vn', 'en-us'). |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `10` | `int, optional` | `5` |  | Max number of results. Defaults to 10. |

## Source details

### Source `msn`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_alt.explorer.msn.listing.Listing`
- Provider method: `search_symbol_id`

```text
symbol, symbol_id, exchange_name, short_name, friendly_name, eng_name, description, local_name
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `symbol_id` | `symbol_id` |
| `exchange_name` | `exchange` |
| `short_name` | `short_name` |
| `friendly_name` | `name` |
| `eng_name` | `name_en` |
| `description` | `description` |
| `local_name` | `name_local` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, name, exchange, short_name, description, name_en, name_local, symbol_id
```

Enum/value normalization:

- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

- Captured at: `2026-03-17T05:27:09.260820+00:00`
- Success: `True`
- Row count: `2`

```text
symbol, name, exchange, short_name, description, name_en, name_local, symbol_id
```
- Dtypes: `{'symbol': 'str', 'name': 'str', 'exchange': 'str', 'short_name': 'str', 'description': 'str', 'name_en': 'str', 'name_local': 'str', 'symbol_id': 'str'}`

```json
[
  {
    "symbol": "1708",
    "name": "Sample Tech",
    "exchange": "Hong Kong",
    "short_name": "Sample Tech",
    "description": "Nanjing Sample Technology Co Ltd is a China-based company primarily engaged in the provision of system integration services. The Company mainly operates two businesses. The intelligent transportation business is engaged in the development and application in areas such as highway toll systems, command and dispatch systems, tunnel control systems, and guidance systems. The smart logistics business is engaged in providing comprehensive intelligent solutions including information planning, software products, hardware products, information system integration, and operation and maintenance services for logistics customers such as customs, special customs supervision zones (including bonded zones and cross-border comprehensive experimental zones), port terminals and airports. The Company conducts its business in the domestic market.",
    "name_en": "Nanjing Sample Technology Co Ltd",
    "name_local": "三寶科技",
    "symbol_id": "ah34jc"
  },
  {
    "symbol": "",
    "name": "Sample Growth Fund;A",
    "exchange": "",
    "short_name": "Sample Growth Fund;A",
    "description": "",
    "name_en": "Sample Growth A",
    "name_local": "Sample Growth A",
    "symbol_id": "cax13m"
  }
]
```
