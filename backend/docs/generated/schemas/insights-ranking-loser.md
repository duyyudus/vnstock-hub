# insights.ranking.loser

- Class: `RankingReference`
- Method: `loser`
- Signature: `(index = 'VNINDEX', limit = 10)`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `vnd`
- Declared signature: `(index='VNINDEX', limit=10, **B)`
- Default route source: `vnd`
- Default provider: `insight.TopStock.loser`

Top 10 stocks with highest price decrease.

## Purpose

Top 10 stocks with highest price decrease.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `index` | `POSITIONAL_OR_KEYWORD` | `False` | `VNINDEX` | `` | `omitted; default 'VNINDEX'` |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `10` | `` | `5` |

## Source details

### Source `vnd`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vnd.insight.TopStock`
- Provider method: `loser`

```text
symbol, index, last_price, last_updated, price_change_1d, price_change_pct_1d, accumulated_value
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `index` | `exchange` |
| `last_price` | `last_price` |
| `last_updated` | `last_updated` |
| `price_change_1d` | `price_change_1d` |
| `price_change_pct_1d` | `price_change_pct_1d` |
| `accumulated_value` | `total_value` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, exchange, last_price, last_updated, price_change_1d, price_change_pct_1d, total_value
```

Enum/value normalization:

- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

- Captured at: `2026-03-16T11:15:05.495804+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, exchange, last_price, last_updated, price_change_1d, price_change_pct_1d, total_value
```
- Dtypes: `{'symbol': 'str', 'exchange': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'total_value': 'float64'}`

```json
[
  {
    "symbol": "PVD",
    "exchange": "HOSE",
    "last_price": 37.2,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": -2.799999999999997,
    "price_change_pct_1d": -6.999999999999995,
    "total_value": 359934675000.0
  },
  {
    "symbol": "DCM",
    "exchange": "HOSE",
    "last_price": 44.55,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": -3.3500000000000014,
    "price_change_pct_1d": -6.993736951983298,
    "total_value": 368183510000.0
  },
  {
    "symbol": "VVS",
    "exchange": "HOSE",
    "last_price": 149.3,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": -11.199999999999989,
    "price_change_pct_1d": -6.978193146417433,
    "total_value": 22070820000.0
  }
]
```
