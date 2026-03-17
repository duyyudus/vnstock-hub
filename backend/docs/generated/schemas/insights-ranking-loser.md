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

- Captured at: `2026-03-17T05:26:51.960443+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, exchange, last_price, last_updated, price_change_1d, price_change_pct_1d, total_value
```
- Dtypes: `{'symbol': 'str', 'exchange': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'total_value': 'float64'}`

```json
[
  {
    "symbol": "BSR",
    "exchange": "HOSE",
    "last_price": 31.05,
    "last_updated": "2026-03-17 12:26",
    "price_change_1d": -1.6999999999999993,
    "price_change_pct_1d": -5.190839694656491,
    "total_value": 315492080000.0
  },
  {
    "symbol": "BFC",
    "exchange": "HOSE",
    "last_price": 60.0,
    "last_updated": "2026-03-17 12:26",
    "price_change_1d": -2.5,
    "price_change_pct_1d": -4.0000000000000036,
    "total_value": 10581230000.0
  },
  {
    "symbol": "ACC",
    "exchange": "HOSE",
    "last_price": 12.0,
    "last_updated": "2026-03-17 12:26",
    "price_change_1d": -0.5,
    "price_change_pct_1d": -4.0000000000000036,
    "total_value": 333760000.0
  }
]
```
