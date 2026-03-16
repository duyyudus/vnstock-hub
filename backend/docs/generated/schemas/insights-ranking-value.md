# insights.ranking.value

- Class: `RankingReference`
- Method: `value`
- Signature: `(index = 'VNINDEX', limit = 10)`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `vnd`
- Declared signature: `(index='VNINDEX', limit=10, **B)`
- Default route source: `vnd`
- Default provider: `insight.TopStock.value`

Top 10 stocks with highest trading value.

## Purpose

Top 10 stocks with highest trading value.

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
- Provider method: `value`

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

- Captured at: `2026-03-16T11:15:05.724698+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, exchange, last_price, last_updated, price_change_1d, price_change_pct_1d, total_value
```
- Dtypes: `{'symbol': 'str', 'exchange': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'total_value': 'float64'}`

```json
[
  {
    "symbol": "SHB",
    "exchange": "HOSE",
    "last_price": 15.2,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 0.25,
    "price_change_pct_1d": 1.6722408026755842,
    "total_value": 1258550880000.0
  },
  {
    "symbol": "SSI",
    "exchange": "HOSE",
    "last_price": 28.4,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 0.0,
    "price_change_pct_1d": 0.0,
    "total_value": 744127825000.0
  },
  {
    "symbol": "STB",
    "exchange": "HOSE",
    "last_price": 66.6,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 0.7999999999999972,
    "price_change_pct_1d": 1.2158054711246091,
    "total_value": 734224430000.0
  }
]
```
