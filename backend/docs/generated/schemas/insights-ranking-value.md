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

- Captured at: `2026-03-17T05:26:52.100948+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, exchange, last_price, last_updated, price_change_1d, price_change_pct_1d, total_value
```
- Dtypes: `{'symbol': 'str', 'exchange': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'total_value': 'float64'}`

```json
[
  {
    "symbol": "VIX",
    "exchange": "HOSE",
    "last_price": 17.1,
    "last_updated": "2026-03-17 12:26",
    "price_change_1d": 0.9000000000000021,
    "price_change_pct_1d": 5.555555555555558,
    "total_value": 531107880000.0
  },
  {
    "symbol": "SSI",
    "exchange": "HOSE",
    "last_price": 29.15,
    "last_updated": "2026-03-17 12:26",
    "price_change_1d": 0.75,
    "price_change_pct_1d": 2.640845070422526,
    "total_value": 501466090000.0
  },
  {
    "symbol": "FPT",
    "exchange": "HOSE",
    "last_price": 80.3,
    "last_updated": "2026-03-17 12:26",
    "price_change_1d": 2.0999999999999943,
    "price_change_pct_1d": 2.6854219948849067,
    "total_value": 414958600000.0
  }
]
```
