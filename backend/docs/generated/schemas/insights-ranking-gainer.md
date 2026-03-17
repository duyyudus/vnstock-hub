# insights.ranking.gainer

- Class: `RankingReference`
- Method: `gainer`
- Signature: `(index = 'VNINDEX', limit = 10)`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `vnd`
- Declared signature: `(index='VNINDEX', limit=10, **B)`
- Default route source: `vnd`
- Default provider: `insight.TopStock.gainer`

Top 10 stocks with highest price increase.

## Purpose

Top 10 stocks with highest price increase.

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
- Provider method: `gainer`

```text
symbol, index, last_price, last_updated, price_change_1d, price_change_pct_1d, accumulated_value, avg_volume_20d, volume_spike_20d_pct
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
| `avg_volume_20d` | `avg_volume_20d` |
| `volume_spike_20d_pct` | `volume_spike_20d_pct` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, exchange, last_price, last_updated, price_change_1d, price_change_pct_1d, total_value, avg_volume_20d, volume_spike_20d_pct
```

Enum/value normalization:

- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

- Captured at: `2026-03-17T05:26:51.813648+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, exchange, last_price, last_updated, price_change_1d, price_change_pct_1d, total_value, avg_volume_20d, volume_spike_20d_pct
```
- Dtypes: `{'symbol': 'str', 'exchange': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'total_value': 'float64', 'avg_volume_20d': 'float64', 'volume_spike_20d_pct': 'float64'}`

```json
[
  {
    "symbol": "GEE",
    "exchange": "HOSE",
    "last_price": 155.1,
    "last_updated": "2026-03-17 12:26",
    "price_change_1d": 10.099999999999994,
    "price_change_pct_1d": 6.965517241379304,
    "total_value": 178933220000.0,
    "avg_volume_20d": 801425.0,
    "volume_spike_20d_pct": 145.04164457060858
  },
  {
    "symbol": "NO1",
    "exchange": "HOSE",
    "last_price": 5.99,
    "last_updated": "2026-03-17 12:26",
    "price_change_1d": 0.39000000000000057,
    "price_change_pct_1d": 6.964285714285734,
    "total_value": 538024000.0,
    "avg_volume_20d": 35220.0,
    "volume_spike_20d_pct": 257.8080636002271
  },
  {
    "symbol": "PTL",
    "exchange": "HOSE",
    "last_price": 3.09,
    "last_updated": "2026-03-17 12:26",
    "price_change_1d": 0.19999999999999973,
    "price_change_pct_1d": 6.92041522491349,
    "total_value": 279027000.0,
    "avg_volume_20d": 38530.0,
    "volume_spike_20d_pct": 234.36283415520376
  }
]
```
