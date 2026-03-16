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

- Captured at: `2026-03-16T11:15:05.352528+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, exchange, last_price, last_updated, price_change_1d, price_change_pct_1d, total_value, avg_volume_20d, volume_spike_20d_pct
```
- Dtypes: `{'symbol': 'str', 'exchange': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'total_value': 'float64', 'avg_volume_20d': 'float64', 'volume_spike_20d_pct': 'float64'}`

```json
[
  {
    "symbol": "MCH",
    "exchange": "HOSE",
    "last_price": 149.8,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 9.800000000000011,
    "price_change_pct_1d": 7.000000000000006,
    "total_value": 221697820000.0,
    "avg_volume_20d": 307070.0,
    "volume_spike_20d_pct": 482.88663822581174
  },
  {
    "symbol": "VCK",
    "exchange": "HOSE",
    "last_price": 33.4,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 2.1499999999999986,
    "price_change_pct_1d": 6.879999999999997,
    "total_value": 357437005000.0,
    "avg_volume_20d": 4083135.0,
    "volume_spike_20d_pct": 265.87413837651707
  },
  {
    "symbol": "NVL",
    "exchange": "HOSE",
    "last_price": 13.5,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 0.8499999999999996,
    "price_change_pct_1d": 6.719367588932812,
    "total_value": 656318670000.0,
    "avg_volume_20d": 10664010.0,
    "volume_spike_20d_pct": 459.486628388383
  }
]
```
