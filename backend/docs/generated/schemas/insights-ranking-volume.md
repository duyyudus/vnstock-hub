# insights.ranking.volume

- Class: `RankingReference`
- Method: `volume`
- Signature: `(index = 'VNINDEX', limit = 10)`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `vnd`
- Declared signature: `(index='VNINDEX', limit=10, **B)`
- Default route source: `vnd`
- Default provider: `insight.TopStock.volume`

Top 10 stocks with highest volume spikes.

## Purpose

Top 10 stocks with highest volume spikes.

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
- Provider method: `volume`

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

- Captured at: `2026-03-16T11:15:05.862557+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, exchange, last_price, last_updated, price_change_1d, price_change_pct_1d, total_value, avg_volume_20d, volume_spike_20d_pct
```
- Dtypes: `{'symbol': 'str', 'exchange': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'total_value': 'float64', 'avg_volume_20d': 'float64', 'volume_spike_20d_pct': 'float64'}`

```json
[
  {
    "symbol": "PTL",
    "exchange": "HOSE",
    "last_price": 2.89,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 0.18000000000000016,
    "price_change_pct_1d": 6.6420664206642055,
    "total_value": 657445000.0,
    "avg_volume_20d": 35095.0,
    "volume_spike_20d_pct": 648.2404900983046
  },
  {
    "symbol": "TRA",
    "exchange": "HOSE",
    "last_price": 69.0,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": -0.9000000000000057,
    "price_change_pct_1d": -1.2875536480686733,
    "total_value": 5934880000.0,
    "avg_volume_20d": 16370.0,
    "volume_spike_20d_pct": 552.8405620036652
  },
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
  }
]
```
