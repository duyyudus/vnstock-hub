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

- Captured at: `2026-03-17T05:26:52.499920+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, exchange, last_price, last_updated, price_change_1d, price_change_pct_1d, total_value, avg_volume_20d, volume_spike_20d_pct
```
- Dtypes: `{'symbol': 'str', 'exchange': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'total_value': 'float64', 'avg_volume_20d': 'float64', 'volume_spike_20d_pct': 'float64'}`

```json
[
  {
    "symbol": "APG",
    "exchange": "HOSE",
    "last_price": 6.55,
    "last_updated": "2026-03-17 12:26",
    "price_change_1d": -0.08000000000000007,
    "price_change_pct_1d": -1.2066365007541435,
    "total_value": 74535602000.0,
    "avg_volume_20d": 769010.0,
    "volume_spike_20d_pct": 1558.4972887218632
  },
  {
    "symbol": "MCH",
    "exchange": "HOSE",
    "last_price": 157.2,
    "last_updated": "2026-03-17 12:26",
    "price_change_1d": 7.399999999999977,
    "price_change_pct_1d": 4.939919893190914,
    "total_value": 196749260000.0,
    "avg_volume_20d": 352855.0,
    "volume_spike_20d_pct": 358.6459027079112
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
  }
]
```
