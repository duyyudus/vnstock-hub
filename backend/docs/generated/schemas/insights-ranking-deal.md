# insights.ranking.deal

- Class: `RankingReference`
- Method: `deal`
- Signature: `(index = 'VNINDEX', limit = 10)`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `vnd`
- Declared signature: `(index='VNINDEX', limit=10, **B)`
- Default route source: `vnd`
- Default provider: `insight.TopStock.deal`

Top 10 stocks with highest put-through/deal volume spikes.

## Purpose

Top 10 stocks with highest put-through/deal volume spikes.

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
- Provider method: `deal`

```text
symbol, index, last_price, last_updated, price_change_1d, price_change_pct_1d, accumulated_value, deal_volume_spike_20d_pct
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
| `deal_volume_spike_20d_pct` | `deal_volume_spike_20d_pct` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, exchange, last_price, last_updated, price_change_1d, price_change_pct_1d, total_value, deal_volume_spike_20d_pct
```

Enum/value normalization:

- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

- Captured at: `2026-03-17T05:26:50.071508+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, exchange, last_price, last_updated, price_change_1d, price_change_pct_1d, total_value, deal_volume_spike_20d_pct
```
- Dtypes: `{'symbol': 'str', 'exchange': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'total_value': 'float64', 'deal_volume_spike_20d_pct': 'float64'}`

```json
[
  {
    "symbol": "SMC",
    "exchange": "HOSE",
    "last_price": 11.3,
    "last_updated": "2026-03-17 12:26",
    "price_change_1d": 0.3000000000000007,
    "price_change_pct_1d": 2.7272727272727337,
    "total_value": 1059515000.0,
    "deal_volume_spike_20d_pct": 83.95776924207124
  },
  {
    "symbol": "SSB",
    "exchange": "HOSE",
    "last_price": 16.65,
    "last_updated": "2026-03-17 12:26",
    "price_change_1d": 0.09999999999999787,
    "price_change_pct_1d": 0.6042296072507503,
    "total_value": 10733495000.0,
    "deal_volume_spike_20d_pct": 25.956526023420157
  },
  {
    "symbol": "ACC",
    "exchange": "HOSE",
    "last_price": 12.0,
    "last_updated": "2026-03-17 12:26",
    "price_change_1d": -0.5,
    "price_change_pct_1d": -4.0000000000000036,
    "total_value": 333760000.0,
    "deal_volume_spike_20d_pct": 24.217472906702188
  }
]
```
