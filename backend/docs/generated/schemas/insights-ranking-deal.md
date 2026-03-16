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

- Captured at: `2026-03-16T11:15:04.753952+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, exchange, last_price, last_updated, price_change_1d, price_change_pct_1d, total_value, deal_volume_spike_20d_pct
```
- Dtypes: `{'symbol': 'str', 'exchange': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'total_value': 'float64', 'deal_volume_spike_20d_pct': 'float64'}`

```json
[
  {
    "symbol": "TVB",
    "exchange": "HOSE",
    "last_price": 7.34,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 0.0,
    "price_change_pct_1d": 0.0,
    "total_value": 35586000.0,
    "deal_volume_spike_20d_pct": 667.7400282087873
  },
  {
    "symbol": "SAM",
    "exchange": "HOSE",
    "last_price": 6.72,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 0.019999999999999574,
    "price_change_pct_1d": 0.29850746268655914,
    "total_value": 1323511000.0,
    "deal_volume_spike_20d_pct": 54.78522143024737
  },
  {
    "symbol": "SHI",
    "exchange": "HOSE",
    "last_price": 14.85,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 0.8499999999999996,
    "price_change_pct_1d": 6.071428571428572,
    "total_value": 8993785000.0,
    "deal_volume_spike_20d_pct": 47.23759718374761
  }
]
```
