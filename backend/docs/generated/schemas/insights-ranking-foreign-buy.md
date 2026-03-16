# insights.ranking.foreign_buy

- Class: `RankingReference`
- Method: `foreign_buy`
- Signature: `(date = None, limit = 10)`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `vnd`
- Declared signature: `(date=None, limit=10, **B)`
- Default route source: `vnd`
- Default provider: `insight.TopStock.foreign_buy`

Top 10 stocks with highest foreign net buy value.

## Purpose

Top 10 stocks with highest foreign net buy value.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `date` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `10` | `` | `5` |

## Source details

### Source `vnd`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vnd.insight.TopStock`
- Provider method: `foreign_buy`

```text
symbol, date, net_value
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `date` | `date` |
| `net_value` | `net_value` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, date, net_value
```

#### Live-observed sample

- Captured at: `2026-03-16T11:15:04.898231+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, date, net_value
```
- Dtypes: `{'symbol': 'str', 'date': 'str', 'net_value': 'float64'}`

```json
[
  {
    "symbol": "MCH",
    "date": "2026-03-16",
    "net_value": 114295100000.0
  },
  {
    "symbol": "PVS",
    "date": "2026-03-16",
    "net_value": 81343240000.0
  },
  {
    "symbol": "VCK",
    "date": "2026-03-16",
    "net_value": 55230680000.0
  }
]
```
