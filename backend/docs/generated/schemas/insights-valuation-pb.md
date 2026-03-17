# insights.valuation.pb

- Class: `MarketValuation`
- Method: `pb`
- Signature: `(duration = '5Y')`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `vnd`
- Declared signature: `(duration='5Y', **B)`
- Default route source: `vnd`
- Default provider: `market.Market.pb`

Retrieves P/B (Price-to-Book) ratio data.

## Purpose

Retrieves P/B (Price-to-Book) ratio data.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `duration` | `POSITIONAL_OR_KEYWORD` | `False` | `5Y` | `` | `1Y` |

## Source details

### Source `vnd`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vnd.market.Market`
- Provider method: `pb`

```text
pb
```

| Raw | Normalized |
| --- | --- |
| `pb` | `pb` |

#### Normalized output schema

- Coverage: `declared`

```text
pb
```

#### Live-observed sample

- Captured at: `2026-03-17T05:26:53.460181+00:00`
- Success: `True`
- Row count: `248`

```text
pb
```
- Dtypes: `{'pb': 'float64'}`

```json
[
  {
    "pb": 1.7153317177819771
  },
  {
    "pb": 1.7084414301346786
  },
  {
    "pb": 1.7003046099204708
  }
]
```
