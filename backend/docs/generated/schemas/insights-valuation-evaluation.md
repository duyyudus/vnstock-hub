# insights.valuation.evaluation

- Class: `MarketValuation`
- Method: `evaluation`
- Signature: `(duration = '5Y')`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `vnd`
- Declared signature: `(duration='5Y', **B)`
- Default route source: `vnd`
- Default provider: `market.Market.evaluation`

Retrieves an overview of the market with both P/E and P/B ratios.

## Purpose

Retrieves an overview of the market with both P/E and P/B ratios.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `duration` | `POSITIONAL_OR_KEYWORD` | `False` | `5Y` | `` | `1Y` |

## Source details

### Source `vnd`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vnd.market.Market`
- Provider method: `evaluation`

```text
pe, pb
```

| Raw | Normalized |
| --- | --- |
| `pe` | `pe` |
| `pb` | `pb` |

#### Normalized output schema

- Coverage: `declared`

```text
pe, pb
```

#### Live-observed sample

- Captured at: `2026-03-17T05:26:53.386720+00:00`
- Success: `True`
- Row count: `248`

```text
pe, pb
```
- Dtypes: `{'pe': 'float64', 'pb': 'float64'}`

```json
[
  {
    "pe": 13.263777753970954,
    "pb": 1.7153317177819771
  },
  {
    "pe": 13.210795996016383,
    "pb": 1.7084414301346786
  },
  {
    "pe": 13.147344392107996,
    "pb": 1.7003046099204708
  }
]
```
