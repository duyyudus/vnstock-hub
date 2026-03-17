# insights.valuation.pe

- Class: `MarketValuation`
- Method: `pe`
- Signature: `(duration = '5Y')`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `vnd`
- Declared signature: `(duration='5Y', **B)`
- Default route source: `vnd`
- Default provider: `market.Market.pe`

Retrieves P/E (Price-to-Earnings) ratio data.

## Purpose

Retrieves P/E (Price-to-Earnings) ratio data.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `duration` | `POSITIONAL_OR_KEYWORD` | `False` | `5Y` | `` | `1Y` |

## Source details

### Source `vnd`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vnd.market.Market`
- Provider method: `pe`

```text
pe
```

| Raw | Normalized |
| --- | --- |
| `pe` | `pe` |

#### Normalized output schema

- Coverage: `declared`

```text
pe
```

#### Live-observed sample

- Captured at: `2026-03-17T05:26:53.641549+00:00`
- Success: `True`
- Row count: `248`

```text
pe
```
- Dtypes: `{'pe': 'float64'}`

```json
[
  {
    "pe": 13.263777753970954
  },
  {
    "pe": 13.210795996016383
  },
  {
    "pe": 13.147344392107996
  }
]
```
