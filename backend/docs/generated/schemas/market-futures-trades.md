# market.futures.trades

- Class: `FuturesMarket`
- Method: `trades`
- Signature: `(limit: int = 1000) -> pd.DataFrame`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `kbs, msn`
- Declared signature: `(limit=1000, **A)`
- Default route source: `kbs`
- Default provider: `quote.Quote.intraday`

Real-time or intraday tick-by-tick trading tape (Time & Sales).

## Purpose

Real-time or intraday tick-by-tick trading tape (Time & Sales).

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `1000` | `int` | `5` | Number of records to fetch (default: 1000). |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.quote.Quote`
- Provider method: `intraday`

```text
time, price, volume, side, match_type, id
```

| Raw | Normalized |
| --- | --- |
| `time` | `time` |
| `price` | `price` |
| `volume` | `volume` |
| `side` | `side` |
| `match_type` | `match_type` |
| `id` | `id` |

#### Normalized output schema

- Coverage: `declared`

```text
time, price, volume, side, match_type, id
```

Enum/value normalization:

- `match_type`: {'buy': 'Buy', 'sell': 'Sell', 'B': 'Buy', 'S': 'Sell', 'unknown': 'Unknown', 'U': 'Unknown', '': 'Unknown'}

#### Live-observed sample

- Captured at: `2026-03-16T11:15:20.299529+00:00`
- Success: `True`
- Row count: `5`

```text
time, price, volume, match_type, id
```
- Dtypes: `{'time': 'datetime64[us]', 'price': 'float64', 'volume': 'int64', 'match_type': 'object', 'id': 'str'}`

```json
[
  {
    "time": "2026-03-16T14:45:01",
    "price": 1.85,
    "volume": 1133,
    "match_type": "Unknown",
    "id": "2026-03-16_144501_18513_1133"
  },
  {
    "time": "2026-03-16T14:45:01",
    "price": 1.85,
    "volume": 1218,
    "match_type": "Unknown",
    "id": "2026-03-16_144501_18513_1218"
  },
  {
    "time": "2026-03-16T14:45:00",
    "price": 1.85,
    "volume": 1993,
    "match_type": "Unknown",
    "id": "2026-03-16_144500_18513_1993"
  }
]
```

### Source `msn`

#### Raw output contract

- Coverage: `declared`

```text
time, price, volume, side, match_type, id
```

| Raw | Normalized |
| --- | --- |
| `time` | `time` |
| `price` | `price` |
| `volume` | `volume` |
| `side` | `side` |
| `match_type` | `match_type` |
| `id` | `id` |

#### Normalized output schema

- Coverage: `declared`

```text
time, price, volume, side, match_type, id
```

Enum/value normalization:

- `match_type`: {'buy': 'Buy', 'sell': 'Sell', 'B': 'Buy', 'S': 'Sell', 'unknown': 'Unknown', 'U': 'Unknown', '': 'Unknown'}

#### Live-observed sample

- Captured at: `2026-03-16T11:15:20.449227+00:00`
- Success: `True`
- Row count: `5`

```text
time, price, volume, match_type, id
```
- Dtypes: `{'time': 'datetime64[us]', 'price': 'float64', 'volume': 'int64', 'match_type': 'object', 'id': 'str'}`

```json
[
  {
    "time": "2026-03-16T14:45:01",
    "price": 1.85,
    "volume": 1133,
    "match_type": "Unknown",
    "id": "2026-03-16_144501_18513_1133"
  },
  {
    "time": "2026-03-16T14:45:01",
    "price": 1.85,
    "volume": 1218,
    "match_type": "Unknown",
    "id": "2026-03-16_144501_18513_1218"
  },
  {
    "time": "2026-03-16T14:45:00",
    "price": 1.85,
    "volume": 1993,
    "match_type": "Unknown",
    "id": "2026-03-16_144500_18513_1993"
  }
]
```
