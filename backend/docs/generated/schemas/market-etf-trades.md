# market.etf.trades

- Class: `ETFMarket`
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

- Captured at: `2026-03-17T05:27:04.639417+00:00`
- Success: `True`
- Row count: `5`

```text
time, price, volume, match_type, id
```
- Dtypes: `{'time': 'datetime64[us]', 'price': 'float64', 'volume': 'int64', 'match_type': 'object', 'id': 'str'}`

```json
[
  {
    "time": "2026-03-17T11:28:01",
    "price": 33.5,
    "volume": 800,
    "match_type": "Buy",
    "id": "2026-03-17_112801_335000_800"
  },
  {
    "time": "2026-03-17T11:28:01",
    "price": 33.49,
    "volume": 200,
    "match_type": "Buy",
    "id": "2026-03-17_112801_334900_200"
  },
  {
    "time": "2026-03-17T11:27:33",
    "price": 33.45,
    "volume": 200,
    "match_type": "Sell",
    "id": "2026-03-17_112733_334500_200"
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

- Captured at: `2026-03-17T05:27:04.795124+00:00`
- Success: `True`
- Row count: `5`

```text
time, price, volume, match_type, id
```
- Dtypes: `{'time': 'datetime64[us]', 'price': 'float64', 'volume': 'int64', 'match_type': 'object', 'id': 'str'}`

```json
[
  {
    "time": "2026-03-17T11:28:01",
    "price": 33.49,
    "volume": 200,
    "match_type": "Buy",
    "id": "2026-03-17_112801_334900_200"
  },
  {
    "time": "2026-03-17T11:28:01",
    "price": 33.5,
    "volume": 800,
    "match_type": "Buy",
    "id": "2026-03-17_112801_335000_800"
  },
  {
    "time": "2026-03-17T11:27:33",
    "price": 33.45,
    "volume": 200,
    "match_type": "Sell",
    "id": "2026-03-17_112733_334500_200"
  }
]
```
