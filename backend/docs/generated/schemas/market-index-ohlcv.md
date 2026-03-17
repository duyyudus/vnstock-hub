# market.index.ohlcv

- Class: `IndexMarket`
- Method: `ohlcv`
- Signature: `(start: str = None, end: str = None, interval: str = None, count_back: int = None) -> pd.DataFrame`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `kbs, msn`
- Declared signature: `(**B)`
- Default route source: `kbs`
- Default provider: `quote.Quote.history`

Historical OHLCV bars.

## Purpose

Historical OHLCV bars.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `start` | `DOC` | `False` | `None` | `str` | `2025-03-01` |  | Start date (YYYY-MM-DD). Optional if count_back is provided. |
| `end` | `DOC` | `False` | `None` | `str` | `2025-03-07` |  | End date (YYYY-MM-DD). Default is today. |
| `interval` | `DOC` | `False` | `None` | `str` | `1D` | `1D`, `1W`, `1M`, `1m`, `5m`, `15m`, `1H` | Timeframe interval ('1D', '1W', '1M', '1m', '5m', '15m', '1H'). |
| `count_back` | `DOC` | `False` | `None` | `int` | `5` |  | Number of bars to fetch backward from end date. |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.quote.Quote`
- Provider method: `history`

```text
time, open, high, low, close, volume, value, ticker, symbol
```

| Raw | Normalized |
| --- | --- |
| `time` | `time` |
| `open` | `open` |
| `high` | `high` |
| `low` | `low` |
| `close` | `close` |
| `volume` | `volume` |
| `value` | `value` |
| `ticker` | `ticker` |
| `symbol` | `ticker` |

#### Normalized output schema

- Coverage: `declared`

```text
time, open, high, low, close, volume, value, ticker
```

#### Live-observed sample

- Captured at: `2026-03-17T05:27:07.760286+00:00`
- Success: `True`
- Row count: `5`

```text
time, open, high, low, close, volume
```
- Dtypes: `{'time': 'datetime64[ns]', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'int64'}`

```json
[
  {
    "time": "2025-03-03T07:00:00",
    "open": 1.31,
    "high": 1.31,
    "low": 1.3,
    "close": 1.31,
    "volume": 801877900
  },
  {
    "time": "2025-03-04T07:00:00",
    "open": 1.31,
    "high": 1.31,
    "low": 1.3,
    "close": 1.31,
    "volume": 935915000
  },
  {
    "time": "2025-03-05T07:00:00",
    "open": 1.31,
    "high": 1.32,
    "low": 1.3,
    "close": 1.3,
    "volume": 770489000
  }
]
```

### Source `msn`

#### Raw output contract

- Coverage: `declared`

```text
time, open, high, low, close, volume, value, ticker, symbol
```

| Raw | Normalized |
| --- | --- |
| `time` | `time` |
| `open` | `open` |
| `high` | `high` |
| `low` | `low` |
| `close` | `close` |
| `volume` | `volume` |
| `value` | `value` |
| `ticker` | `ticker` |
| `symbol` | `ticker` |

#### Normalized output schema

- Coverage: `declared`

```text
time, open, high, low, close, volume, value, ticker
```

#### Live-observed sample

- Captured at: `2026-03-17T05:27:07.905828+00:00`
- Success: `True`
- Row count: `5`

```text
time, open, high, low, close, volume
```
- Dtypes: `{'time': 'datetime64[ns]', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'int64'}`

```json
[
  {
    "time": "2025-03-03T07:00:00",
    "open": 1.31,
    "high": 1.31,
    "low": 1.3,
    "close": 1.31,
    "volume": 801877900
  },
  {
    "time": "2025-03-04T07:00:00",
    "open": 1.31,
    "high": 1.31,
    "low": 1.3,
    "close": 1.31,
    "volume": 935915000
  },
  {
    "time": "2025-03-05T07:00:00",
    "open": 1.31,
    "high": 1.32,
    "low": 1.3,
    "close": 1.3,
    "volume": 770489000
  }
]
```
