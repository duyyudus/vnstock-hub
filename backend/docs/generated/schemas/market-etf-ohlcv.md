# market.etf.ohlcv

- Class: `ETFMarket`
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

- Captured at: `2026-03-16T11:15:15.682063+00:00`
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
    "open": 23.7,
    "high": 23.78,
    "low": 23.67,
    "close": 23.76,
    "volume": 358300
  },
  {
    "time": "2025-03-04T07:00:00",
    "open": 23.76,
    "high": 23.84,
    "low": 23.61,
    "close": 23.83,
    "volume": 713600
  },
  {
    "time": "2025-03-05T07:00:00",
    "open": 23.83,
    "high": 23.99,
    "low": 23.83,
    "close": 23.88,
    "volume": 1507000
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

- Captured at: `2026-03-16T11:15:15.839609+00:00`
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
    "open": 23.7,
    "high": 23.78,
    "low": 23.67,
    "close": 23.76,
    "volume": 358300
  },
  {
    "time": "2025-03-04T07:00:00",
    "open": 23.76,
    "high": 23.84,
    "low": 23.61,
    "close": 23.83,
    "volume": 713600
  },
  {
    "time": "2025-03-05T07:00:00",
    "open": 23.83,
    "high": 23.99,
    "low": 23.83,
    "close": 23.88,
    "volume": 1507000
  }
]
```
