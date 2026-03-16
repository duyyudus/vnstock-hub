# market.warrant.ohlcv

- Class: `WarrantMarket`
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

| Name | Kind | Required | Default | Annotation | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `start` | `DOC` | `False` | `None` | `str` |  | Start date (YYYY-MM-DD). Optional if count_back is provided. |
| `end` | `DOC` | `False` | `None` | `str` |  | End date (YYYY-MM-DD). Default is today. |
| `interval` | `DOC` | `False` | `None` | `str` | `1D`, `1W`, `1M`, `1m`, `5m`, `15m`, `1H` | Timeframe interval ('1D', '1W', '1M', '1m', '5m', '15m', '1H'). |
| `count_back` | `DOC` | `False` | `None` | `int` |  | Number of bars to fetch backward from end date. |

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

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

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

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
