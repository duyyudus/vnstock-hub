# market.market_wide.ohlcv

- Class: `_unknown_`
- Method: `ohlcv`
- Signature: `()`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `kbs, msn`

## Parameters

_None._

## Source details

### Source `kbs`

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
