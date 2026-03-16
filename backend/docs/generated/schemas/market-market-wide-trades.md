# market.market_wide.trades

- Class: `_unknown_`
- Method: `trades`
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

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

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

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
