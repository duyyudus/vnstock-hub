# market.crypto.volume_profile

- Class: `_unknown_`
- Method: `volume_profile`
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
price, buyVol, sellVol, unknownVol, totalVol, percent
```

| Raw | Normalized |
| --- | --- |
| `price` | `price` |
| `buyVol` | `buy_volume` |
| `sellVol` | `sell_volume` |
| `unknownVol` | `unknown_volume` |
| `totalVol` | `total_volume` |
| `percent` | `match_percent` |

#### Normalized output schema

- Coverage: `declared`

```text
price, buy_volume, sell_volume, unknown_volume, total_volume, match_percent
```

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

### Source `msn`

#### Raw output contract

- Coverage: `declared`

```text
price, buyVol, sellVol, unknownVol, totalVol, percent
```

| Raw | Normalized |
| --- | --- |
| `price` | `price` |
| `buyVol` | `buy_volume` |
| `sellVol` | `sell_volume` |
| `unknownVol` | `unknown_volume` |
| `totalVol` | `total_volume` |
| `percent` | `match_percent` |

#### Normalized output schema

- Coverage: `declared`

```text
price, buy_volume, sell_volume, unknown_volume, total_volume, match_percent
```

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
