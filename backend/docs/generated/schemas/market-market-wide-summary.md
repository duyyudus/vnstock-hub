# market.market_wide.summary

- Class: `_unknown_`
- Method: `summary`
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
MAP52, MIP52, DIV, BT, EPS, BVPS, MC, PER, PBR, ROE, CMCM, CMCY, YD, FTO
```

| Raw | Normalized |
| --- | --- |
| `MAP52` | `high_52w` |
| `MIP52` | `low_52w` |
| `DIV` | `dividend` |
| `BT` | `beta` |
| `EPS` | `eps` |
| `BVPS` | `bvps` |
| `MC` | `market_cap` |
| `PER` | `pe` |
| `PBR` | `pb` |
| `ROE` | `roe` |
| `CMCM` | `change_1m` |
| `CMCY` | `change_1y` |
| `YD` | `dividend_yield` |
| `FTO` | `foreign_ownership_pct` |

#### Normalized output schema

- Coverage: `declared`

```text
high_52w, low_52w, dividend, beta, eps, bvps, market_cap, pe, pb, roe, change_1m, change_1y, dividend_yield, foreign_ownership_pct
```

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

### Source `msn`

#### Raw output contract

- Coverage: `declared`

```text
MAP52, MIP52, DIV, BT, EPS, BVPS, MC, PER, PBR, ROE, CMCM, CMCY, YD, FTO
```

| Raw | Normalized |
| --- | --- |
| `MAP52` | `high_52w` |
| `MIP52` | `low_52w` |
| `DIV` | `dividend` |
| `BT` | `beta` |
| `EPS` | `eps` |
| `BVPS` | `bvps` |
| `MC` | `market_cap` |
| `PER` | `pe` |
| `PBR` | `pb` |
| `ROE` | `roe` |
| `CMCM` | `change_1m` |
| `CMCY` | `change_1y` |
| `YD` | `dividend_yield` |
| `FTO` | `foreign_ownership_pct` |

#### Normalized output schema

- Coverage: `declared`

```text
high_52w, low_52w, dividend, beta, eps, bvps, market_cap, pe, pb, roe, change_1m, change_1y, dividend_yield, foreign_ownership_pct
```

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
