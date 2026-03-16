# market.crypto.warrant_summary

- Class: `_unknown_`
- Method: `warrant_summary`
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
symbol, SB, exchange, EX, reference_price, RE, ceiling_price, CL, floor_price, FL, open_price, OP, high_price, HI, low_price, LO, close_price, CP, break_even_point, intrinsic_value, CPR
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `SB` | `symbol` |
| `exchange` | `exchange` |
| `EX` | `exchange` |
| `reference_price` | `reference_price` |
| `RE` | `reference_price` |
| `ceiling_price` | `ceiling_price` |
| `CL` | `ceiling_price` |
| `floor_price` | `floor_price` |
| `FL` | `floor_price` |
| `open_price` | `open_price` |
| `OP` | `open_price` |
| `high_price` | `high_price` |
| `HI` | `high_price` |
| `low_price` | `low_price` |
| `LO` | `low_price` |
| `close_price` | `close_price` |
| `CP` | `close_price` |
| `break_even_point` | `break_even_point` |
| `intrinsic_value` | `intrinsic_value` |
| `CPR` | `underlying_price` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, exchange, reference_price, ceiling_price, floor_price, open_price, high_price, low_price, close_price, break_even_point, intrinsic_value, underlying_price
```

Enum/value normalization:

- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

### Source `msn`

#### Raw output contract

- Coverage: `declared`

```text
symbol, SB, exchange, EX, reference_price, RE, ceiling_price, CL, floor_price, FL, open_price, OP, high_price, HI, low_price, LO, close_price, CP, break_even_point, intrinsic_value, CPR
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `SB` | `symbol` |
| `exchange` | `exchange` |
| `EX` | `exchange` |
| `reference_price` | `reference_price` |
| `RE` | `reference_price` |
| `ceiling_price` | `ceiling_price` |
| `CL` | `ceiling_price` |
| `floor_price` | `floor_price` |
| `FL` | `floor_price` |
| `open_price` | `open_price` |
| `OP` | `open_price` |
| `high_price` | `high_price` |
| `HI` | `high_price` |
| `low_price` | `low_price` |
| `LO` | `low_price` |
| `close_price` | `close_price` |
| `CP` | `close_price` |
| `break_even_point` | `break_even_point` |
| `intrinsic_value` | `intrinsic_value` |
| `CPR` | `underlying_price` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, exchange, reference_price, ceiling_price, floor_price, open_price, high_price, low_price, close_price, break_even_point, intrinsic_value, underlying_price
```

Enum/value normalization:

- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
