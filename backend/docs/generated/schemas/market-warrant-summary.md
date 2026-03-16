# market.warrant.summary

- Class: `WarrantMarket`
- Method: `summary`
- Signature: `(symbol: str = None) -> pd.DataFrame`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `kbs`
- Declared signature: `(**B)`
- Default route source: `kbs`
- Default provider: `derivatives.KBSDerivatives.warrant_profile`

Stock Info / Snapshot summary metrics including pricing,

## Purpose

Stock Info / Snapshot summary metrics including pricing, 
52-week ranges, and fundamental ratios.

## Parameters

| Name | Kind | Required | Default | Annotation | Description |
| --- | --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `str` | Optional warrant symbol. Defaults to instance symbol. |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.derivatives.KBSDerivatives`
- Provider method: `warrant_profile`

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
