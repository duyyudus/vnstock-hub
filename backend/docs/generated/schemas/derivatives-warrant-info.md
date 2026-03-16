# derivatives.warrant.info

- Class: `WarrantReference`
- Method: `info`
- Signature: `(symbol: str = None) -> pd.DataFrame`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `kbs`
- Declared signature: `()`
- Default route source: `kbs`
- Default provider: `derivatives.KBSDerivatives.warrant_profile`

Get info and realtime information for the specific covered warrant.

## Purpose

Get info and realtime information for the specific covered warrant.

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
SB, IN, ULS, EP, ER, LTD, EX, LS, CWT, CPR, RE, CL, FL, CP, TV, FR, FB, FS, break_even_point, break_even_point_diff, intrinsic_value
```

| Raw | Normalized |
| --- | --- |
| `SB` | `symbol` |
| `IN` | `issuer` |
| `ULS` | `underlying_symbol` |
| `EP` | `exercise_price` |
| `ER` | `exercise_ratio` |
| `LTD` | `last_trading_date` |
| `EX` | `exchange` |
| `LS` | `listed_share` |
| `CWT` | `warrant_type` |
| `CPR` | `underlying_price` |
| `RE` | `reference_price` |
| `CL` | `ceiling_price` |
| `FL` | `floor_price` |
| `CP` | `match_price` |
| `TV` | `total_volume` |
| `FR` | `foreign_room` |
| `FB` | `foreign_buy_volume` |
| `FS` | `foreign_sell_volume` |
| `break_even_point` | `break_even_point` |
| `break_even_point_diff` | `break_even_point_diff` |
| `intrinsic_value` | `intrinsic_value` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, name, isin, issuer, underlying_symbol, exercise_price, exercise_ratio, issue_date, first_trading_date, last_trading_date, maturity_date, listed_share, exchange, warrant_type, underlying_price, reference_price, ceiling_price, floor_price, match_price, total_volume, foreign_room, foreign_buy_volume, foreign_sell_volume, break_even_point, break_even_point_diff, intrinsic_value
```

Enum/value normalization:

- `warrant_type`: {'CWT': 'Call', 'C': 'Call', 'PWT': 'Put', 'P': 'Put'}
- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
