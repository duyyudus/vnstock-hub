# market.market_wide.block_trades

- Class: `_unknown_`
- Method: `block_trades`
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
symbol, time, exchange, match_price, match_volume, trading_date, reference_price, floor_price
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `time` | `time` |
| `exchange` | `exchange` |
| `match_price` | `match_price` |
| `match_volume` | `match_volume` |
| `trading_date` | `trading_date` |
| `reference_price` | `reference_price` |
| `floor_price` | `floor_price` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, time, exchange, match_price, match_volume, trading_date, reference_price, floor_price
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
symbol, time, exchange, match_price, match_volume, trading_date, reference_price, floor_price
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `time` | `time` |
| `exchange` | `exchange` |
| `match_price` | `match_price` |
| `match_volume` | `match_volume` |
| `trading_date` | `trading_date` |
| `reference_price` | `reference_price` |
| `floor_price` | `floor_price` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, time, exchange, match_price, match_volume, trading_date, reference_price, floor_price
```

Enum/value normalization:

- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
