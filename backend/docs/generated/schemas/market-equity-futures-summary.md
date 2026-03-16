# market.equity.futures_summary

- Class: `_unknown_`
- Method: `futures_summary`
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
symbol, exchange, reference_price, ceiling_price, floor_price, open_interest, basis, first_trading_date, last_trading_date, underlying_symbol, foreign_buy_volume, foreign_sell_volume
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `exchange` | `exchange` |
| `reference_price` | `reference_price` |
| `ceiling_price` | `ceiling_price` |
| `floor_price` | `floor_price` |
| `open_interest` | `open_interest` |
| `basis` | `basis` |
| `first_trading_date` | `first_trading_date` |
| `last_trading_date` | `last_trading_date` |
| `underlying_symbol` | `underlying_symbol` |
| `foreign_buy_volume` | `foreign_buy_volume` |
| `foreign_sell_volume` | `foreign_sell_volume` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, exchange, reference_price, ceiling_price, floor_price, open_interest, basis, first_trading_date, last_trading_date, underlying_symbol, foreign_buy_volume, foreign_sell_volume
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
symbol, exchange, reference_price, ceiling_price, floor_price, open_interest, basis, first_trading_date, last_trading_date, underlying_symbol, foreign_buy_volume, foreign_sell_volume
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `exchange` | `exchange` |
| `reference_price` | `reference_price` |
| `ceiling_price` | `ceiling_price` |
| `floor_price` | `floor_price` |
| `open_interest` | `open_interest` |
| `basis` | `basis` |
| `first_trading_date` | `first_trading_date` |
| `last_trading_date` | `last_trading_date` |
| `underlying_symbol` | `underlying_symbol` |
| `foreign_buy_volume` | `foreign_buy_volume` |
| `foreign_sell_volume` | `foreign_sell_volume` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, exchange, reference_price, ceiling_price, floor_price, open_interest, basis, first_trading_date, last_trading_date, underlying_symbol, foreign_buy_volume, foreign_sell_volume
```

Enum/value normalization:

- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
