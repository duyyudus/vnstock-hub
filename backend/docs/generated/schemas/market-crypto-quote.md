# market.crypto.quote

- Class: `CryptoMarket`
- Method: `quote`
- Signature: `(**A) -> pd.DataFrame`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `kbs, msn`
- Declared signature: `(**A)`

Real-time single-symbol pricing snapshot.

## Purpose

Real-time single-symbol pricing snapshot.
If the underlying provider expects `symbols_list`, it injects `[self.symbol]`.

## Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `A` | `VAR_KEYWORD` | `True` | `None` | `` |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `declared`

```text
symbol, exchange, reference_price, ceiling_price, floor_price, close_price, match_price, current_vol, total_trades, open_price, high_price, low_price, basis, open_interest, foreign_buy_volume, foreign_sell_volume, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `exchange` | `exchange` |
| `reference_price` | `reference_price` |
| `ceiling_price` | `ceiling_price` |
| `floor_price` | `floor_price` |
| `close_price` | `close_price` |
| `match_price` | `match_price` |
| `current_vol` | `match_vol` |
| `total_trades` | `total_volume` |
| `open_price` | `open_price` |
| `high_price` | `high_price` |
| `low_price` | `low_price` |
| `basis` | `basis` |
| `open_interest` | `open_interest` |
| `foreign_buy_volume` | `foreign_buy_volume` |
| `foreign_sell_volume` | `foreign_sell_volume` |
| `bid_price_1` | `bid_price_1` |
| `bid_vol_1` | `bid_vol_1` |
| `bid_price_2` | `bid_price_2` |
| `bid_vol_2` | `bid_vol_2` |
| `bid_price_3` | `bid_price_3` |
| `bid_vol_3` | `bid_vol_3` |
| `ask_price_1` | `ask_price_1` |
| `ask_vol_1` | `ask_vol_1` |
| `ask_price_2` | `ask_price_2` |
| `ask_vol_2` | `ask_vol_2` |
| `ask_price_3` | `ask_price_3` |
| `ask_vol_3` | `ask_vol_3` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, exchange, reference_price, ceiling_price, floor_price, open_price, high_price, low_price, close_price, match_price, match_vol, total_volume, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, basis, open_interest, foreign_buy_volume, foreign_sell_volume
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
symbol, exchange, reference_price, ceiling_price, floor_price, close_price, match_price, current_vol, total_trades, open_price, high_price, low_price, basis, open_interest, foreign_buy_volume, foreign_sell_volume, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `exchange` | `exchange` |
| `reference_price` | `reference_price` |
| `ceiling_price` | `ceiling_price` |
| `floor_price` | `floor_price` |
| `close_price` | `close_price` |
| `match_price` | `match_price` |
| `current_vol` | `match_vol` |
| `total_trades` | `total_volume` |
| `open_price` | `open_price` |
| `high_price` | `high_price` |
| `low_price` | `low_price` |
| `basis` | `basis` |
| `open_interest` | `open_interest` |
| `foreign_buy_volume` | `foreign_buy_volume` |
| `foreign_sell_volume` | `foreign_sell_volume` |
| `bid_price_1` | `bid_price_1` |
| `bid_vol_1` | `bid_vol_1` |
| `bid_price_2` | `bid_price_2` |
| `bid_vol_2` | `bid_vol_2` |
| `bid_price_3` | `bid_price_3` |
| `bid_vol_3` | `bid_vol_3` |
| `ask_price_1` | `ask_price_1` |
| `ask_vol_1` | `ask_vol_1` |
| `ask_price_2` | `ask_price_2` |
| `ask_vol_2` | `ask_vol_2` |
| `ask_price_3` | `ask_price_3` |
| `ask_vol_3` | `ask_vol_3` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, exchange, reference_price, ceiling_price, floor_price, open_price, high_price, low_price, close_price, match_price, match_vol, total_volume, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, basis, open_interest, foreign_buy_volume, foreign_sell_volume
```

Enum/value normalization:

- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
