# market.futures.quote

- Class: `FuturesMarket`
- Method: `quote`
- Signature: `(symbols_list: List[str], board: str = 'stock', exchange: str = 'HOSE', show_log: bool = False, get_all: bool = False) -> pd.DataFrame`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `kbs, msn`
- Declared signature: `(**A)`
- Default route source: `kbs`
- Default provider: `trading.Trading.price_board`

Real-time single-symbol pricing snapshot.

## Purpose

Real-time single-symbol pricing snapshot.
If the underlying provider expects `symbols_list`, it injects `[self.symbol]`.

## Parameters

| Name | Kind | Required | Default | Annotation | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `symbols_list` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `List[str]` | `ACB`, `VNM` | List of symbols (e.g., ['ACB', 'VNM']). |
| `board` | `POSITIONAL_OR_KEYWORD` | `False` | `stock` | `str` | `stock`, `odd_lot`, `put_through`, `derivatives` | Board type ('stock', 'odd_lot', 'put_through', 'derivatives'). |
| `exchange` | `POSITIONAL_OR_KEYWORD` | `False` | `HOSE` | `str` | `HOSE`, `HNX`, `UPCOM` | Exchange ('HOSE', 'HNX', 'UPCOM'). |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `bool` |  | Display debug logs. |
| `get_all` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `bool` |  | If True, return all raw columns. Otherwise, standard columns. |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.trading.Trading`
- Provider method: `price_board`

```text
symbol, time, exchange, ceiling_price, floor_price, reference_price, open_price, high_price, low_price, close_price, average_price, volume_accumulated, total_value, price_change, percent_change, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, foreign_buy_volume, foreign_sell_volume, foreign_room
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
- Note: Derived from `app.lib.vnstock_data_alt.explorer.kbs.trading._PRICE_BOARD_STANDARD_COLUMNS`.

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
