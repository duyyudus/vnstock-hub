# market.futures.order_book

- Class: `FuturesMarket`
- Method: `order_book`
- Signature: `(symbols_list: List[str], board: str = 'stock', exchange: str = 'HOSE', show_log: bool = False, get_all: bool = False) -> pd.DataFrame`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `kbs, msn`
- Declared signature: `(**A)`
- Default route source: `kbs`
- Default provider: `trading.Trading.price_board`

Order book levels (Best Bid/Ask L2/L3).

## Purpose

Order book levels (Best Bid/Ask L2/L3).

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
| `bid_price_1` | `bid_price_1` |
| `bid_vol_1` | `bid_vol_1` |
| `ask_price_1` | `ask_price_1` |
| `ask_vol_1` | `ask_vol_1` |
| `bid_price_2` | `bid_price_2` |
| `bid_vol_2` | `bid_vol_2` |
| `bid_price_3` | `bid_price_3` |
| `bid_vol_3` | `bid_vol_3` |
| `bid_price_4` | `bid_price_4` |
| `bid_vol_4` | `bid_vol_4` |
| `bid_price_5` | `bid_price_5` |
| `bid_vol_5` | `bid_vol_5` |
| `bid_price_6` | `bid_price_6` |
| `bid_vol_6` | `bid_vol_6` |
| `bid_price_7` | `bid_price_7` |
| `bid_vol_7` | `bid_vol_7` |
| `bid_price_8` | `bid_price_8` |
| `bid_vol_8` | `bid_vol_8` |
| `bid_price_9` | `bid_price_9` |
| `bid_vol_9` | `bid_vol_9` |
| `bid_price_10` | `bid_price_10` |
| `bid_vol_10` | `bid_vol_10` |
| `ask_price_2` | `ask_price_2` |
| `ask_vol_2` | `ask_vol_2` |
| `ask_price_3` | `ask_price_3` |
| `ask_vol_3` | `ask_vol_3` |
| `ask_price_4` | `ask_price_4` |
| `ask_vol_4` | `ask_vol_4` |
| `ask_price_5` | `ask_price_5` |
| `ask_vol_5` | `ask_vol_5` |
| `ask_price_6` | `ask_price_6` |
| `ask_vol_6` | `ask_vol_6` |
| `ask_price_7` | `ask_price_7` |
| `ask_vol_7` | `ask_vol_7` |
| `ask_price_8` | `ask_price_8` |
| `ask_vol_8` | `ask_vol_8` |
| `ask_price_9` | `ask_price_9` |
| `ask_vol_9` | `ask_vol_9` |
| `ask_price_10` | `ask_price_10` |
| `ask_vol_10` | `ask_vol_10` |
- Note: Derived from `app.lib.vnstock_data_alt.explorer.kbs.trading._PRICE_BOARD_STANDARD_COLUMNS`.

#### Normalized output schema

- Coverage: `declared`

```text
bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, bid_price_4, bid_vol_4, bid_price_5, bid_vol_5, bid_price_6, bid_vol_6, bid_price_7, bid_vol_7, bid_price_8, bid_vol_8, bid_price_9, bid_vol_9, bid_price_10, bid_vol_10, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, ask_price_4, ask_vol_4, ask_price_5, ask_vol_5, ask_price_6, ask_vol_6, ask_price_7, ask_vol_7, ask_price_8, ask_vol_8, ask_price_9, ask_vol_9, ask_price_10, ask_vol_10
```

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

### Source `msn`

#### Raw output contract

- Coverage: `declared`

```text
bid_price_1, bid_vol_1, ask_price_1, ask_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, bid_price_4, bid_vol_4, bid_price_5, bid_vol_5, bid_price_6, bid_vol_6, bid_price_7, bid_vol_7, bid_price_8, bid_vol_8, bid_price_9, bid_vol_9, bid_price_10, bid_vol_10, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, ask_price_4, ask_vol_4, ask_price_5, ask_vol_5, ask_price_6, ask_vol_6, ask_price_7, ask_vol_7, ask_price_8, ask_vol_8, ask_price_9, ask_vol_9, ask_price_10, ask_vol_10
```

| Raw | Normalized |
| --- | --- |
| `bid_price_1` | `bid_price_1` |
| `bid_vol_1` | `bid_vol_1` |
| `ask_price_1` | `ask_price_1` |
| `ask_vol_1` | `ask_vol_1` |
| `bid_price_2` | `bid_price_2` |
| `bid_vol_2` | `bid_vol_2` |
| `bid_price_3` | `bid_price_3` |
| `bid_vol_3` | `bid_vol_3` |
| `bid_price_4` | `bid_price_4` |
| `bid_vol_4` | `bid_vol_4` |
| `bid_price_5` | `bid_price_5` |
| `bid_vol_5` | `bid_vol_5` |
| `bid_price_6` | `bid_price_6` |
| `bid_vol_6` | `bid_vol_6` |
| `bid_price_7` | `bid_price_7` |
| `bid_vol_7` | `bid_vol_7` |
| `bid_price_8` | `bid_price_8` |
| `bid_vol_8` | `bid_vol_8` |
| `bid_price_9` | `bid_price_9` |
| `bid_vol_9` | `bid_vol_9` |
| `bid_price_10` | `bid_price_10` |
| `bid_vol_10` | `bid_vol_10` |
| `ask_price_2` | `ask_price_2` |
| `ask_vol_2` | `ask_vol_2` |
| `ask_price_3` | `ask_price_3` |
| `ask_vol_3` | `ask_vol_3` |
| `ask_price_4` | `ask_price_4` |
| `ask_vol_4` | `ask_vol_4` |
| `ask_price_5` | `ask_price_5` |
| `ask_vol_5` | `ask_vol_5` |
| `ask_price_6` | `ask_price_6` |
| `ask_vol_6` | `ask_vol_6` |
| `ask_price_7` | `ask_price_7` |
| `ask_vol_7` | `ask_vol_7` |
| `ask_price_8` | `ask_price_8` |
| `ask_vol_8` | `ask_vol_8` |
| `ask_price_9` | `ask_price_9` |
| `ask_vol_9` | `ask_vol_9` |
| `ask_price_10` | `ask_price_10` |
| `ask_vol_10` | `ask_vol_10` |

#### Normalized output schema

- Coverage: `declared`

```text
bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, bid_price_4, bid_vol_4, bid_price_5, bid_vol_5, bid_price_6, bid_vol_6, bid_price_7, bid_vol_7, bid_price_8, bid_vol_8, bid_price_9, bid_vol_9, bid_price_10, bid_vol_10, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, ask_price_4, ask_vol_4, ask_price_5, ask_vol_5, ask_price_6, ask_vol_6, ask_price_7, ask_vol_7, ask_price_8, ask_vol_8, ask_price_9, ask_vol_9, ask_price_10, ask_vol_10
```

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
