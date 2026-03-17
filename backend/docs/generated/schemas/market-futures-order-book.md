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

| Name | Kind | Required | Default | Annotation | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `symbols_list` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `List[str]` | `['VCB', 'TCB']` | `ACB`, `VNM` | List of symbols (e.g., ['ACB', 'VNM']). |
| `board` | `POSITIONAL_OR_KEYWORD` | `False` | `stock` | `str` | `omitted; default 'stock'` | `stock`, `odd_lot`, `put_through`, `derivatives` | Board type ('stock', 'odd_lot', 'put_through', 'derivatives'). |
| `exchange` | `POSITIONAL_OR_KEYWORD` | `False` | `HOSE` | `str` | `HOSE` | `HOSE`, `HNX`, `UPCOM` | Exchange ('HOSE', 'HNX', 'UPCOM'). |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `bool` | `False` |  | Display debug logs. |
| `get_all` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `bool` | `True` |  | If True, return all raw columns. Otherwise, standard columns. |

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

- Captured at: `2026-03-17T05:27:06.389222+00:00`
- Success: `True`
- Row count: `2`

```text
bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3
```
- Dtypes: `{'bid_price_1': 'str', 'bid_vol_1': 'int64', 'bid_price_2': 'int64', 'bid_vol_2': 'int64', 'bid_price_3': 'int64', 'bid_vol_3': 'int64', 'ask_price_1': 'str', 'ask_vol_1': 'int64', 'ask_price_2': 'int64', 'ask_vol_2': 'int64', 'ask_price_3': 'int64', 'ask_vol_3': 'int64'}`

```json
[
  {
    "bid_price_1": "59900.0",
    "bid_vol_1": 33900,
    "bid_price_2": 59800,
    "bid_vol_2": 111300,
    "bid_price_3": 59700,
    "bid_vol_3": 209300,
    "ask_price_1": "60000.0",
    "ask_vol_1": 269400,
    "ask_price_2": 60100,
    "ask_vol_2": 279400,
    "ask_price_3": 60200,
    "ask_vol_3": 167400
  },
  {
    "bid_price_1": "30650.0",
    "bid_vol_1": 19400,
    "bid_price_2": 30600,
    "bid_vol_2": 45300,
    "bid_price_3": 30550,
    "bid_vol_3": 68900,
    "ask_price_1": "30700.0",
    "ask_vol_1": 4900,
    "ask_price_2": 30750,
    "ask_vol_2": 25000,
    "ask_price_3": 30800,
    "ask_vol_3": 297500
  }
]
```

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

- Captured at: `2026-03-17T05:27:06.618554+00:00`
- Success: `True`
- Row count: `2`

```text
bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3
```
- Dtypes: `{'bid_price_1': 'str', 'bid_vol_1': 'int64', 'bid_price_2': 'int64', 'bid_vol_2': 'int64', 'bid_price_3': 'int64', 'bid_vol_3': 'int64', 'ask_price_1': 'str', 'ask_vol_1': 'int64', 'ask_price_2': 'int64', 'ask_vol_2': 'int64', 'ask_price_3': 'int64', 'ask_vol_3': 'int64'}`

```json
[
  {
    "bid_price_1": "59900.0",
    "bid_vol_1": 33900,
    "bid_price_2": 59800,
    "bid_vol_2": 111300,
    "bid_price_3": 59700,
    "bid_vol_3": 209300,
    "ask_price_1": "60000.0",
    "ask_vol_1": 269400,
    "ask_price_2": 60100,
    "ask_vol_2": 279400,
    "ask_price_3": 60200,
    "ask_vol_3": 167400
  },
  {
    "bid_price_1": "30650.0",
    "bid_vol_1": 19400,
    "bid_price_2": 30600,
    "bid_vol_2": 45300,
    "bid_price_3": 30550,
    "bid_vol_3": 68900,
    "ask_price_1": "30700.0",
    "ask_vol_1": 4900,
    "ask_price_2": 30750,
    "ask_vol_2": 25000,
    "ask_price_3": 30800,
    "ask_vol_3": 297500
  }
]
```
