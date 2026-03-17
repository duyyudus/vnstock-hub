# market.etf.quote

- Class: `ETFMarket`
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

- Captured at: `2026-03-17T05:27:02.891974+00:00`
- Success: `True`
- Row count: `2`

```text
symbol, exchange, reference_price, ceiling_price, floor_price, open_price, high_price, low_price, close_price, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, foreign_buy_volume, foreign_sell_volume
```
- Dtypes: `{'symbol': 'str', 'exchange': 'str', 'reference_price': 'int64', 'ceiling_price': 'int64', 'floor_price': 'int64', 'open_price': 'int64', 'high_price': 'int64', 'low_price': 'int64', 'close_price': 'int64', 'bid_price_1': 'str', 'bid_vol_1': 'int64', 'bid_price_2': 'int64', 'bid_vol_2': 'int64', 'bid_price_3': 'int64', 'bid_vol_3': 'int64', 'ask_price_1': 'str', 'ask_vol_1': 'int64', 'ask_price_2': 'int64', 'ask_vol_2': 'int64', 'ask_price_3': 'int64', 'ask_vol_3': 'int64', 'foreign_buy_volume': 'int64', 'foreign_sell_volume': 'int64'}`

```json
[
  {
    "symbol": "VCB",
    "exchange": "HOSE",
    "reference_price": 58800,
    "ceiling_price": 62900,
    "floor_price": 54700,
    "open_price": 59500,
    "high_price": 60100,
    "low_price": 59400,
    "close_price": 60000,
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
    "ask_vol_3": 167400,
    "foreign_buy_volume": 671400,
    "foreign_sell_volume": 253800
  },
  {
    "symbol": "TCB",
    "exchange": "HOSE",
    "reference_price": 30200,
    "ceiling_price": 32300,
    "floor_price": 28100,
    "open_price": 30250,
    "high_price": 30900,
    "low_price": 30250,
    "close_price": 30700,
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
    "ask_vol_3": 297500,
    "foreign_buy_volume": 10100,
    "foreign_sell_volume": 0
  }
]
```

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

- Captured at: `2026-03-17T05:27:03.045708+00:00`
- Success: `True`
- Row count: `2`

```text
symbol, exchange, reference_price, ceiling_price, floor_price, open_price, high_price, low_price, close_price, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, foreign_buy_volume, foreign_sell_volume
```
- Dtypes: `{'symbol': 'str', 'exchange': 'str', 'reference_price': 'int64', 'ceiling_price': 'int64', 'floor_price': 'int64', 'open_price': 'int64', 'high_price': 'int64', 'low_price': 'int64', 'close_price': 'int64', 'bid_price_1': 'str', 'bid_vol_1': 'int64', 'bid_price_2': 'int64', 'bid_vol_2': 'int64', 'bid_price_3': 'int64', 'bid_vol_3': 'int64', 'ask_price_1': 'str', 'ask_vol_1': 'int64', 'ask_price_2': 'int64', 'ask_vol_2': 'int64', 'ask_price_3': 'int64', 'ask_vol_3': 'int64', 'foreign_buy_volume': 'int64', 'foreign_sell_volume': 'int64'}`

```json
[
  {
    "symbol": "VCB",
    "exchange": "HOSE",
    "reference_price": 58800,
    "ceiling_price": 62900,
    "floor_price": 54700,
    "open_price": 59500,
    "high_price": 60100,
    "low_price": 59400,
    "close_price": 60000,
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
    "ask_vol_3": 167400,
    "foreign_buy_volume": 671400,
    "foreign_sell_volume": 253800
  },
  {
    "symbol": "TCB",
    "exchange": "HOSE",
    "reference_price": 30200,
    "ceiling_price": 32300,
    "floor_price": 28100,
    "open_price": 30250,
    "high_price": 30900,
    "low_price": 30250,
    "close_price": 30700,
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
    "ask_vol_3": 297500,
    "foreign_buy_volume": 10100,
    "foreign_sell_volume": 0
  }
]
```
