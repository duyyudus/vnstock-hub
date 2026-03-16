# market.equity.odd_lot

- Class: `EquityMarket`
- Method: `odd_lot`
- Signature: `(symbols_list = None, exchange = 'HOSE', show_log = False) -> DataFrame chứa dữ liệu giao dịch lô lẻ với các cột chuẩn hóa.`
- Return type: `DataFrame chứa dữ liệu giao dịch lô lẻ với các cột chuẩn hóa.`
- Normalization mode: `contractual`
- Supported sources: `kbs, msn`
- Declared signature: `(**A)`
- Default route source: `kbs`
- Default provider: `trading.Trading.odd_lot`

Real-time pricing or trades for odd-lot execution (Lô lẻ).

## Purpose

Real-time pricing or trades for odd-lot execution (Lô lẻ).

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `symbols_list` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `['VCB']` |  | Danh sách mã chứng khoán. Nếu None, truy xuất toàn bộ sàn. |
| `exchange` | `POSITIONAL_OR_KEYWORD` | `False` | `HOSE` | `` | `HOSE` | `HOSE`, `HNX`, `UPCOM`, `HOSE` | Sàn giao dịch ('HOSE', 'HNX', 'UPCOM'). Mặc định 'HOSE'. |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` |  | Hiển thị log debug. |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.trading.Trading`
- Provider method: `odd_lot`

```text
symbol, underlying_symbol, time, exchange, ceiling_price, floor_price, reference_price, open_price, high_price, low_price, close_price, basis, open_interest, total_trades, total_value, price_change, percent_change, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, foreign_buy_volume, foreign_sell_volume, last_trading_date
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
- Note: Derived from `app.lib.vnstock_data_alt.explorer.kbs.trading._DERIVATIVE_STANDARD_COLUMNS`.

#### Normalized output schema

- Coverage: `declared`

```text
symbol, exchange, reference_price, ceiling_price, floor_price, open_price, high_price, low_price, close_price, match_price, match_vol, total_volume, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, basis, open_interest, foreign_buy_volume, foreign_sell_volume
```

Enum/value normalization:

- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

- Captured at: `2026-03-16T11:15:13.022692+00:00`
- Success: `True`
- Row count: `1`

```text
symbol, exchange, reference_price, ceiling_price, floor_price, open_price, high_price, low_price, close_price, match_vol, total_volume, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, foreign_buy_volume, foreign_sell_volume
```
- Dtypes: `{'symbol': 'str', 'exchange': 'str', 'reference_price': 'int64', 'ceiling_price': 'int64', 'floor_price': 'int64', 'open_price': 'int64', 'high_price': 'int64', 'low_price': 'int64', 'close_price': 'int64', 'match_vol': 'int64', 'total_volume': 'int64', 'bid_price_1': 'str', 'bid_vol_1': 'int64', 'bid_price_2': 'int64', 'bid_vol_2': 'int64', 'bid_price_3': 'int64', 'bid_vol_3': 'int64', 'ask_price_1': 'str', 'ask_vol_1': 'int64', 'ask_price_2': 'int64', 'ask_vol_2': 'int64', 'ask_price_3': 'int64', 'ask_vol_3': 'int64', 'foreign_buy_volume': 'int64', 'foreign_sell_volume': 'int64'}`

```json
[
  {
    "symbol": "VCB",
    "exchange": "HOSE",
    "reference_price": 59000,
    "ceiling_price": 63100,
    "floor_price": 54900,
    "open_price": 59300,
    "high_price": 59900,
    "low_price": 59000,
    "close_price": 59000,
    "match_vol": 1,
    "total_volume": 17451,
    "bid_price_1": "59000.0",
    "bid_vol_1": 916,
    "bid_price_2": 58900,
    "bid_vol_2": 4934,
    "bid_price_3": 58800,
    "bid_vol_3": 6292,
    "ask_price_1": "59200.0",
    "ask_vol_1": 43,
    "ask_price_2": 59300,
    "ask_vol_2": 76,
    "ask_price_3": 59400,
    "ask_vol_3": 25,
    "foreign_buy_volume": 356600,
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

- Captured at: `2026-03-16T11:15:13.170398+00:00`
- Success: `True`
- Row count: `1`

```text
symbol, exchange, reference_price, ceiling_price, floor_price, open_price, high_price, low_price, close_price, match_vol, total_volume, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, foreign_buy_volume, foreign_sell_volume
```
- Dtypes: `{'symbol': 'str', 'exchange': 'str', 'reference_price': 'int64', 'ceiling_price': 'int64', 'floor_price': 'int64', 'open_price': 'int64', 'high_price': 'int64', 'low_price': 'int64', 'close_price': 'int64', 'match_vol': 'int64', 'total_volume': 'int64', 'bid_price_1': 'str', 'bid_vol_1': 'int64', 'bid_price_2': 'int64', 'bid_vol_2': 'int64', 'bid_price_3': 'int64', 'bid_vol_3': 'int64', 'ask_price_1': 'str', 'ask_vol_1': 'int64', 'ask_price_2': 'int64', 'ask_vol_2': 'int64', 'ask_price_3': 'int64', 'ask_vol_3': 'int64', 'foreign_buy_volume': 'int64', 'foreign_sell_volume': 'int64'}`

```json
[
  {
    "symbol": "VCB",
    "exchange": "HOSE",
    "reference_price": 59000,
    "ceiling_price": 63100,
    "floor_price": 54900,
    "open_price": 59300,
    "high_price": 59900,
    "low_price": 59000,
    "close_price": 59000,
    "match_vol": 1,
    "total_volume": 17451,
    "bid_price_1": "59000.0",
    "bid_vol_1": 916,
    "bid_price_2": 58900,
    "bid_vol_2": 4934,
    "bid_price_3": 58800,
    "bid_vol_3": 6292,
    "ask_price_1": "59200.0",
    "ask_vol_1": 43,
    "ask_price_2": 59300,
    "ask_vol_2": 76,
    "ask_price_3": 59400,
    "ask_vol_3": 25,
    "foreign_buy_volume": 356600,
    "foreign_sell_volume": 0
  }
]
```
