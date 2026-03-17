# market.equity.volume_profile

- Class: `EquityMarket`
- Method: `volume_profile`
- Signature: `(show_log = False) -> DataFrame chứa dữ liệu khớp lệnh theo giá với các cột chuẩn hóa.`
- Return type: `DataFrame chứa dữ liệu khớp lệnh theo giá với các cột chuẩn hóa.`
- Normalization mode: `contractual`
- Supported sources: `kbs, msn`
- Declared signature: `(**B)`
- Default route source: `kbs`
- Default provider: `trading.Trading.matched_by_price`

Aggregated volume distributed across executed price levels (Volume Profile).

## Purpose

Aggregated volume distributed across executed price levels (Volume Profile).

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` | Hiển thị log debug. |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.trading.Trading`
- Provider method: `matched_by_price`

```text
symbol, underlying_symbol, time, exchange, ceiling_price, floor_price, reference_price, open_price, high_price, low_price, close_price, basis, open_interest, total_trades, total_value, price_change, percent_change, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, foreign_buy_volume, foreign_sell_volume, last_trading_date
```

| Raw | Normalized |
| --- | --- |
| `price` | `price` |
| `buyVol` | `buy_volume` |
| `sellVol` | `sell_volume` |
| `unknownVol` | `unknown_volume` |
| `totalVol` | `total_volume` |
| `percent` | `match_percent` |
- Note: Derived from `app.lib.vnstock_data_alt.explorer.kbs.trading._DERIVATIVE_STANDARD_COLUMNS`.

#### Normalized output schema

- Coverage: `declared`

```text
price, buy_volume, sell_volume, unknown_volume, total_volume, match_percent
```

#### Live-observed sample

- Captured at: `2026-03-17T05:27:00.721529+00:00`
- Success: `True`
- Row count: `8`

```text
price, buy_volume, sell_volume, unknown_volume, total_volume
```
- Dtypes: `{'price': 'int64', 'buy_volume': 'int64', 'sell_volume': 'int64', 'unknown_volume': 'int64', 'total_volume': 'int64'}`

```json
[
  {
    "price": 59400,
    "buy_volume": 0,
    "sell_volume": 2400,
    "unknown_volume": 0,
    "total_volume": 2400
  },
  {
    "price": 59500,
    "buy_volume": 55400,
    "sell_volume": 2000,
    "unknown_volume": 64300,
    "total_volume": 121700
  },
  {
    "price": 59600,
    "buy_volume": 20700,
    "sell_volume": 150300,
    "unknown_volume": 0,
    "total_volume": 171000
  }
]
```

### Source `msn`

#### Raw output contract

- Coverage: `declared`

```text
price, buyVol, sellVol, unknownVol, totalVol, percent
```

| Raw | Normalized |
| --- | --- |
| `price` | `price` |
| `buyVol` | `buy_volume` |
| `sellVol` | `sell_volume` |
| `unknownVol` | `unknown_volume` |
| `totalVol` | `total_volume` |
| `percent` | `match_percent` |

#### Normalized output schema

- Coverage: `declared`

```text
price, buy_volume, sell_volume, unknown_volume, total_volume, match_percent
```

#### Live-observed sample

- Captured at: `2026-03-17T05:27:01.994622+00:00`
- Success: `True`
- Row count: `8`

```text
price, buy_volume, sell_volume, unknown_volume, total_volume
```
- Dtypes: `{'price': 'int64', 'buy_volume': 'int64', 'sell_volume': 'int64', 'unknown_volume': 'int64', 'total_volume': 'int64'}`

```json
[
  {
    "price": 59400,
    "buy_volume": 0,
    "sell_volume": 2400,
    "unknown_volume": 0,
    "total_volume": 2400
  },
  {
    "price": 59500,
    "buy_volume": 55400,
    "sell_volume": 2000,
    "unknown_volume": 64300,
    "total_volume": 121700
  },
  {
    "price": 59600,
    "buy_volume": 20700,
    "sell_volume": 150300,
    "unknown_volume": 0,
    "total_volume": 171000
  }
]
```
