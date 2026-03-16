# market.index.summary

- Class: `IndexMarket`
- Method: `summary`
- Signature: `(show_log = False) -> pd.DataFrame`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `kbs`
- Declared signature: `(**B)`
- Default route source: `kbs`
- Default provider: `trading.Trading.index_summary`

Stock Info / Snapshot summary metrics including pricing,

## Purpose

Stock Info / Snapshot summary metrics including pricing, 
52-week ranges, and fundamental ratios.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` | Hiển thị log debug. |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.trading.Trading`
- Provider method: `index_summary`

```text
symbol, underlying_symbol, time, exchange, ceiling_price, floor_price, reference_price, open_price, high_price, low_price, close_price, basis, open_interest, total_trades, total_value, price_change, percent_change, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, foreign_buy_volume, foreign_sell_volume, last_trading_date
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `timestamp` | `time` |
| `close_price` | `close_price` |
| `price_change` | `price_change` |
| `percent_change` | `percent_change` |
| `open_price` | `open_price` |
| `high_price` | `high_price` |
| `low_price` | `low_price` |
| `reference_price` | `reference_price` |
| `advances` | `advances` |
| `declines` | `declines` |
| `no_change` | `no_change` |
| `accumulated_volume` | `accumulated_volume` |
| `accumulated_value` | `accumulated_value` |
| `total_volume` | `total_volume` |
| `put_through_volume` | `put_through_volume` |
| `put_through_value` | `put_through_value` |
| `previous_close` | `previous_close` |
- Note: Derived from `app.lib.vnstock_data_alt.explorer.kbs.trading._DERIVATIVE_STANDARD_COLUMNS`.

#### Normalized output schema

- Coverage: `declared`

```text
symbol, time, close_price, price_change, percent_change, open_price, high_price, low_price, reference_price, advances, declines, no_change, accumulated_volume, accumulated_value, total_volume, put_through_volume, put_through_value, previous_close
```

#### Live-observed sample

- Captured at: `2026-03-16T11:15:21.125519+00:00`
- Success: `True`
- Row count: `1`

```text
symbol, time, close_price, price_change, percent_change, open_price, high_price, low_price, reference_price, advances, declines, no_change, accumulated_volume, accumulated_value, total_volume, put_through_volume, put_through_value, previous_close
```
- Dtypes: `{'symbol': 'str', 'time': 'datetime64[ms]', 'close_price': 'float64', 'price_change': 'float64', 'percent_change': 'str', 'open_price': 'float64', 'high_price': 'float64', 'low_price': 'float64', 'reference_price': 'float64', 'advances': 'int64', 'declines': 'int64', 'no_change': 'int64', 'accumulated_volume': 'int64', 'accumulated_value': 'float64', 'total_volume': 'float64', 'put_through_volume': 'float64', 'put_through_value': 'float64', 'previous_close': 'float64'}`

```json
[
  {
    "symbol": "VNINDEX",
    "time": "2026-03-16T08:33:15.351000",
    "close_price": 1693.21,
    "price_change": -3.03,
    "percent_change": "-0.18",
    "open_price": 1697.34,
    "high_price": 1707.0,
    "low_price": 1680.7,
    "reference_price": 1696.24,
    "advances": 149,
    "declines": 155,
    "no_change": 74,
    "accumulated_volume": 418435480,
    "accumulated_value": 23059878527710.0,
    "total_volume": 873491825.0,
    "put_through_volume": 127185506.0,
    "put_through_value": 2714721841620.0,
    "previous_close": 1696.24
  }
]
```
