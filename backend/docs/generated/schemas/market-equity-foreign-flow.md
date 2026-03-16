# market.equity.foreign_flow

- Class: `EquityMarket`
- Method: `foreign_flow`
- Signature: `(resolution = '1D', start = None, end = None, limit = 100)`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `vci`
- Declared signature: `(**B)`
- Default route source: `vci`
- Default provider: `trading.Trading.foreign_trade`

Historical or daily foreign buy/sell volume and value.

## Purpose

Historical or daily foreign buy/sell volume and value.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `resolution` | `POSITIONAL_OR_KEYWORD` | `False` | `1D` | `` | `omitted; default '1D'` |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-07` |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `100` | `` | `5` |

## Source details

### Source `vci`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.trading.Trading`
- Provider method: `foreign_trade`

```text
symbol, price, volume, highest, lowest, open, avg_price, accumulated_volume, accumulated_value, session, time, exchange
```

| Raw | Normalized |
| --- | --- |
| `trading_date` | `time` |
| `fr_buy_volume_total` | `buy_vol` |
| `fr_buy_value_total` | `buy_val` |
| `fr_sell_volume_total` | `sell_vol` |
| `fr_sell_value_total` | `sell_val` |
| `fr_net_volume_total` | `net_vol` |
| `fr_net_value_total` | `net_val` |
- Note: Derived from `app.lib.vnstock_data_alt.explorer.vci.trading._ODD_LOT_STANDARD_COLUMNS`.

#### Normalized output schema

- Coverage: `declared`

```text
time, buy_vol, buy_val, sell_vol, sell_val, net_vol, net_val
```

#### Live-observed sample

- Captured at: `2026-03-16T11:15:12.807725+00:00`
- Success: `True`
- Row count: `5`

```text
time, buy_vol, buy_val, sell_vol, sell_val, net_vol, net_val
```
- Dtypes: `{'time': 'datetime64[us]', 'buy_vol': 'float64', 'buy_val': 'float64', 'sell_vol': 'float64', 'sell_val': 'float64', 'net_vol': 'float64', 'net_val': 'float64'}`

```json
[
  {
    "time": "2025-03-07T00:00:00",
    "buy_vol": 1329100.0,
    "buy_val": 125242390000.0,
    "sell_vol": 955602.0,
    "sell_val": 89723568200.0,
    "net_vol": 373498.0,
    "net_val": 35518821800.0
  },
  {
    "time": "2025-03-06T00:00:00",
    "buy_vol": 736400.0,
    "buy_val": 68907160000.0,
    "sell_vol": 1159400.0,
    "sell_val": 108409420000.0,
    "net_vol": -423000.0,
    "net_val": -39502260000.0
  },
  {
    "time": "2025-03-05T00:00:00",
    "buy_vol": 370330.0,
    "buy_val": 34529117500.0,
    "sell_vol": 1080101.0,
    "sell_val": 100694593300.0,
    "net_vol": -709771.0,
    "net_val": -66165475800.0
  }
]
```
