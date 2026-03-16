# market.futures.summary

- Class: `FuturesMarket`
- Method: `summary`
- Signature: `(symbol: str = None) -> pd.DataFrame`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `kbs`
- Declared signature: `(**B)`
- Default route source: `kbs`
- Default provider: `derivatives.KBSDerivatives.future_profile`

Stock Info / Snapshot summary metrics including pricing,

## Purpose

Stock Info / Snapshot summary metrics including pricing, 
52-week ranges, and fundamental ratios.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `str` | `VN30F1M` | Optional futures symbol. Defaults to instance symbol. |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.derivatives.KBSDerivatives`
- Provider method: `future_profile`

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

- Captured at: `2026-03-16T11:15:20.054177+00:00`
- Success: `True`
- Row count: `1`

```text
symbol, exchange, reference_price, ceiling_price, floor_price, open_interest, basis, first_trading_date, last_trading_date, underlying_symbol, foreign_buy_volume, foreign_sell_volume
```
- Dtypes: `{'symbol': 'str', 'exchange': 'str', 'reference_price': 'int64', 'ceiling_price': 'float64', 'floor_price': 'float64', 'open_interest': 'str', 'basis': 'str', 'first_trading_date': 'str', 'last_trading_date': 'str', 'underlying_symbol': 'str', 'foreign_buy_volume': 'int64', 'foreign_sell_volume': 'int64'}`

```json
[
  {
    "symbol": "41I1G3000",
    "exchange": "HNX",
    "reference_price": 1840,
    "ceiling_price": 1968.8,
    "floor_price": 1711.2,
    "open_interest": "40334",
    "basis": "-1.69",
    "first_trading_date": "18/07/2025",
    "last_trading_date": "19/03/2026",
    "underlying_symbol": "VN30",
    "foreign_buy_volume": 11202,
    "foreign_sell_volume": 0
  }
]
```
