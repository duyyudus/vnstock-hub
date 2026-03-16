# market.equity.block_trades

- Class: `EquityMarket`
- Method: `block_trades`
- Signature: `(limit: int = 1000)`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `kbs, msn`
- Declared signature: `(limit=1000, **A)`
- Default route source: `kbs`
- Default provider: `trading.Trading.put_through`

Real-time or historical data for negotiated/block trades (giao dịch thoả thuận).

## Purpose

Real-time or historical data for negotiated/block trades (giao dịch thoả thuận).

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `1000` | `int` | `5` | Number of records to fetch (default: 1000). |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.trading.Trading`
- Provider method: `put_through`

```text
symbol, time, exchange, match_price, match_volume, trading_date, reference_price, floor_price
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `time` | `time` |
| `exchange` | `exchange` |
| `match_price` | `match_price` |
| `match_volume` | `match_volume` |
| `trading_date` | `trading_date` |
| `reference_price` | `reference_price` |
| `floor_price` | `floor_price` |
- Note: Derived from `app.lib.vnstock_data_alt.explorer.kbs.trading._PUT_THROUGH_STANDARD_COLUMNS`.

#### Normalized output schema

- Coverage: `declared`

```text
symbol, time, exchange, match_price, match_volume, trading_date, reference_price, floor_price
```

Enum/value normalization:

- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

- Captured at: `2026-03-16T11:15:10.334722+00:00`
- Success: `True`
- Row count: `0`

```text
transaction_id, match_volume, match_price, counterparty_id, contract_number, floor_price, listed_shares, market_id, is_active, symbol, trading_date, reference_price, time, exchange, total_volume, total_value
```
- Dtypes: `{'transaction_id': 'str', 'match_volume': 'int64', 'match_price': 'str', 'counterparty_id': 'int64', 'contract_number': 'str', 'floor_price': 'int64', 'listed_shares': 'int64', 'market_id': 'int64', 'is_active': 'bool', 'symbol': 'str', 'trading_date': 'str', 'reference_price': 'int64', 'time': 'str', 'exchange': 'str', 'total_volume': 'int64', 'total_value': 'int64'}`

### Source `msn`

#### Raw output contract

- Coverage: `declared`

```text
symbol, time, exchange, match_price, match_volume, trading_date, reference_price, floor_price
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `time` | `time` |
| `exchange` | `exchange` |
| `match_price` | `match_price` |
| `match_volume` | `match_volume` |
| `trading_date` | `trading_date` |
| `reference_price` | `reference_price` |
| `floor_price` | `floor_price` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, time, exchange, match_price, match_volume, trading_date, reference_price, floor_price
```

Enum/value normalization:

- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

- Captured at: `2026-03-16T11:15:12.573441+00:00`
- Success: `True`
- Row count: `0`

```text
transaction_id, match_volume, match_price, counterparty_id, contract_number, floor_price, listed_shares, market_id, is_active, symbol, trading_date, reference_price, time, exchange, total_volume, total_value
```
- Dtypes: `{'transaction_id': 'str', 'match_volume': 'int64', 'match_price': 'str', 'counterparty_id': 'int64', 'contract_number': 'str', 'floor_price': 'int64', 'listed_shares': 'int64', 'market_id': 'int64', 'is_active': 'bool', 'symbol': 'str', 'trading_date': 'str', 'reference_price': 'int64', 'time': 'str', 'exchange': 'str', 'total_volume': 'int64', 'total_value': 'int64'}`
