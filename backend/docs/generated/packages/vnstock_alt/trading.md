# Trading

- Qualified name: `app.lib.vnstock_alt.api.trading.Trading`
- Signature: `(source: str = 'kbs', symbol: str = None, random_agent: bool = False, show_log: bool = False)`
- Supported sources: `kbs, vci`

Base adapter that uses ProviderRegistry to discover and instantiate

## Purpose

Base adapter that uses ProviderRegistry to discover and instantiate
providers from both explorer and connector packages.

## Members

### price_board

- Kind: `method`
- Signature: `(symbols_list: List[str], exchange: str = 'HOSE', show_log: Optional[bool] = False, get_all: bool = False) -> Any`
- Declared signature: `(*args: Any, **kwargs: Any) -> Any`
- Effective signature source: provider `kbs`
- Return type: `Any`
- Purpose: Retrieve the price board (order book) for a list of symbols.

#### Parameters

| Name | Kind | Required | Default | Annotation | Example | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `symbols_list` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `List[str]` |  | `['VCB', 'TCB']` | `ACB`, `VNM`, `HPG` | Danh sách mã chứng khoán (VD: ['ACB', 'VNM', 'HPG']). |
| `exchange` | `POSITIONAL_OR_KEYWORD` | `False` | `HOSE` | `str` |  | `HOSE` | `HOSE`, `HNX`, `UPCOM`, `HOSE` | Sàn giao dịch ('HOSE', 'HNX', 'UPCOM'). Mặc định 'HOSE'. |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `Optional[bool]` |  | `False` |  | Hiển thị log debug. |
| `get_all` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `bool` | `True` | `True` |  | Nếu True, trả về tất cả các cột. Nếu False (mặc định), chỉ trả về các cột tiêu chuẩn. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_alt.explorer.kbs.trading.Trading`
- Provider method: `price_board`

```text
symbol, time, exchange, ceiling_price, floor_price, reference_price, open_price, high_price, low_price, close_price, average_price, total_trades, total_value, price_change, percent_change, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, foreign_buy_volume, foreign_sell_volume
```
- Note: Derived from `app.lib.vnstock_alt.explorer.kbs.trading._PRICE_BOARD_STANDARD_COLUMNS`.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:26:38.439322+00:00`
- Success: `True`
- Row count: `2`

```text
total_trades, high_price, total_value, low_price, listed_shares, percent_change, bid_vol_1, bid_vol_2, bid_vol_3, bid_price_1, average_price, bid_price_2, bid_price_3, reference_price, exchange, foreign_buy_volume, foreign_buy_count, ask_price_1, ask_price_2, ask_price_3, floor_price, foreign_ownership_ratio, foreign_sell_volume, put_through_qty, foreign_sell_count, symbol, put_through_value, total_listed_qty, ST, open_price, price_change, ceiling_price, close_price, time, ask_vol_1, ask_vol_2, ask_vol_3
```
- Dtypes: `{'total_trades': 'int64', 'high_price': 'int64', 'total_value': 'int64', 'low_price': 'int64', 'listed_shares': 'str', 'percent_change': 'float64', 'bid_vol_1': 'int64', 'bid_vol_2': 'int64', 'bid_vol_3': 'int64', 'bid_price_1': 'str', 'average_price': 'int64', 'bid_price_2': 'int64', 'bid_price_3': 'int64', 'reference_price': 'int64', 'exchange': 'str', 'foreign_buy_volume': 'int64', 'foreign_buy_count': 'str', 'ask_price_1': 'str', 'ask_price_2': 'int64', 'ask_price_3': 'int64', 'floor_price': 'int64', 'foreign_ownership_ratio': 'int64', 'foreign_sell_volume': 'int64', 'put_through_qty': 'int64', 'foreign_sell_count': 'int64', 'symbol': 'str', 'put_through_value': 'int64', 'total_listed_qty': 'str', 'ST': 'str', 'open_price': 'int64', 'price_change': 'int64', 'ceiling_price': 'int64', 'close_price': 'int64', 'time': 'int64', 'ask_vol_1': 'int64', 'ask_vol_2': 'int64', 'ask_vol_3': 'int64'}`

```json
[
  {
    "total_trades": 2900300,
    "high_price": 60100,
    "total_value": 173411560000,
    "low_price": 59400,
    "listed_shares": "8355675094",
    "percent_change": 2.0408163265306123,
    "bid_vol_1": 33900,
    "bid_vol_2": 111300,
    "bid_vol_3": 209300,
    "bid_price_1": "59900.0",
    "average_price": 59791,
    "bid_price_2": 59800,
    "bid_price_3": 59700,
    "reference_price": 58800,
    "exchange": "HOSE",
    "foreign_buy_volume": 671400,
    "foreign_buy_count": "10",
    "ask_price_1": "60000.0",
    "ask_price_2": 60100,
    "ask_price_3": 60200,
    "floor_price": 54700,
    "foreign_ownership_ratio": 2506702528,
    "foreign_sell_volume": 790700917,
    "put_through_qty": 0,
    "foreign_sell_count": 253800,
    "symbol": "VCB",
    "put_through_value": 0,
    "total_listed_qty": "8355675094",
    "ST": "2",
    "open_price": 59500,
    "price_change": 1200,
    "ceiling_price": 62900,
    "close_price": 60000,
    "time": 1773721823119,
    "ask_vol_1": 269400,
    "ask_vol_2": 279400,
    "ask_vol_3": 167400
  },
  {
    "total_trades": 6384500,
    "high_price": 30900,
    "total_value": 195978440000,
    "low_price": 30250,
    "listed_shares": "7086240414",
    "percent_change": 1.6556291390728477,
    "bid_vol_1": 19400,
    "bid_vol_2": 45300,
    "bid_vol_3": 68900,
    "bid_price_1": "30650.0",
    "average_price": 30696,
    "bid_price_2": 30600,
    "bid_price_3": 30550,
    "reference_price": 30200,
    "exchange": "HOSE",
    "foreign_buy_volume": 10100,
    "foreign_buy_count": "10",
    "ask_price_1": "30700.0",
    "ask_price_2": 30750,
    "ask_price_3": 30800,
    "floor_price": 28100,
    "foreign_ownership_ratio": 1597139381,
    "foreign_sell_volume": 0,
    "put_through_qty": 2300016,
    "foreign_sell_count": 0,
    "symbol": "TCB",
    "put_through_value": 74290483200,
    "total_listed_qty": "7086240414",
    "ST": "2",
    "open_price": 30250,
    "price_change": 500,
    "ceiling_price": 32300,
    "close_price": 30700,
    "time": 1773721796422,
    "ask_vol_1": 4900,
    "ask_vol_2": 25000,
    "ask_vol_3": 297500
  }
]
```

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.vci.trading.Trading`
- Provider method: `price_board`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:26:38.565640+00:00`
- Success: `True`
- Row count: `2`

```text
['listing', 'symbol'], ['listing', 'ceiling'], ['listing', 'floor'], ['listing', 'ref_price'], ['listing', 'stock_type'], ['listing', 'exchange'], ['listing', 'trading_status'], ['listing', 'trading_status_code'], ['listing', 'trading_status_group'], ['listing', 'security_status'], ['listing', 'last_trading_date'], ['listing', 'issue_date'], ['listing', 'listed_share'], ['listing', 'coupon_rate'], ['listing', 'yield'], ['listing', 'sending_time'], ['listing', 'type'], ['listing', 'organ_name'], ['listing', 'mapping_symbol'], ['listing', 'product_grp_id'], ['listing', 'partition'], ['listing', 'index_type'], ['listing', 'trading_date'], ['listing', 'lst_trading_status'], ['listing', 'is_delisted'], ['bid_ask', 'transaction_time'], ['bid_ask', 'bid_count'], ['bid_ask', 'ask_count'], ['match', 'accumulated_value'], ['match', 'accumulated_volume'], ['match', 'accumulated_value_g1'], ['match', 'accumulated_volume_g1'], ['match', 'match_price_ato'], ['match', 'match_volume_ato'], ['match', 'match_price_atc'], ['match', 'match_volume_atc'], ['match', 'trading_session_id'], ['match', 'is_last_ato'], ['match', 'avg_match_price'], ['match', 'current_room'], ['match', 'foreign_buy_volume'], ['match', 'foreign_sell_volume'], ['match', 'foreign_buy_value'], ['match', 'foreign_sell_value'], ['match', 'highest'], ['match', 'lowest'], ['match', 'match_price'], ['match', 'open_price'], ['match', 'first_time_match_price'], ['match', 'match_type'], ['match', 'match_vol'], ['match', 'sending_time'], ['match', 'total_room'], ['match', 'total_buy_orders'], ['match', 'total_sell_orders'], ['match', 'bid_count'], ['match', 'ask_count'], ['match', 'underlying'], ['match', 'open_interest'], ['match', 'stock_type'], ['match', 'partition'], ['match', 'is_match_price'], ['match', 'ceiling_price'], ['match', 'floor_price'], ['match', 'reference_price'], ['match', 'last_ato'], ['bid_ask', 'bid_1_price'], ['bid_ask', 'bid_1_volume'], ['bid_ask', 'bid_2_price'], ['bid_ask', 'bid_2_volume'], ['bid_ask', 'bid_3_price'], ['bid_ask', 'bid_3_volume'], ['bid_ask', 'ask_1_price'], ['bid_ask', 'ask_1_volume'], ['bid_ask', 'ask_2_price'], ['bid_ask', 'ask_2_volume'], ['bid_ask', 'ask_3_price'], ['bid_ask', 'ask_3_volume']
```
- Dtypes: `{"('listing', 'symbol')": 'str', "('listing', 'ceiling')": 'int64', "('listing', 'floor')": 'int64', "('listing', 'ref_price')": 'int64', "('listing', 'stock_type')": 'str', "('listing', 'exchange')": 'str', "('listing', 'trading_status')": 'str', "('listing', 'trading_status_code')": 'str', "('listing', 'trading_status_group')": 'str', "('listing', 'security_status')": 'str', "('listing', 'last_trading_date')": 'object', "('listing', 'issue_date')": 'object', "('listing', 'listed_share')": 'int64', "('listing', 'coupon_rate')": 'int64', "('listing', 'yield')": 'int64', "('listing', 'sending_time')": 'str', "('listing', 'type')": 'str', "('listing', 'organ_name')": 'str', "('listing', 'mapping_symbol')": 'object', "('listing', 'product_grp_id')": 'str', "('listing', 'partition')": 'int64', "('listing', 'index_type')": 'object', "('listing', 'trading_date')": 'str', "('listing', 'lst_trading_status')": 'object', "('listing', 'is_delisted')": 'int64', "('bid_ask', 'transaction_time')": 'str', "('bid_ask', 'bid_count')": 'int64', "('bid_ask', 'ask_count')": 'int64', "('match', 'accumulated_value')": 'float64', "('match', 'accumulated_volume')": 'int64', "('match', 'accumulated_value_g1')": 'float64', "('match', 'accumulated_volume_g1')": 'int64', "('match', 'match_price_ato')": 'int64', "('match', 'match_volume_ato')": 'int64', "('match', 'match_price_atc')": 'int64', "('match', 'match_volume_atc')": 'int64', "('match', 'trading_session_id')": 'str', "('match', 'is_last_ato')": 'bool', "('match', 'avg_match_price')": 'float64', "('match', 'current_room')": 'int64', "('match', 'foreign_buy_volume')": 'int64', "('match', 'foreign_sell_volume')": 'int64', "('match', 'foreign_buy_value')": 'int64', "('match', 'foreign_sell_value')": 'int64', "('match', 'highest')": 'int64', "('match', 'lowest')": 'int64', "('match', 'match_price')": 'int64', "('match', 'open_price')": 'int64', "('match', 'first_time_match_price')": 'str', "('match', 'match_type')": 'str', "('match', 'match_vol')": 'int64', "('match', 'sending_time')": 'str', "('match', 'total_room')": 'int64', "('match', 'total_buy_orders')": 'int64', "('match', 'total_sell_orders')": 'int64', "('match', 'bid_count')": 'int64', "('match', 'ask_count')": 'int64', "('match', 'underlying')": 'object', "('match', 'open_interest')": 'object', "('match', 'stock_type')": 'str', "('match', 'partition')": 'int64', "('match', 'is_match_price')": 'bool', "('match', 'ceiling_price')": 'int64', "('match', 'floor_price')": 'int64', "('match', 'reference_price')": 'int64', "('match', 'last_ato')": 'bool', "('bid_ask', 'bid_1_price')": 'int64', "('bid_ask', 'bid_1_volume')": 'int64', "('bid_ask', 'bid_2_price')": 'int64', "('bid_ask', 'bid_2_volume')": 'int64', "('bid_ask', 'bid_3_price')": 'int64', "('bid_ask', 'bid_3_volume')": 'int64', "('bid_ask', 'ask_1_price')": 'int64', "('bid_ask', 'ask_1_volume')": 'int64', "('bid_ask', 'ask_2_price')": 'int64', "('bid_ask', 'ask_2_volume')": 'int64', "('bid_ask', 'ask_3_price')": 'int64', "('bid_ask', 'ask_3_volume')": 'int64'}`

```json
[
  {
    "('listing', 'symbol')": "VCB",
    "('listing', 'ceiling')": 62900,
    "('listing', 'floor')": 54700,
    "('listing', 'ref_price')": 58800,
    "('listing', 'stock_type')": "STOCK",
    "('listing', 'exchange')": "HSX",
    "('listing', 'trading_status')": "TRADING_ACTIVATED",
    "('listing', 'trading_status_code')": "20",
    "('listing', 'trading_status_group')": "2",
    "('listing', 'security_status')": "N",
    "('listing', 'last_trading_date')": null,
    "('listing', 'issue_date')": null,
    "('listing', 'listed_share')": 8355675094,
    "('listing', 'coupon_rate')": 0,
    "('listing', 'yield')": 0,
    "('listing', 'sending_time')": "20260317 01:43:58.508",
    "('listing', 'type')": "STOCK",
    "('listing', 'organ_name')": "Ngân hàng Thương mại Cổ phần Ngoại thương Việt Nam",
    "('listing', 'mapping_symbol')": null,
    "('listing', 'product_grp_id')": "STO",
    "('listing', 'partition')": 4,
    "('listing', 'index_type')": null,
    "('listing', 'trading_date')": "2026-03-17",
    "('listing', 'lst_trading_status')": null,
    "('listing', 'is_delisted')": 0,
    "('bid_ask', 'transaction_time')": "042958739",
    "('bid_ask', 'bid_count')": 0,
    "('bid_ask', 'ask_count')": 0,
    "('match', 'accumulated_value')": 173411.56,
    "('match', 'accumulated_volume')": 2900300,
    "('match', 'accumulated_value_g1')": 173411.56,
    "('match', 'accumulated_volume_g1')": 2900300,
    "('match', 'match_price_ato')": 59500,
    "('match', 'match_volume_ato')": 64300,
    "('match', 'match_price_atc')": 0,
    "('match', 'match_volume_atc')": 0,
    "('match', 'trading_session_id')": "40",
    "('match', 'is_last_ato')": false,
    "('match', 'avg_match_price')": 59790.90438920112,
    "('match', 'current_room')": 790700917,
    "('match', 'foreign_buy_volume')": 671400,
    "('match', 'foreign_sell_volume')": 253800,
    "('match', 'foreign_buy_value')": 40147440000,
    "('match', 'foreign_sell_value')": 15172170000,
    "('match', 'highest')": 60100,
    "('match', 'lowest')": 59400,
    "('match', 'match_price')": 60000,
    "('match', 'open_price')": 59500,
    "('match', 'first_time_match_price')": "2026-03-17T02:15:00.006Z",
    "('match', 'match_type')": "b",
    "('match', 'match_vol')": 200,
    "('match', 'sending_time')": "20260317 04:29:58.739",
    "('match', 'total_room')": 2506702528,
    "('match', 'total_buy_orders')": 0,
    "('match', 'total_sell_orders')": 0,
    "('match', 'bid_count')": 0,
    "('match', 'ask_count')": 0,
    "('match', 'underlying')": null,
    "('match', 'open_interest')": null,
    "('match', 'stock_type')": "STOCK",
    "('match', 'partition')": 4,
    "('match', 'is_match_price')": false,
    "('match', 'ceiling_price')": 62900,
    "('match', 'floor_price')": 54700,
    "('match', 'reference_price')": 58800,
    "('match', 'last_ato')": false,
    "('bid_ask', 'bid_1_price')": 59900,
    "('bid_ask', 'bid_1_volume')": 33900,
    "('bid_ask', 'bid_2_price')": 59800,
    "('bid_ask', 'bid_2_volume')": 111300,
    "('bid_ask', 'bid_3_price')": 59700,
    "('bid_ask', 'bid_3_volume')": 209300,
    "('bid_ask', 'ask_1_price')": 60000,
    "('bid_ask', 'ask_1_volume')": 269400,
    "('bid_ask', 'ask_2_price')": 60100,
    "('bid_ask', 'ask_2_volume')": 279400,
    "('bid_ask', 'ask_3_price')": 60200,
    "('bid_ask', 'ask_3_volume')": 167400
  },
  {
    "('listing', 'symbol')": "TCB",
    "('listing', 'ceiling')": 32300,
    "('listing', 'floor')": 28100,
    "('listing', 'ref_price')": 30200,
    "('listing', 'stock_type')": "STOCK",
    "('listing', 'exchange')": "HSX",
    "('listing', 'trading_status')": "TRADING_ACTIVATED",
    "('listing', 'trading_status_code')": "20",
    "('listing', 'trading_status_group')": "2",
    "('listing', 'security_status')": "N",
    "('listing', 'last_trading_date')": null,
    "('listing', 'issue_date')": null,
    "('listing', 'listed_share')": 7086240414,
    "('listing', 'coupon_rate')": 0,
    "('listing', 'yield')": 0,
    "('listing', 'sending_time')": "20260317 01:43:55.078",
    "('listing', 'type')": "STOCK",
    "('listing', 'organ_name')": "Ngân hàng Thương mại Cổ phần Kỹ thương Việt Nam",
    "('listing', 'mapping_symbol')": null,
    "('listing', 'product_grp_id')": "STO",
    "('listing', 'partition')": 5,
    "('listing', 'index_type')": null,
    "('listing', 'trading_date')": "2026-03-17",
    "('listing', 'lst_trading_status')": null,
    "('listing', 'is_delisted')": 0,
    "('bid_ask', 'transaction_time')": "042956232",
    "('bid_ask', 'bid_count')": 0,
    "('bid_ask', 'ask_count')": 0,
    "('match', 'accumulated_value')": 195978.44,
    "('match', 'accumulated_volume')": 6384500,
    "('match', 'accumulated_value_g1')": 195978.44,
    "('match', 'accumulated_volume_g1')": 6384500,
    "('match', 'match_price_ato')": 30250,
    "('match', 'match_volume_ato')": 592300,
    "('match', 'match_price_atc')": 0,
    "('match', 'match_volume_atc')": 0,
    "('match', 'trading_session_id')": "40",
    "('match', 'is_last_ato')": false,
    "('match', 'avg_match_price')": 30695.97305975409,
    "('match', 'current_room')": 0,
    "('match', 'foreign_buy_volume')": 10100,
    "('match', 'foreign_sell_volume')": 16,
    "('match', 'foreign_buy_value')": 305525000,
    "('match', 'foreign_sell_value')": 483200,
    "('match', 'highest')": 30900,
    "('match', 'lowest')": 30250,
    "('match', 'match_price')": 30700,
    "('match', 'open_price')": 30250,
    "('match', 'first_time_match_price')": "2026-03-17T02:15:00.006Z",
    "('match', 'match_type')": "b",
    "('match', 'match_vol')": 500,
    "('match', 'sending_time')": "20260317 04:29:56.232",
    "('match', 'total_room')": 1597139381,
    "('match', 'total_buy_orders')": 0,
    "('match', 'total_sell_orders')": 0,
    "('match', 'bid_count')": 0,
    "('match', 'ask_count')": 0,
    "('match', 'underlying')": null,
    "('match', 'open_interest')": null,
    "('match', 'stock_type')": "STOCK",
    "('match', 'partition')": 5,
    "('match', 'is_match_price')": false,
    "('match', 'ceiling_price')": 32300,
    "('match', 'floor_price')": 28100,
    "('match', 'reference_price')": 30200,
    "('match', 'last_ato')": false,
    "('bid_ask', 'bid_1_price')": 30650,
    "('bid_ask', 'bid_1_volume')": 19400,
    "('bid_ask', 'bid_2_price')": 30600,
    "('bid_ask', 'bid_2_volume')": 45300,
    "('bid_ask', 'bid_3_price')": 30550,
    "('bid_ask', 'bid_3_volume')": 68900,
    "('bid_ask', 'ask_1_price')": 30700,
    "('bid_ask', 'ask_1_volume')": 4900,
    "('bid_ask', 'ask_2_price')": 30750,
    "('bid_ask', 'ask_2_volume')": 25000,
    "('bid_ask', 'ask_3_price')": 30800,
    "('bid_ask', 'ask_3_volume')": 297500
  }
]
```

#### Notes / caveats

Retrieve the price board (order book) for a list of symbols.
