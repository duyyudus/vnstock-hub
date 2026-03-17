# Trading

- Qualified name: `app.lib.vnstock_data_alt.api.trading.Trading`
- Signature: `(source='KBS', symbol='', random_agent=False, show_log=False)`
- Supported sources: `cafef, kbs, vci`

Base adapter that uses ProviderRegistry to discover and instantiate

## Purpose

Base adapter that uses ProviderRegistry to discover and instantiate
providers from both explorer and connector packages.

## Members

### foreign_trade

- Kind: `method`
- Signature: `(start, end, page = 1, limit = None)`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `cafef`
- Purpose: Retrieve foreign trade data for the given symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` | `2025-03-07` |
| `page` | `POSITIONAL_OR_KEYWORD` | `False` | `1` | `` | `1` |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `5` |

#### Source details

##### Source `cafef`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.cafef.trading.Trading`
- Provider method: `foreign_trade`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:02.890618+00:00`
- Success: `True`
- Row count: `5`

```text
fr_net_volume, fr_net_value, fr_buy_volume, fr_buy_value, fr_sell_volume, fr_sell_value, fr_remaining_room, fr_ownership
```
- Dtypes: `{'fr_net_volume': 'int64', 'fr_net_value': 'int64', 'fr_buy_volume': 'int64', 'fr_buy_value': 'int64', 'fr_sell_volume': 'int64', 'fr_sell_value': 'int64', 'fr_remaining_room': 'int64', 'fr_ownership': 'float64'}`

```json
[
  {
    "fr_net_volume": 373500,
    "fr_net_value": 35519010000,
    "fr_buy_volume": 1329100,
    "fr_buy_value": 125242390000,
    "fr_sell_volume": 955600,
    "fr_sell_value": 89723380000,
    "fr_remaining_room": 405027508,
    "fr_ownership": 22.75
  },
  {
    "fr_net_volume": -423000,
    "fr_net_value": -39502260000,
    "fr_buy_volume": 736400,
    "fr_buy_value": 68907160000,
    "fr_sell_volume": 1159400,
    "fr_sell_value": 108409420000,
    "fr_remaining_room": 405270507,
    "fr_ownership": 22.75
  },
  {
    "fr_net_volume": -709800,
    "fr_net_value": -66168180000,
    "fr_buy_volume": 370300,
    "fr_buy_value": 34526320000,
    "fr_sell_volume": 1080100,
    "fr_sell_value": 100694500000,
    "fr_remaining_room": 405049681,
    "fr_ownership": 22.75
  }
]
```

##### Source `kbs`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `foreign_trade` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.trading.Trading`
- Provider method: `foreign_trade`

```text
symbol, price, volume, highest, lowest, open, avg_price, accumulated_volume, accumulated_value, session, time, exchange
```
- Note: Derived from `app.lib.vnstock_data_alt.explorer.vci.trading._ODD_LOT_STANDARD_COLUMNS`.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:03.173043+00:00`
- Success: `True`
- Row count: `5`

```text
trading_date, fr_buy_value_matched, fr_buy_volume_matched, fr_sell_value_matched, fr_sell_volume_matched, fr_buy_value_deal, fr_buy_volume_deal, fr_sell_value_deal, fr_sell_volume_deal, fr_buy_value_total, fr_buy_volume_total, fr_sell_value_total, fr_sell_volume_total, fr_total_room, fr_current_room, fr_room_percentage, fr_owned_percentage, fr_available_percentage, fr_owned, fr_net_volume_total, fr_net_value_total, fr_net_volume_matched, fr_net_value_matched, fr_net_volume_deal, fr_net_value_deal
```
- Dtypes: `{'trading_date': 'datetime64[us]', 'fr_buy_value_matched': 'float64', 'fr_buy_volume_matched': 'float64', 'fr_sell_value_matched': 'float64', 'fr_sell_volume_matched': 'float64', 'fr_buy_value_deal': 'float64', 'fr_buy_volume_deal': 'float64', 'fr_sell_value_deal': 'float64', 'fr_sell_volume_deal': 'float64', 'fr_buy_value_total': 'float64', 'fr_buy_volume_total': 'float64', 'fr_sell_value_total': 'float64', 'fr_sell_volume_total': 'float64', 'fr_total_room': 'float64', 'fr_current_room': 'float64', 'fr_room_percentage': 'float64', 'fr_owned_percentage': 'float64', 'fr_available_percentage': 'float64', 'fr_owned': 'float64', 'fr_net_volume_total': 'float64', 'fr_net_value_total': 'float64', 'fr_net_volume_matched': 'float64', 'fr_net_value_matched': 'float64', 'fr_net_volume_deal': 'float64', 'fr_net_value_deal': 'float64'}`

```json
[
  {
    "trading_date": "2025-03-07T00:00:00",
    "fr_buy_value_matched": 125242390000.0,
    "fr_buy_volume_matched": 1329100.0,
    "fr_sell_value_matched": 89723568200.0,
    "fr_sell_volume_matched": 955602.0,
    "fr_buy_value_deal": 0.0,
    "fr_buy_volume_deal": 0.0,
    "fr_sell_value_deal": 0.0,
    "fr_sell_volume_deal": 0.0,
    "fr_buy_value_total": 125242390000.0,
    "fr_buy_volume_total": 1329100.0,
    "fr_sell_value_total": 89723568200.0,
    "fr_sell_volume_total": 955602.0,
    "fr_total_room": 1676727378.0,
    "fr_current_room": 405027508.0,
    "fr_room_percentage": 0.3,
    "fr_owned_percentage": 0.2275,
    "fr_available_percentage": 0.0725,
    "fr_owned": 1271699870.0,
    "fr_net_volume_total": 373498.0,
    "fr_net_value_total": 35518821800.0,
    "fr_net_volume_matched": 373498.0,
    "fr_net_value_matched": 35518821800.0,
    "fr_net_volume_deal": 0.0,
    "fr_net_value_deal": 0.0
  },
  {
    "trading_date": "2025-03-06T00:00:00",
    "fr_buy_value_matched": 68907160000.0,
    "fr_buy_volume_matched": 736400.0,
    "fr_sell_value_matched": 108409420000.0,
    "fr_sell_volume_matched": 1159400.0,
    "fr_buy_value_deal": 0.0,
    "fr_buy_volume_deal": 0.0,
    "fr_sell_value_deal": 0.0,
    "fr_sell_volume_deal": 0.0,
    "fr_buy_value_total": 68907160000.0,
    "fr_buy_volume_total": 736400.0,
    "fr_sell_value_total": 108409420000.0,
    "fr_sell_volume_total": 1159400.0,
    "fr_total_room": 1676727378.0,
    "fr_current_room": 405270507.0,
    "fr_room_percentage": 0.3,
    "fr_owned_percentage": 0.2275,
    "fr_available_percentage": 0.0725,
    "fr_owned": 1271456871.0,
    "fr_net_volume_total": -423000.0,
    "fr_net_value_total": -39502260000.0,
    "fr_net_volume_matched": -423000.0,
    "fr_net_value_matched": -39502260000.0,
    "fr_net_volume_deal": 0.0,
    "fr_net_value_deal": 0.0
  },
  {
    "trading_date": "2025-03-05T00:00:00",
    "fr_buy_value_matched": 34529117500.0,
    "fr_buy_volume_matched": 370330.0,
    "fr_sell_value_matched": 100694593300.0,
    "fr_sell_volume_matched": 1080101.0,
    "fr_buy_value_deal": 0.0,
    "fr_buy_volume_deal": 0.0,
    "fr_sell_value_deal": 0.0,
    "fr_sell_volume_deal": 0.0,
    "fr_buy_value_total": 34529117500.0,
    "fr_buy_volume_total": 370330.0,
    "fr_sell_value_total": 100694593300.0,
    "fr_sell_volume_total": 1080101.0,
    "fr_total_room": 1676727378.0,
    "fr_current_room": 405049681.0,
    "fr_room_percentage": 0.3,
    "fr_owned_percentage": 0.2275,
    "fr_available_percentage": 0.0725,
    "fr_owned": 1271677697.0,
    "fr_net_volume_total": -709771.0,
    "fr_net_value_total": -66165475800.0,
    "fr_net_volume_matched": -709771.0,
    "fr_net_value_matched": -66165475800.0,
    "fr_net_volume_deal": 0.0,
    "fr_net_value_deal": 0.0
  }
]
```

#### Notes / caveats

Retrieve foreign trade data for the given symbol.

### insider_deal

- Kind: `method`
- Signature: `(start, end, page = 1, limit = None)`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `cafef`
- Purpose: Retrieve insider deal data for the given symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` | `2025-03-07` |
| `page` | `POSITIONAL_OR_KEYWORD` | `False` | `1` | `` | `1` |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `5` |

#### Source details

##### Source `cafef`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.cafef.trading.Trading`
- Provider method: `insider_deal`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:03.503717+00:00`
- Success: `True`
- Row count: `2`

```text
symbol, holder_id, transaction_man, transaction_man_position, related_man_position, related_man, volume_before_transaction, plan_buy_volume, plan_sell_volume, plan_begin_date, plan_end_date, real_buy_volume, real_sell_volume, real_end_date, published_date, order_date, volume_after_transaction, transaction_note, shareholder_code, ownership_percentage
```
- Dtypes: `{'symbol': 'str', 'holder_id': 'str', 'transaction_man': 'str', 'transaction_man_position': 'str', 'related_man_position': 'str', 'related_man': 'str', 'volume_before_transaction': 'int64', 'plan_buy_volume': 'int64', 'plan_sell_volume': 'int64', 'plan_begin_date': 'datetime64[ms]', 'plan_end_date': 'datetime64[ms]', 'real_buy_volume': 'int64', 'real_sell_volume': 'int64', 'real_end_date': 'datetime64[ms]', 'published_date': 'datetime64[ms]', 'order_date': 'datetime64[ms]', 'volume_after_transaction': 'int64', 'transaction_note': 'str', 'shareholder_code': 'str', 'ownership_percentage': 'float64'}`

```json
[
  {
    "symbol": "VCB",
    "holder_id": "0",
    "transaction_man": "Phùng Nguyễn Hải Yến",
    "transaction_man_position": "Phó Tổng GĐ",
    "related_man_position": "",
    "related_man": "",
    "volume_before_transaction": 22339,
    "plan_buy_volume": 20000,
    "plan_sell_volume": 0,
    "plan_begin_date": "2025-04-16T17:00:00",
    "plan_end_date": "2025-05-15T17:00:00",
    "real_buy_volume": 20000,
    "real_sell_volume": 0,
    "real_end_date": "2025-05-13T17:00:00",
    "published_date": "2025-04-17T17:00:00",
    "order_date": "2025-04-16T17:00:00",
    "volume_after_transaction": 42339,
    "transaction_note": "",
    "shareholder_code": "CEO_30014",
    "ownership_percentage": 0.0005067095060984668
  },
  {
    "symbol": "VCB",
    "holder_id": "0",
    "transaction_man": "Phùng Nguyễn Hải Yến",
    "transaction_man_position": "Phó Tổng GĐ",
    "related_man_position": "",
    "related_man": "",
    "volume_before_transaction": 4943,
    "plan_buy_volume": 10000,
    "plan_sell_volume": 0,
    "plan_begin_date": "2025-02-11T17:00:00",
    "plan_end_date": "2025-03-11T17:00:00",
    "real_buy_volume": 10000,
    "real_sell_volume": 0,
    "real_end_date": "2025-02-18T17:00:00",
    "published_date": "2025-02-09T17:00:00",
    "order_date": "2025-02-11T17:00:00",
    "volume_after_transaction": 14943,
    "transaction_note": "",
    "shareholder_code": "CEO_30014",
    "ownership_percentage": 0.00017883653722641982
  }
]
```

##### Source `kbs`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `insider_deal` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.trading.Trading`
- Provider method: `insider_deal`

```text
symbol, price, volume, highest, lowest, open, avg_price, accumulated_volume, accumulated_value, session, time, exchange
```
- Note: Derived from `app.lib.vnstock_data_alt.explorer.vci.trading._ODD_LOT_STANDARD_COLUMNS`.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:03.819875+00:00`
- Success: `True`
- Row count: `5`

```text
start_date, end_date, public_date, share_before_trade, share_after_trade, share_register, share_acquire, ownership_after_trade, trader_person_id, event_name, organ_name, source_url, trade_status, trader_name, trader_position, action_type, role_name, relative_name, trader_organ_name
```
- Dtypes: `{'start_date': 'datetime64[us]', 'end_date': 'datetime64[us]', 'public_date': 'datetime64[us]', 'share_before_trade': 'float64', 'share_after_trade': 'float64', 'share_register': 'float64', 'share_acquire': 'float64', 'ownership_after_trade': 'float64', 'trader_person_id': 'float64', 'event_name': 'str', 'organ_name': 'str', 'source_url': 'str', 'trade_status': 'str', 'trader_name': 'str', 'trader_position': 'str', 'action_type': 'str', 'role_name': 'str', 'relative_name': 'str', 'trader_organ_name': 'str'}`

```json
[
  {
    "start_date": "2026-02-26T00:00:00",
    "end_date": "2026-02-26T00:00:00",
    "public_date": "2026-03-04T00:00:00",
    "share_before_trade": 0.0,
    "share_after_trade": 18000.0,
    "share_register": 18000.0,
    "share_acquire": 18000.0,
    "ownership_after_trade": 2.2e-06,
    "trader_person_id": 4818637.0,
    "event_name": "Giao dịch nội bộ: Giao dịch cá nhân",
    "organ_name": "Ngân hàng Thương mại Cổ phần Ngoại thương Việt Nam",
    "source_url": "http://fiinpro.com/News/Detail/11905193?lang=vi-vn",
    "trade_status": "Đã thực hiện xong",
    "trader_name": "Nguyễn Tuấn Anh",
    "trader_position": "Thành viên Hội đồng Quản trị",
    "action_type": "Mua",
    "role_name": "Nguyễn Tuấn Anh",
    "relative_name": "Nguyễn Tuấn Anh",
    "trader_organ_name": NaN
  },
  {
    "start_date": "2026-02-23T00:00:00",
    "end_date": "2026-02-23T00:00:00",
    "public_date": "2026-02-24T00:00:00",
    "share_before_trade": 42339.0,
    "share_after_trade": 52339.0,
    "share_register": 10000.0,
    "share_acquire": 10000.0,
    "ownership_after_trade": 6.3e-06,
    "trader_person_id": 9642.0,
    "event_name": "Giao dịch nội bộ: Giao dịch cá nhân",
    "organ_name": "Ngân hàng Thương mại Cổ phần Ngoại thương Việt Nam",
    "source_url": "http://fiinpro.com/News/Detail/11895561?lang=vi-vn",
    "trade_status": "Đã thực hiện xong",
    "trader_name": "Phùng Nguyễn Hải Yến",
    "trader_position": "Phó Tổng Giám đốc",
    "action_type": "Mua",
    "role_name": "Phùng Nguyễn Hải Yến",
    "relative_name": "Phùng Nguyễn Hải Yến",
    "trader_organ_name": NaN
  },
  {
    "start_date": "2025-08-28T00:00:00",
    "end_date": "2025-08-29T00:00:00",
    "public_date": "2025-09-05T00:00:00",
    "share_before_trade": 58757.0,
    "share_after_trade": 0.0,
    "share_register": 58757.0,
    "share_acquire": -58757.0,
    "ownership_after_trade": 0.0,
    "trader_person_id": NaN,
    "event_name": "Giao dịch nội bộ: Giao dịch tổ chức",
    "organ_name": "Ngân hàng Thương mại Cổ phần Ngoại thương Việt Nam",
    "source_url": "http://fiinpro.com/News/Detail/11703100?lang=vi-vn",
    "trade_status": "Đã thực hiện xong",
    "trader_name": NaN,
    "trader_position": NaN,
    "action_type": "Bán",
    "role_name": NaN,
    "relative_name": NaN,
    "trader_organ_name": "Ngân hàng Thương mại TNHH MTV Ngoại thương Công nghệ Số"
  }
]
```

#### Notes / caveats

Retrieve insider deal data for the given symbol.

### matched_by_price

- Kind: `method`
- Signature: `(show_log = False) -> DataFrame chứa dữ liệu khớp lệnh theo giá với các cột chuẩn hóa.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa dữ liệu khớp lệnh theo giá với các cột chuẩn hóa.`
- Purpose: Retrieve trade data matched by price level for the given symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` | Hiển thị log debug. |

#### Source details

##### Source `cafef`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `matched_by_price` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `kbs`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.trading.Trading`
- Provider method: `matched_by_price`

```text
symbol, underlying_symbol, time, exchange, ceiling_price, floor_price, reference_price, open_price, high_price, low_price, close_price, basis, open_interest, total_trades, total_value, price_change, percent_change, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, foreign_buy_volume, foreign_sell_volume, last_trading_date
```
- Note: Derived from `app.lib.vnstock_data_alt.explorer.kbs.trading._DERIVATIVE_STANDARD_COLUMNS`.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:04.044045+00:00`
- Success: `True`
- Row count: `8`

```text
price, buyVol, sellVol, unknownVol, totalVol
```
- Dtypes: `{'price': 'int64', 'buyVol': 'int64', 'sellVol': 'int64', 'unknownVol': 'int64', 'totalVol': 'int64'}`

```json
[
  {
    "price": 59400,
    "buyVol": 0,
    "sellVol": 2400,
    "unknownVol": 0,
    "totalVol": 2400
  },
  {
    "price": 59500,
    "buyVol": 55400,
    "sellVol": 2000,
    "unknownVol": 64300,
    "totalVol": 121700
  },
  {
    "price": 59600,
    "buyVol": 20700,
    "sellVol": 150300,
    "unknownVol": 0,
    "totalVol": 171000
  }
]
```

##### Source `vci`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `matched_by_price` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve trade data matched by price level for the given symbol.

### odd_lot

- Kind: `method`
- Signature: `(symbols_list = None, exchange = 'HOSE', show_log = False) -> DataFrame chứa dữ liệu giao dịch lô lẻ với các cột chuẩn hóa.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa dữ liệu giao dịch lô lẻ với các cột chuẩn hóa.`
- Purpose: Retrieve odd-lot (lô lẻ) trading data.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `symbols_list` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `['VCB']` |  | Danh sách mã chứng khoán. Nếu None, truy xuất toàn bộ sàn. |
| `exchange` | `POSITIONAL_OR_KEYWORD` | `False` | `HOSE` | `` | `HOSE` | `HOSE`, `HNX`, `UPCOM`, `HOSE` | Sàn giao dịch ('HOSE', 'HNX', 'UPCOM'). Mặc định 'HOSE'. |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` |  | Hiển thị log debug. |

#### Source details

##### Source `cafef`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `odd_lot` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `kbs`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.trading.Trading`
- Provider method: `odd_lot`

```text
symbol, underlying_symbol, time, exchange, ceiling_price, floor_price, reference_price, open_price, high_price, low_price, close_price, basis, open_interest, total_trades, total_value, price_change, percent_change, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, foreign_buy_volume, foreign_sell_volume, last_trading_date
```
- Note: Derived from `app.lib.vnstock_data_alt.explorer.kbs.trading._DERIVATIVE_STANDARD_COLUMNS`.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:04.188508+00:00`
- Success: `True`
- Row count: `0`

##### Source `vci`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.trading.Trading`
- Provider method: `odd_lot`

```text
symbol, price, volume, highest, lowest, open, avg_price, accumulated_volume, accumulated_value, session, time, exchange
```
- Note: Derived from `app.lib.vnstock_data_alt.explorer.vci.trading._ODD_LOT_STANDARD_COLUMNS`.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:04.453093+00:00`
- Success: `True`
- Row count: `0`

#### Notes / caveats

Retrieve odd-lot (lô lẻ) trading data.

### order_stats

- Kind: `method`
- Signature: `(start, end, page = 1, limit = None)`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `cafef`
- Purpose: Retrieve order statistics for the given symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` | `2025-03-07` |
| `page` | `POSITIONAL_OR_KEYWORD` | `False` | `1` | `` | `1` |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `5` |

#### Source details

##### Source `cafef`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.cafef.trading.Trading`
- Provider method: `order_stats`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:04.799679+00:00`
- Success: `True`
- Row count: `5`

```text
buy_orders, sell_orders, buy_volume, sell_volume, volume_diff, avg_buy_order_volume, avg_sell_order_volume
```
- Dtypes: `{'buy_orders': 'int64', 'sell_orders': 'int64', 'buy_volume': 'int64', 'sell_volume': 'int64', 'volume_diff': 'str', 'avg_buy_order_volume': 'int64', 'avg_sell_order_volume': 'int64'}`

```json
[
  {
    "buy_orders": 3438,
    "sell_orders": 3404,
    "buy_volume": 7961700,
    "sell_volume": 6151500,
    "volume_diff": "1.810.200",
    "avg_buy_order_volume": 2316,
    "avg_sell_order_volume": 1807
  },
  {
    "buy_orders": 2733,
    "sell_orders": 2924,
    "buy_volume": 3875900,
    "sell_volume": 4743500,
    "volume_diff": "-867.600",
    "avg_buy_order_volume": 1418,
    "avg_sell_order_volume": 1622
  },
  {
    "buy_orders": 2053,
    "sell_orders": 1741,
    "buy_volume": 3057400,
    "sell_volume": 3680100,
    "volume_diff": "-622.700",
    "avg_buy_order_volume": 1489,
    "avg_sell_order_volume": 2114
  }
]
```

##### Source `kbs`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `order_stats` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `order_stats` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve order statistics for the given symbol.

### price_board

- Kind: `method`
- Signature: `(symbols_list: List[str], board: str = 'stock', exchange: str = 'HOSE', show_log: bool = False, get_all: bool = False)`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Purpose: Retrieve the price board (order book) for a list of symbols.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `symbols_list` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `List[str]` | `['VCB', 'TCB']` | `ACB`, `VNM` | List of symbols (e.g., ['ACB', 'VNM']). |
| `board` | `POSITIONAL_OR_KEYWORD` | `False` | `stock` | `str` | `omitted; default 'stock'` | `stock`, `odd_lot`, `put_through`, `derivatives` | Board type ('stock', 'odd_lot', 'put_through', 'derivatives'). |
| `exchange` | `POSITIONAL_OR_KEYWORD` | `False` | `HOSE` | `str` | `HOSE` | `HOSE`, `HNX`, `UPCOM` | Exchange ('HOSE', 'HNX', 'UPCOM'). |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `bool` | `False` |  | Display debug logs. |
| `get_all` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `bool` | `True` |  | If True, return all raw columns. Otherwise, standard columns. |

#### Source details

##### Source `cafef`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `price_board` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `kbs`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.trading.Trading`
- Provider method: `price_board`

```text
symbol, time, exchange, ceiling_price, floor_price, reference_price, open_price, high_price, low_price, close_price, average_price, volume_accumulated, total_value, price_change, percent_change, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, foreign_buy_volume, foreign_sell_volume, foreign_room
```
- Note: Derived from `app.lib.vnstock_data_alt.explorer.kbs.trading._PRICE_BOARD_STANDARD_COLUMNS`.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:05.020188+00:00`
- Success: `True`
- Row count: `2`

```text
volume_accumulated, high_price, total_value, low_price, listed_shares, percent_change, bid_vol_1, bid_vol_2, bid_vol_3, bid_price_1, average_price, bid_price_2, bid_price_3, reference_price, exchange, foreign_buy_volume, foreign_buy_count, ask_price_1, ask_price_2, ask_price_3, floor_price, foreign_ownership_ratio, foreign_room, put_through_qty, foreign_sell_volume, symbol, put_through_value, total_listed_qty, ST, open_price, price_change, ceiling_price, close_price, time, ask_vol_1, ask_vol_2, ask_vol_3
```
- Dtypes: `{'volume_accumulated': 'int64', 'high_price': 'int64', 'total_value': 'int64', 'low_price': 'int64', 'listed_shares': 'str', 'percent_change': 'float64', 'bid_vol_1': 'int64', 'bid_vol_2': 'int64', 'bid_vol_3': 'int64', 'bid_price_1': 'str', 'average_price': 'int64', 'bid_price_2': 'int64', 'bid_price_3': 'int64', 'reference_price': 'int64', 'exchange': 'str', 'foreign_buy_volume': 'int64', 'foreign_buy_count': 'str', 'ask_price_1': 'str', 'ask_price_2': 'int64', 'ask_price_3': 'int64', 'floor_price': 'int64', 'foreign_ownership_ratio': 'int64', 'foreign_room': 'int64', 'put_through_qty': 'int64', 'foreign_sell_volume': 'int64', 'symbol': 'str', 'put_through_value': 'int64', 'total_listed_qty': 'str', 'ST': 'str', 'open_price': 'int64', 'price_change': 'int64', 'ceiling_price': 'int64', 'close_price': 'int64', 'time': 'int64', 'ask_vol_1': 'int64', 'ask_vol_2': 'int64', 'ask_vol_3': 'int64'}`

```json
[
  {
    "volume_accumulated": 2900300,
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
    "foreign_room": 790700917,
    "put_through_qty": 0,
    "foreign_sell_volume": 253800,
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
    "volume_accumulated": 6384500,
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
    "foreign_room": 0,
    "put_through_qty": 2300016,
    "foreign_sell_volume": 0,
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

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.trading.Trading`
- Provider method: `price_board`

```text
symbol, time, exchange, ceiling_price, floor_price, reference_price, open_price, high_price, low_price, close_price, average_price, total_trades, total_value, price_change, percent_change, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, foreign_buy_volume, foreign_sell_volume
```
- Note: Derived from `app.lib.vnstock_data_alt.explorer.vci.trading._PRICE_BOARD_STANDARD_COLUMNS`.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:05.113742+00:00`
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
    "('listing', 'exchange')": "HOSE",
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
    "('listing', 'exchange')": "HOSE",
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

### price_history

- Kind: `method`
- Signature: `(start, end, page = 1, limit = None)`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `cafef`
- Purpose: Retrieve the price history for a list of symbols.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` | `2025-03-07` |
| `page` | `POSITIONAL_OR_KEYWORD` | `False` | `1` | `` | `1` |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `5` |

#### Source details

##### Source `cafef`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.cafef.trading.Trading`
- Provider method: `price_history`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:05.409525+00:00`
- Success: `True`
- Row count: `5`

```text
open, high, low, close, adjusted_price, change_pct, matched_volume, matched_value, deal_volume, deal_value
```
- Dtypes: `{'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'adjusted_price': 'float64', 'change_pct': 'float64', 'matched_volume': 'int64', 'matched_value': 'int64', 'deal_volume': 'int64', 'deal_value': 'int64'}`

```json
[
  {
    "open": 93.6,
    "high": 95.3,
    "low": 93.5,
    "close": 95.1,
    "adjusted_price": 53.47,
    "change_pct": NaN,
    "matched_volume": 3999500,
    "matched_value": 376981000000,
    "deal_volume": 0,
    "deal_value": 0
  },
  {
    "open": 93.5,
    "high": 94.2,
    "low": 93.2,
    "close": 93.5,
    "adjusted_price": 52.57,
    "change_pct": NaN,
    "matched_volume": 2357600,
    "matched_value": 220606000000,
    "deal_volume": 0,
    "deal_value": 0
  },
  {
    "open": 93.0,
    "high": 93.8,
    "low": 93.0,
    "close": 93.0,
    "adjusted_price": 52.29,
    "change_pct": NaN,
    "matched_volume": 1820200,
    "matched_value": 169769000000,
    "deal_volume": 0,
    "deal_value": 0
  }
]
```

##### Source `kbs`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.trading.Trading`
- Provider method: `price_history`

```text
symbol, underlying_symbol, time, exchange, ceiling_price, floor_price, reference_price, open_price, high_price, low_price, close_price, basis, open_interest, total_trades, total_value, price_change, percent_change, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, foreign_buy_volume, foreign_sell_volume, last_trading_date
```
- Note: Derived from `app.lib.vnstock_data_alt.explorer.kbs.trading._DERIVATIVE_STANDARD_COLUMNS`.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:05.724913+00:00`
- Success: `True`
- Row count: `1000`

```text
timestamp, trading_date, symbol, time, side, price, volume, match_volume, accumulated_volume, accumulated_value
```
- Dtypes: `{'timestamp': 'datetime64[s]', 'trading_date': 'str', 'symbol': 'str', 'time': 'str', 'side': 'str', 'price': 'int64', 'volume': 'int64', 'match_volume': 'int64', 'accumulated_volume': 'int64', 'accumulated_value': 'int64'}`

```json
[
  {
    "timestamp": "NaT",
    "trading_date": "17/03/2026",
    "symbol": "VCB",
    "time": "11:29:58",
    "side": "B",
    "price": 60000,
    "volume": 1200,
    "match_volume": 200,
    "accumulated_volume": 2900300,
    "accumulated_value": 173411560000
  },
  {
    "timestamp": "NaT",
    "trading_date": "17/03/2026",
    "symbol": "VCB",
    "time": "11:29:51",
    "side": "B",
    "price": 60000,
    "volume": 1200,
    "match_volume": 100,
    "accumulated_volume": 2900100,
    "accumulated_value": 173399560000
  },
  {
    "timestamp": "NaT",
    "trading_date": "17/03/2026",
    "symbol": "VCB",
    "time": "11:29:49",
    "side": "B",
    "price": 60000,
    "volume": 1200,
    "match_volume": 2000,
    "accumulated_volume": 2900000,
    "accumulated_value": 173393560000
  }
]
```

##### Source `vci`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.trading.Trading`
- Provider method: `price_history`

```text
symbol, price, volume, highest, lowest, open, avg_price, accumulated_volume, accumulated_value, session, time, exchange
```
- Note: Derived from `app.lib.vnstock_data_alt.explorer.vci.trading._ODD_LOT_STANDARD_COLUMNS`.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:06.030741+00:00`
- Success: `True`
- Row count: `5`

```text
trading_date, end_trading_date, market_cap, total_shares, ceiling_price, floor_price, reference_price, open, close, match_price, price_change, percent_price_change, high, low, average_price, matched_volume, matched_value, deal_volume, deal_value, total_volume, total_value, total_buy_trade, total_buy_trade_volume, total_sell_trade, total_sell_trade_volume, reference_price_adjusted, open_price_adjusted, close_price_adjusted, price_change_adjusted, percent_price_change_adjusted, highest_price_adjusted, lowest_price_adjusted, price_change_value, total_buy_unmatched_volume, total_sell_unmatched_volume, average_buy_trade_volume, average_sell_trade_volume, total_net_trade_volume
```
- Dtypes: `{'trading_date': 'datetime64[us]', 'end_trading_date': 'str', 'market_cap': 'float64', 'total_shares': 'float64', 'ceiling_price': 'float64', 'floor_price': 'float64', 'reference_price': 'float64', 'open': 'float64', 'close': 'float64', 'match_price': 'float64', 'price_change': 'float64', 'percent_price_change': 'float64', 'high': 'float64', 'low': 'float64', 'average_price': 'float64', 'matched_volume': 'float64', 'matched_value': 'float64', 'deal_volume': 'float64', 'deal_value': 'float64', 'total_volume': 'float64', 'total_value': 'float64', 'total_buy_trade': 'float64', 'total_buy_trade_volume': 'float64', 'total_sell_trade': 'float64', 'total_sell_trade_volume': 'float64', 'reference_price_adjusted': 'float64', 'open_price_adjusted': 'float64', 'close_price_adjusted': 'float64', 'price_change_adjusted': 'float64', 'percent_price_change_adjusted': 'float64', 'highest_price_adjusted': 'float64', 'lowest_price_adjusted': 'float64', 'price_change_value': 'str', 'total_buy_unmatched_volume': 'float64', 'total_sell_unmatched_volume': 'float64', 'average_buy_trade_volume': 'float64', 'average_sell_trade_volume': 'float64', 'total_net_trade_volume': 'float64'}`

```json
[
  {
    "trading_date": "2025-03-07T00:00:00",
    "end_trading_date": "2025-03-07T00:00:00",
    "market_cap": 531522579016200.0,
    "total_shares": 5589091262.0,
    "ceiling_price": 100000.0,
    "floor_price": 87000.0,
    "reference_price": 93500.0,
    "open": 93600.0,
    "close": 95100.0,
    "match_price": 95100.0,
    "price_change": 1600.0,
    "percent_price_change": 0.0171123,
    "high": 95300.0,
    "low": 93500.0,
    "average_price": 94172.81934038871,
    "matched_volume": 4003619.0,
    "matched_value": 377369832800.0,
    "deal_volume": 83000.0,
    "deal_value": 7478600000.0,
    "total_volume": 4086619.0,
    "total_value": 384848432800.0,
    "total_buy_trade": 4045.0,
    "total_buy_trade_volume": 7978107.0,
    "total_sell_trade": 3617.0,
    "total_sell_trade_volume": 6156778.0,
    "reference_price_adjusted": 62091.85,
    "open_price_adjusted": 62158.25,
    "close_price_adjusted": 63154.38,
    "price_change_adjusted": 1062.53,
    "percent_price_change_adjusted": 0.0171123,
    "highest_price_adjusted": 63287.2,
    "lowest_price_adjusted": 62091.85,
    "price_change_value": "+1.063 (+1.7%)",
    "total_buy_unmatched_volume": 3974488.0,
    "total_sell_unmatched_volume": 2153159.0,
    "average_buy_trade_volume": 1972.3379480841,
    "average_sell_trade_volume": 1702.1780481062,
    "total_net_trade_volume": 1821329.0
  },
  {
    "trading_date": "2025-03-06T00:00:00",
    "end_trading_date": "2025-03-06T00:00:00",
    "market_cap": 522580032997000.0,
    "total_shares": 5589091262.0,
    "ceiling_price": 99500.0,
    "floor_price": 86500.0,
    "reference_price": 93000.0,
    "open": 93500.0,
    "close": 93500.0,
    "match_price": 93500.0,
    "price_change": 500.0,
    "percent_price_change": 0.00537634,
    "high": 94200.0,
    "low": 93200.0,
    "average_price": 93543.2621303033,
    "matched_volume": 2359429.0,
    "matched_value": 220778154900.0,
    "deal_volume": 101000.0,
    "deal_value": 9378400000.0,
    "total_volume": 2460429.0,
    "total_value": 230156554900.0,
    "total_buy_trade": 3274.0,
    "total_buy_trade_volume": 3888134.0,
    "total_sell_trade": 3044.0,
    "total_sell_trade_volume": 4746316.0,
    "reference_price_adjusted": 61759.8,
    "open_price_adjusted": 62091.85,
    "close_price_adjusted": 62091.85,
    "price_change_adjusted": 332.04,
    "percent_price_change_adjusted": 0.00537634,
    "highest_price_adjusted": 62556.71,
    "lowest_price_adjusted": 61892.62,
    "price_change_value": "+332 (+0.5%)",
    "total_buy_unmatched_volume": 1528705.0,
    "total_sell_unmatched_volume": 2386887.0,
    "average_buy_trade_volume": 1187.5791081246,
    "average_sell_trade_volume": 1559.2365308804,
    "total_net_trade_volume": -858182.0
  },
  {
    "trading_date": "2025-03-05T00:00:00",
    "end_trading_date": "2025-03-05T00:00:00",
    "market_cap": 519785487366000.0,
    "total_shares": 5589091262.0,
    "ceiling_price": 99500.0,
    "floor_price": 86500.0,
    "reference_price": 93000.0,
    "open": 93000.0,
    "close": 93000.0,
    "match_price": 93000.0,
    "price_change": 0.0,
    "percent_price_change": 0.0,
    "high": 93800.0,
    "low": 93000.0,
    "average_price": 93219.19151747697,
    "matched_volume": 1822854.0,
    "matched_value": 170017685200.0,
    "deal_volume": 63000.0,
    "deal_value": 5780100000.0,
    "total_volume": 1885854.0,
    "total_value": 175797785200.0,
    "total_buy_trade": 2545.0,
    "total_buy_trade_volume": 3067625.0,
    "total_sell_trade": 1881.0,
    "total_sell_trade_volume": 3684273.0,
    "reference_price_adjusted": 61759.8,
    "open_price_adjusted": 61759.8,
    "close_price_adjusted": 61759.8,
    "price_change_adjusted": 0.0,
    "percent_price_change_adjusted": 0.0,
    "highest_price_adjusted": 62291.07,
    "lowest_price_adjusted": 61759.8,
    "price_change_value": "0 (0.0%)",
    "total_buy_unmatched_volume": 1244771.0,
    "total_sell_unmatched_volume": 1861419.0,
    "average_buy_trade_volume": 1205.3536345776,
    "average_sell_trade_volume": 1958.677830941,
    "total_net_trade_volume": -616648.0
  }
]
```

#### Notes / caveats

Retrieve the price history for a list of symbols.

### prop_trade

- Kind: `method`
- Signature: `(start, end, page = 1, limit = None)`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `cafef`
- Purpose: Retrieve property trade data for the given symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` | `2025-03-07` |
| `page` | `POSITIONAL_OR_KEYWORD` | `False` | `1` | `` | `1` |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `5` |

#### Source details

##### Source `cafef`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.cafef.trading.Trading`
- Provider method: `prop_trade`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:06.272573+00:00`
- Success: `True`
- Row count: `5`

```text
prop_buy_volume, prop_sell_volume, prop_buy_value, prop_sell_value
```
- Dtypes: `{'prop_buy_volume': 'int64', 'prop_sell_volume': 'int64', 'prop_buy_value': 'int64', 'prop_sell_value': 'int64'}`

```json
[
  {
    "prop_buy_volume": 12400,
    "prop_sell_volume": 451000,
    "prop_buy_value": 1161800000,
    "prop_sell_value": 42577450000
  },
  {
    "prop_buy_volume": 75400,
    "prop_sell_volume": 142600,
    "prop_buy_value": 7051140000,
    "prop_sell_value": 13342800000
  },
  {
    "prop_buy_volume": 88800,
    "prop_sell_volume": 98300,
    "prop_buy_value": 8268070000,
    "prop_sell_value": 9165240000
  }
]
```

##### Source `kbs`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `prop_trade` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.trading.Trading`
- Provider method: `prop_trade`

```text
symbol, price, volume, highest, lowest, open, avg_price, accumulated_volume, accumulated_value, session, time, exchange
```
- Note: Derived from `app.lib.vnstock_data_alt.explorer.vci.trading._ODD_LOT_STANDARD_COLUMNS`.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:06.555802+00:00`
- Success: `True`
- Row count: `5`

```text
trading_date, from_date, end_trading_date, total_buy_trade_volume, percent_buy_trade_volume, total_buy_trade_value, percent_buy_trade_value, total_sell_trade_volume, percent_sell_trade_volume, total_sell_trade_value, percent_sell_trade_value, total_trade_net_volume, total_trade_net_value, total_match_buy_trade_volume, total_match_buy_trade_value, total_match_sell_trade_volume, total_match_sell_trade_value, total_match_trade_net_volume, total_match_trade_net_value, total_deal_buy_trade_volume, total_deal_buy_trade_value, total_deal_sell_trade_volume, total_deal_sell_trade_value, total_deal_trade_net_volume, total_deal_trade_net_value, update_date, total_volume, total_value
```
- Dtypes: `{'trading_date': 'datetime64[us]', 'from_date': 'object', 'end_trading_date': 'object', 'total_buy_trade_volume': 'float64', 'percent_buy_trade_volume': 'float64', 'total_buy_trade_value': 'float64', 'percent_buy_trade_value': 'float64', 'total_sell_trade_volume': 'float64', 'percent_sell_trade_volume': 'float64', 'total_sell_trade_value': 'float64', 'percent_sell_trade_value': 'float64', 'total_trade_net_volume': 'float64', 'total_trade_net_value': 'float64', 'total_match_buy_trade_volume': 'float64', 'total_match_buy_trade_value': 'float64', 'total_match_sell_trade_volume': 'float64', 'total_match_sell_trade_value': 'float64', 'total_match_trade_net_volume': 'float64', 'total_match_trade_net_value': 'float64', 'total_deal_buy_trade_volume': 'float64', 'total_deal_buy_trade_value': 'float64', 'total_deal_sell_trade_volume': 'float64', 'total_deal_sell_trade_value': 'float64', 'total_deal_trade_net_volume': 'float64', 'total_deal_trade_net_value': 'float64', 'update_date': 'str', 'total_volume': 'float64', 'total_value': 'float64'}`

```json
[
  {
    "trading_date": "2025-03-07T00:00:00",
    "from_date": null,
    "end_trading_date": null,
    "total_buy_trade_volume": 12400.0,
    "percent_buy_trade_volume": 0.0030342931,
    "total_buy_trade_value": 1161800000.0,
    "percent_buy_trade_value": 0.0030188508,
    "total_sell_trade_volume": 451000.0,
    "percent_sell_trade_volume": 0.1103601779,
    "total_sell_trade_value": 42577450000.0,
    "percent_sell_trade_value": 0.1106343339,
    "total_trade_net_volume": -438600.0,
    "total_trade_net_value": -41415650000.0,
    "total_match_buy_trade_volume": 12400.0,
    "total_match_buy_trade_value": 1161800000.0,
    "total_match_sell_trade_volume": 451000.0,
    "total_match_sell_trade_value": 42577450000.0,
    "total_match_trade_net_volume": -438600.0,
    "total_match_trade_net_value": -41415650000.0,
    "total_deal_buy_trade_volume": 0.0,
    "total_deal_buy_trade_value": 0.0,
    "total_deal_sell_trade_volume": 0.0,
    "total_deal_sell_trade_value": 0.0,
    "total_deal_trade_net_volume": 0.0,
    "total_deal_trade_net_value": 0.0,
    "update_date": "2025-03-07T16:50:01.337",
    "total_volume": 4086619.0,
    "total_value": 384848432800.0
  },
  {
    "trading_date": "2025-03-06T00:00:00",
    "from_date": null,
    "end_trading_date": null,
    "total_buy_trade_volume": 75400.0,
    "percent_buy_trade_volume": 0.0306450623,
    "total_buy_trade_value": 7051140000.0,
    "percent_buy_trade_value": 0.0306362771,
    "total_sell_trade_volume": 142600.0,
    "percent_sell_trade_volume": 0.0579573725,
    "total_sell_trade_value": 13342800000.0,
    "percent_sell_trade_value": 0.0579727134,
    "total_trade_net_volume": -67200.0,
    "total_trade_net_value": -6291660000.0,
    "total_match_buy_trade_volume": 75400.0,
    "total_match_buy_trade_value": 7051140000.0,
    "total_match_sell_trade_volume": 142600.0,
    "total_match_sell_trade_value": 13342800000.0,
    "total_match_trade_net_volume": -67200.0,
    "total_match_trade_net_value": -6291660000.0,
    "total_deal_buy_trade_volume": 0.0,
    "total_deal_buy_trade_value": 0.0,
    "total_deal_sell_trade_volume": 0.0,
    "total_deal_sell_trade_value": 0.0,
    "total_deal_trade_net_volume": 0.0,
    "total_deal_trade_net_value": 0.0,
    "update_date": "2025-03-06T16:53:45.43",
    "total_volume": 2460429.0,
    "total_value": 230156554900.0
  },
  {
    "trading_date": "2025-03-05T00:00:00",
    "from_date": null,
    "end_trading_date": null,
    "total_buy_trade_volume": 88800.0,
    "percent_buy_trade_volume": 0.0470874203,
    "total_buy_trade_value": 8268070000.0,
    "percent_buy_trade_value": 0.0470317074,
    "total_sell_trade_volume": 98300.0,
    "percent_sell_trade_volume": 0.0521249259,
    "total_sell_trade_value": 9165240000.0,
    "percent_sell_trade_value": 0.0521351278,
    "total_trade_net_volume": -9500.0,
    "total_trade_net_value": -897170000.0,
    "total_match_buy_trade_volume": 88800.0,
    "total_match_buy_trade_value": 8268070000.0,
    "total_match_sell_trade_volume": 98300.0,
    "total_match_sell_trade_value": 9165240000.0,
    "total_match_trade_net_volume": -9500.0,
    "total_match_trade_net_value": -897170000.0,
    "total_deal_buy_trade_volume": 0.0,
    "total_deal_buy_trade_value": 0.0,
    "total_deal_sell_trade_volume": 0.0,
    "total_deal_sell_trade_value": 0.0,
    "total_deal_trade_net_volume": 0.0,
    "total_deal_trade_net_value": 0.0,
    "update_date": "2025-03-05T16:58:01.15",
    "total_volume": 1885854.0,
    "total_value": 175797785200.0
  }
]
```

#### Notes / caveats

Retrieve property trade data for the given symbol.

### put_through

- Kind: `method`
- Signature: `(exchange = 'HOSE', symbol = None, page = 1, page_size = 1000, show_log = False) -> DataFrame chứa dữ liệu giao dịch thỏa thuận với các cột chuẩn hóa.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa dữ liệu giao dịch thỏa thuận với các cột chuẩn hóa.`
- Purpose: Retrieve put-through (thỏa thuận) trading data.

#### Parameters

| Name | Kind | Required | Default | Annotation | Example | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `exchange` | `POSITIONAL_OR_KEYWORD` | `False` | `HOSE` | `` |  | `HOSE` | `HOSE`, `HNX`, `UPCOM`, `HOSE` | Sàn giao dịch ('HOSE', 'HNX', 'UPCOM'). Mặc định 'HOSE'. |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |  | `VCB` |  | Mã chứng khoán để lọc (VD: 'ACB'). Nếu None, lấy toàn bộ sàn. |
| `page` | `POSITIONAL_OR_KEYWORD` | `False` | `1` | `` | `1` | `1` |  | Số trang. Mặc định 1. |
| `page_size` | `POSITIONAL_OR_KEYWORD` | `False` | `1000` | `` |  | `5` |  | Số lượng bản ghi mỗi trang. Mặc định 1000. |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` |  | `False` |  | Hiển thị log debug. |

#### Source details

##### Source `cafef`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `put_through` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `kbs`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.trading.Trading`
- Provider method: `put_through`

```text
symbol, time, exchange, match_price, match_volume, trading_date, reference_price, floor_price
```
- Note: Derived from `app.lib.vnstock_data_alt.explorer.kbs.trading._PUT_THROUGH_STANDARD_COLUMNS`.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:06.717157+00:00`
- Success: `True`
- Row count: `0`

```text
transaction_id, match_volume, match_price, counterparty_id, contract_number, floor_price, listed_shares, market_id, is_active, symbol, trading_date, reference_price, time, exchange, total_volume, total_value
```
- Dtypes: `{'transaction_id': 'str', 'match_volume': 'int64', 'match_price': 'str', 'counterparty_id': 'int64', 'contract_number': 'str', 'floor_price': 'int64', 'listed_shares': 'int64', 'market_id': 'int64', 'is_active': 'bool', 'symbol': 'str', 'trading_date': 'str', 'reference_price': 'int64', 'time': 'str', 'exchange': 'str', 'total_volume': 'int64', 'total_value': 'int64'}`

##### Source `vci`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.trading.Trading`
- Provider method: `put_through`

```text
symbol, time, exchange, match_price, match_volume, trading_date, reference_price, floor_price
```
- Note: Derived from `app.lib.vnstock_data_alt.explorer.vci.trading._PUT_THROUGH_STANDARD_COLUMNS`.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:06.896540+00:00`
- Success: `True`
- Row count: `160`

```text
symbol, time
```
- Dtypes: `{'symbol': 'str', 'time': 'datetime64[us, UTC]'}`

```json
[
  {
    "symbol": "SMC",
    "time": "2026-03-17T04:29:47.233000+00:00"
  },
  {
    "symbol": "LSS",
    "time": "2026-03-17T04:26:19.299000+00:00"
  },
  {
    "symbol": "TCB",
    "time": "2026-03-17T04:24:10.199000+00:00"
  }
]
```

#### Notes / caveats

Retrieve put-through (thỏa thuận) trading data.

### trade_history

- Kind: `method`
- Signature: `(page = 1, page_size = 1000, show_log = False) -> DataFrame chứa lịch sử giao dịch với các cột chuẩn hóa.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa lịch sử giao dịch với các cột chuẩn hóa.`
- Purpose: Retrieve trade history for the given symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation | Example | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `page` | `POSITIONAL_OR_KEYWORD` | `False` | `1` | `` |  | `1` | Số trang. Mặc định 1. |
| `page_size` | `POSITIONAL_OR_KEYWORD` | `False` | `1000` | `` | `100` | `5` | Số lượng bản ghi mỗi trang. Mặc định 1000. |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` |  | `False` | Hiển thị log debug. |

#### Source details

##### Source `cafef`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `trade_history` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `kbs`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.trading.Trading`
- Provider method: `trade_history`

```text
symbol, underlying_symbol, time, exchange, ceiling_price, floor_price, reference_price, open_price, high_price, low_price, close_price, basis, open_interest, total_trades, total_value, price_change, percent_change, bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, foreign_buy_volume, foreign_sell_volume, last_trading_date
```
- Note: Derived from `app.lib.vnstock_data_alt.explorer.kbs.trading._DERIVATIVE_STANDARD_COLUMNS`.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:07.116748+00:00`
- Success: `True`
- Row count: `5`

```text
timestamp, trading_date, symbol, time, side, price, volume, match_volume, accumulated_volume, accumulated_value
```
- Dtypes: `{'timestamp': 'datetime64[s]', 'trading_date': 'str', 'symbol': 'str', 'time': 'str', 'side': 'str', 'price': 'int64', 'volume': 'int64', 'match_volume': 'int64', 'accumulated_volume': 'int64', 'accumulated_value': 'int64'}`

```json
[
  {
    "timestamp": "NaT",
    "trading_date": "17/03/2026",
    "symbol": "VCB",
    "time": "11:29:58",
    "side": "B",
    "price": 60000,
    "volume": 1200,
    "match_volume": 200,
    "accumulated_volume": 2900300,
    "accumulated_value": 173411560000
  },
  {
    "timestamp": "NaT",
    "trading_date": "17/03/2026",
    "symbol": "VCB",
    "time": "11:29:51",
    "side": "B",
    "price": 60000,
    "volume": 1200,
    "match_volume": 100,
    "accumulated_volume": 2900100,
    "accumulated_value": 173399560000
  },
  {
    "timestamp": "NaT",
    "trading_date": "17/03/2026",
    "symbol": "VCB",
    "time": "11:29:49",
    "side": "B",
    "price": 60000,
    "volume": 1200,
    "match_volume": 2000,
    "accumulated_volume": 2900000,
    "accumulated_value": 173393560000
  }
]
```

##### Source `vci`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `trade_history` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve trade history for the given symbol.
