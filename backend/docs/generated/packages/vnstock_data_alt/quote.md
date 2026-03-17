# Quote

- Qualified name: `app.lib.vnstock_data_alt.api.quote.Quote`
- Signature: `(source='KBS', symbol='', random_agent=False, show_log=False)`
- Supported sources: `kbs, mas, msn, vci, vnd`

Base adapter that uses ProviderRegistry to discover and instantiate

## Purpose

Base adapter that uses ProviderRegistry to discover and instantiate
providers from both explorer and connector packages.

## Members

### history

- Kind: `method`
- Signature: `(start = None, end = None, interval = '1D', to_df = True, show_log = False, count_back = None, floating = 2, length = None, get_all = False) -> DataFrame hoặc JSON string chứa dữ liệu OHLCV.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame hoặc JSON string chứa dữ liệu OHLCV.`
- Purpose: Load historical OHLC data for the symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation | Example | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |  | `2025-03-01` |  | Ngày bắt đầu (YYYY-MM-DD hoặc DD-MM-YYYY). Bắt buộc nếu không có length hoặc count_back. |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `'2024-12-31'` | `2025-03-07` |  | Ngày kết thúc (YYYY-MM-DD hoặc DD-MM-YYYY). Mặc định None (lấy đến hiện tại). |
| `interval` | `POSITIONAL_OR_KEYWORD` | `False` | `1D` | `` | `'1D'` | `1D` |  | Khung thời gian trích xuất dữ liệu. Giá trị nhận: 1m, 5m, 15m, 30m, 1H, 1D, 1W, 1M. Mặc định "1D". |
| `to_df` | `POSITIONAL_OR_KEYWORD` | `False` | `True` | `` |  | `True` |  | Trả về DataFrame. Mặc định True. False để trả về JSON. |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` |  | `False` |  | Hiển thị log debug. |
| `count_back` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |  | `5` |  | Số lượng nến (bars) cần lấy. |
| `floating` | `POSITIONAL_OR_KEYWORD` | `False` | `2` | `` |  | `omitted; default 2` |  | Số chữ số thập phân cho giá. Mặc định 2. |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `'3M'` | `omitted in live probe` | `3M`, `150`, `100b` | Khoảng thời gian phân tích (vd: '3M', 150, '150'). Nhận giá trị chuỗi (vd 3M), số ngày (int/str), hoặc số bars (vd '100b'). |
| `get_all` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `True` | `True` |  | Lấy tất cả các cột từ API response. Mặc định False (chỉ lấy cột chuẩn hóa). |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.quote.Quote`
- Provider method: `history`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:40.721698+00:00`
- Success: `True`
- Row count: `5`

```text
time, open, high, low, close, volume
```
- Dtypes: `{'time': 'datetime64[ns]', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'int64'}`

```json
[
  {
    "time": "2025-03-03T07:00:00",
    "open": 62.09,
    "high": 62.09,
    "low": 61.76,
    "close": 61.96,
    "volume": 1251000
  },
  {
    "time": "2025-03-04T07:00:00",
    "open": 61.89,
    "high": 62.03,
    "low": 61.56,
    "close": 61.76,
    "volume": 2498900
  },
  {
    "time": "2025-03-05T07:00:00",
    "open": 61.76,
    "high": 62.29,
    "low": 61.76,
    "close": 61.76,
    "volume": 1820200
  }
]
```

##### Source `mas`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.mas.quote.Quote`
- Provider method: `history`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:41.432462+00:00`
- Success: `True`
- Row count: `5`

```text
time, open, high, low, close, volume
```
- Dtypes: `{'time': 'datetime64[ns]', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'int64'}`

```json
[
  {
    "time": "2025-03-03T00:00:00",
    "open": 62.1,
    "high": 62.1,
    "low": 61.76,
    "close": 61.96,
    "volume": 1251000
  },
  {
    "time": "2025-03-04T00:00:00",
    "open": 61.9,
    "high": 62.03,
    "low": 61.56,
    "close": 61.76,
    "volume": 2498900
  },
  {
    "time": "2025-03-05T00:00:00",
    "open": 61.76,
    "high": 62.29,
    "low": 61.76,
    "close": 61.76,
    "volume": 1820200
  }
]
```

##### Source `msn`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.msn.quote.Quote`
- Provider method: `history`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.quote.Quote`
- Provider method: `history`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:41.622325+00:00`
- Success: `True`
- Row count: `5`

```text
time, open, high, low, close, volume
```
- Dtypes: `{'time': 'datetime64[ns]', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'int64'}`

```json
[
  {
    "time": "2025-03-03T00:00:00",
    "open": 62.09,
    "high": 62.09,
    "low": 61.76,
    "close": 61.96,
    "volume": 1252949
  },
  {
    "time": "2025-03-04T00:00:00",
    "open": 61.89,
    "high": 62.03,
    "low": 61.56,
    "close": 61.76,
    "volume": 2501823
  },
  {
    "time": "2025-03-05T00:00:00",
    "open": 61.76,
    "high": 62.29,
    "low": 61.76,
    "close": 61.76,
    "volume": 1822854
  }
]
```

##### Source `vnd`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vnd.quote.Quote`
- Provider method: `history`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:41.950252+00:00`
- Success: `True`
- Row count: `5`

```text
time, open, high, low, close, volume
```
- Dtypes: `{'time': 'datetime64[ns]', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'int64'}`

```json
[
  {
    "time": "2025-03-03T00:00:00",
    "open": 62.092,
    "high": 62.092,
    "low": 61.76,
    "close": 61.959,
    "volume": 1251000
  },
  {
    "time": "2025-03-04T00:00:00",
    "open": 61.893,
    "high": 62.025,
    "low": 61.561,
    "close": 61.76,
    "volume": 2498900
  },
  {
    "time": "2025-03-05T00:00:00",
    "open": 61.76,
    "high": 62.291,
    "low": 61.76,
    "close": 61.76,
    "volume": 1820200
  }
]
```

#### Notes / caveats

Load historical OHLC data for the symbol.

Forwards only supported kwargs to provider.history().

### intraday

- Kind: `method`
- Signature: `(page_size = 1000, page = 1, to_df = True, get_all = False, show_log = False, floating = 2) -> DataFrame hoặc JSON string chứa dữ liệu khớp lệnh intraday.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame hoặc JSON string chứa dữ liệu khớp lệnh intraday.`
- Purpose: Load intraday trade data for the symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation | Example | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `page_size` | `POSITIONAL_OR_KEYWORD` | `False` | `1000` | `` | `100` | `5` | Số lượng bản ghi trên mỗi trang (mặc định 1000). Thường 1 ngày có thể lên đến 100K dòng (VN30 derivatives) hoặc 50-70K (cổ phiếu cơ sở). |
| `page` | `POSITIONAL_OR_KEYWORD` | `False` | `1` | `` |  | `1` | Trang dữ liệu (mặc định 1). |
| `to_df` | `POSITIONAL_OR_KEYWORD` | `False` | `True` | `` |  | `True` | Trả về DataFrame. Mặc định True. False để trả về JSON. |
| `get_all` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` |  | `True` | Lấy tất cả các cột từ API response. Mặc định False (chỉ lấy cột chuẩn hóa). |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` |  | `False` | Hiển thị log debug. |
| `floating` | `POSITIONAL_OR_KEYWORD` | `False` | `2` | `` |  | `omitted; default 2` | Số chữ số thập phân cho giá. Mặc định 2. Nếu None sẽ không làm tròn. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.quote.Quote`
- Provider method: `intraday`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:42.582468+00:00`
- Success: `True`
- Row count: `5`

```text
time, price, volume, match_type, id, trading_date, symbol, price_change, accumulated_volume, accumulated_value
```
- Dtypes: `{'time': 'datetime64[us]', 'price': 'float64', 'volume': 'int64', 'match_type': 'object', 'id': 'str', 'trading_date': 'object', 'symbol': 'object', 'price_change': 'float64', 'accumulated_volume': 'int64', 'accumulated_value': 'float64'}`

```json
[
  {
    "time": "2026-03-17T11:29:58",
    "price": 60.0,
    "volume": 200,
    "match_type": "buy",
    "id": "2026-03-17_112958_600000_200",
    "trading_date": "17/03/2026",
    "symbol": "VCB",
    "price_change": 1200.0,
    "accumulated_volume": 2900300,
    "accumulated_value": 173411560000.0
  },
  {
    "time": "2026-03-17T11:29:51",
    "price": 60.0,
    "volume": 100,
    "match_type": "buy",
    "id": "2026-03-17_112951_600000_100",
    "trading_date": "17/03/2026",
    "symbol": "VCB",
    "price_change": 1200.0,
    "accumulated_volume": 2900100,
    "accumulated_value": 173399560000.0
  },
  {
    "time": "2026-03-17T11:29:49",
    "price": 60.0,
    "volume": 2000,
    "match_type": "buy",
    "id": "2026-03-17_112949_600000_2000",
    "trading_date": "17/03/2026",
    "symbol": "VCB",
    "price_change": 1200.0,
    "accumulated_volume": 2900000,
    "accumulated_value": 173393560000.0
  }
]
```

##### Source `mas`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.mas.quote.Quote`
- Provider method: `intraday`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:42.735473+00:00`
- Success: `True`
- Row count: `5`

```text
time, price, volume, value, match_type, high, low, change, change_pct, agg_volume
```
- Dtypes: `{'time': 'datetime64[ms, Asia/Ho_Chi_Minh]', 'price': 'float64', 'volume': 'int64', 'value': 'float64', 'match_type': 'str', 'high': 'float64', 'low': 'float64', 'change': 'float64', 'change_pct': 'float64', 'agg_volume': 'int64'}`

```json
[
  {
    "time": "2026-03-17T11:29:42+07:00",
    "price": 60.0,
    "volume": 1000,
    "value": 173262000000.0,
    "match_type": "Buy",
    "high": 60100.0,
    "low": 59400.0,
    "change": 1200.0,
    "change_pct": 0.0204,
    "agg_volume": 2897800
  },
  {
    "time": "2026-03-17T11:29:48+07:00",
    "price": 60.0,
    "volume": 200,
    "value": 173274000000.0,
    "match_type": "Buy",
    "high": 60100.0,
    "low": 59400.0,
    "change": 1200.0,
    "change_pct": 0.0204,
    "agg_volume": 2898000
  },
  {
    "time": "2026-03-17T11:29:49+07:00",
    "price": 60.0,
    "volume": 2000,
    "value": 173394000000.0,
    "match_type": "Buy",
    "high": 60100.0,
    "low": 59400.0,
    "change": 1200.0,
    "change_pct": 0.0204,
    "agg_volume": 2900000
  }
]
```

##### Source `msn`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `intraday` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.quote.Quote`
- Provider method: `intraday`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:43.029937+00:00`
- Success: `True`
- Row count: `5`

```text
time, price, volume, match_type, id
```
- Dtypes: `{'time': 'datetime64[s, Asia/Ho_Chi_Minh]', 'price': 'float64', 'volume': 'int64', 'match_type': 'str', 'id': 'str'}`

```json
[
  {
    "time": "2026-03-17T11:29:42+07:00",
    "price": 60.0,
    "volume": 1000,
    "match_type": "Buy",
    "id": "450617896"
  },
  {
    "time": "2026-03-17T11:29:48+07:00",
    "price": 60.0,
    "volume": 200,
    "match_type": "Buy",
    "id": "450618091"
  },
  {
    "time": "2026-03-17T11:29:49+07:00",
    "price": 60.0,
    "volume": 2000,
    "match_type": "Buy",
    "id": "450618131"
  }
]
```

##### Source `vnd`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vnd.quote.Quote`
- Provider method: `intraday`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:43.031405+00:00`
- Success: `True`
- Row count: `None`

#### Notes / caveats

Load intraday trade data for the symbol.

### price_depth

- Kind: `method`
- Signature: `(show_log = False) -> DataFrame chứa thông tin độ sâu giá.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa thông tin độ sâu giá.`
- Purpose: Load price depth (order book) data for the symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.quote.Quote`
- Provider method: `price_depth`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:43.433867+00:00`
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

##### Source `mas`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.mas.quote.Quote`
- Provider method: `price_depth`

```text
price, volume, buy_volume, sell_volume, undefined_volume
```
- Note: Derived from static analysis of provider DataFrame shaping logic.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:43.718551+00:00`
- Success: `True`
- Row count: `8`

```text
price, volume, buy_volume, sell_volume, undefined_volume
```
- Dtypes: `{'price': 'int64', 'volume': 'int64', 'buy_volume': 'int64', 'sell_volume': 'int64', 'undefined_volume': 'int64'}`

```json
[
  {
    "price": 59400,
    "volume": 2400,
    "buy_volume": 0,
    "sell_volume": 2400,
    "undefined_volume": 0
  },
  {
    "price": 59500,
    "volume": 121700,
    "buy_volume": 117300,
    "sell_volume": 4400,
    "undefined_volume": 0
  },
  {
    "price": 59600,
    "volume": 171000,
    "buy_volume": 17700,
    "sell_volume": 145200,
    "undefined_volume": 8100
  }
]
```

##### Source `msn`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `price_depth` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.quote.Quote`
- Provider method: `price_depth`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:43.813060+00:00`
- Success: `True`
- Row count: `8`

```text
price, volume, buy_volume, sell_volume, undefined_volume
```
- Dtypes: `{'price': 'str', 'volume': 'str', 'buy_volume': 'str', 'sell_volume': 'str', 'undefined_volume': 'str'}`

```json
[
  {
    "price": "60100.0",
    "volume": "15000.0",
    "buy_volume": "15000.0",
    "sell_volume": "0.0",
    "undefined_volume": "0.0"
  },
  {
    "price": "60000.0",
    "volume": "252200.0",
    "buy_volume": "189600.0",
    "sell_volume": "62600.0",
    "undefined_volume": "0.0"
  },
  {
    "price": "59900.0",
    "volume": "569800.0",
    "buy_volume": "519100.0",
    "sell_volume": "50700.0",
    "undefined_volume": "0.0"
  }
]
```

##### Source `vnd`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `price_depth` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Load price depth (order book) data for the symbol.
