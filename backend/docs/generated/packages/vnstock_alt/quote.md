# Quote

- Qualified name: `app.lib.vnstock_alt.api.quote.Quote`
- Signature: `(source: str = <DataSource.KBS: 'kbs'>, symbol: str = '', random_agent: bool = False, show_log: bool = False)`
- Supported sources: `fmp, kbs, msn, vci`

Base adapter that uses ProviderRegistry to discover and instantiate

## Purpose

Base adapter that uses ProviderRegistry to discover and instantiate
providers from both explorer and connector packages.

## Members

### history

- Kind: `method`
- Signature: `(symbol: Optional[str] = None, start: str = None, end: str = None, interval: str = <TimeFrame.DAY_1: '1D'>, **kwargs: Any) -> pandas.DataFrame`
- Return type: `<class 'pandas.DataFrame'>`
- Purpose: Load historical OHLC data for the symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation | Example | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `Optional[str]` |  | `VCB` |  | Stock symbol. Mã chứng khoán. |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `str` | `"2024-01-01"` | `2025-03-01` |  | Start time in format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS. Thời gian bắt đầu định dạng YYYY-MM-DD hoặc YYYY-MM-DD HH:MM:SS. |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `str` | `"2024-04-18"` | `2025-03-07` |  | End time in format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS. Thời gian kết thúc định dạng YYYY-MM-DD hoặc YYYY-MM-DD HH:MM:SS. |
| `interval` | `POSITIONAL_OR_KEYWORD` | `False` | `TimeFrame.DAY_1` | `str` | `TimeResolutions.WEEKLY` | `1D` | `1m`, `5m`, `15m`, `30m`, `1H`, `D`, `1W`, `1M` | Data interval (1m, 5m, 15m, 30m, 1H, D, 1W, 1M). Khoảng thời gian dữ liệu (1m, 5m, 15m, 30m, 1H, D, 1W, 1M). |

#### Source details

##### Source `fmp`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.connector.fmp.quote.Quote`
- Provider method: `history`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `kbs`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_alt.explorer.kbs.quote.Quote`
- Provider method: `history`

```text
time, open, high, low, close, volume
```
- Note: Derived from static analysis of provider DataFrame shaping logic.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:09.409565+00:00`
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
- Provider: `app.lib.vnstock_alt.explorer.vci.quote.Quote`
- Provider method: `history`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:26:38.704574+00:00`
- Success: `True`
- Row count: `6`

```text
time, open, high, low, close, volume
```
- Dtypes: `{'time': 'datetime64[ns]', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'int64'}`

```json
[
  {
    "time": "2025-02-28T00:00:00",
    "open": 62.56,
    "high": 62.56,
    "low": 61.96,
    "close": 61.96,
    "volume": 2073877
  },
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
  }
]
```

#### Notes / caveats

Load historical OHLC data for the symbol.
Tải dữ liệu OHLC lịch sử cho mã chứng khoán.

**Examples**
    >>> quote = Quote(symbol="VCI", source="vci")
    >>> df = quote.history(start="2024-01-01", end="2024-04-18")
    >>> df = quote.history(symbol="FPT", start="2024-01-01", end="2024-04-18", interval=TimeResolutions.WEEKLY)
    >>> df = quote.history(symbol="FPT", start="2024-01-01 09:00:00", end="2024-01-01 14:30:00", interval=TimeResolutions.MINUTE_5)

### intraday

- Kind: `method`
- Signature: `(symbol: Optional[str] = None, page_size: int = 100, page: int = 1, **kwargs: Any) -> pandas.DataFrame`
- Return type: `<class 'pandas.DataFrame'>`
- Purpose: Load intraday trade data for the symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation | Example | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `Optional[str]` |  | `VCB` | Stock symbol. Mã chứng khoán. |
| `page_size` | `POSITIONAL_OR_KEYWORD` | `False` | `100` | `int` | `200` | `5` | Number of records to return. Số lượng bản ghi trả về. |
| `page` | `POSITIONAL_OR_KEYWORD` | `False` | `1` | `int` |  | `1` | Page number. Số trang. |

#### Source details

##### Source `fmp`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.connector.fmp.quote.Quote`
- Provider method: `intraday`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `kbs`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_alt.explorer.kbs.quote.Quote`
- Provider method: `intraday`

```text
time, price, volume, match_type, id
```
- Note: Derived from static analysis of provider DataFrame shaping logic.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:09.641254+00:00`
- Success: `True`
- Row count: `5`

```text
time, price, volume, match_type, id
```
- Dtypes: `{'time': 'datetime64[us]', 'price': 'float64', 'volume': 'int64', 'match_type': 'object', 'id': 'str'}`

```json
[
  {
    "time": "2026-03-17T11:29:58",
    "price": 60.0,
    "volume": 200,
    "match_type": "buy",
    "id": "2026-03-17_112958_600000_200"
  },
  {
    "time": "2026-03-17T11:29:51",
    "price": 60.0,
    "volume": 100,
    "match_type": "buy",
    "id": "2026-03-17_112951_600000_100"
  },
  {
    "time": "2026-03-17T11:29:49",
    "price": 60.0,
    "volume": 2000,
    "match_type": "buy",
    "id": "2026-03-17_112949_600000_2000"
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
- Provider: `app.lib.vnstock_alt.explorer.vci.quote.Quote`
- Provider method: `intraday`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:09.803580+00:00`
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

#### Notes / caveats

Load intraday trade data for the symbol.
Tải dữ liệu giao dịch trong ngày cho mã chứng khoán.

**Examples**
    >>> quote = Quote(symbol="VCI", source="vci")
    >>> df = quote.intraday()
    >>> df = quote.intraday(symbol="FPT", page_size=200)
