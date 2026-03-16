# Trading

- Qualified name: `app.lib.vnstock_alt.api.trading.Trading`
- Signature: `(source: str = 'kbs', symbol: str = None, random_agent: bool = False, show_log: bool = False)`
- Supported sources: `kbs, vci`

Base adapter that uses ProviderRegistry to discover and instantiate

## Purpose

Base adapter that uses ProviderRegistry to discover and instantiate
providers from both explorer and connector packages.

## Members

### foreign_trade

- Kind: `method`
- Signature: `(*args: Any, **kwargs: Any) -> Any`
- Return type: `Any`
- Purpose: Retrieve foreign trade data for the given symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `args` | `VAR_POSITIONAL` | `True` | `None` | `Any` |
| `kwargs` | `VAR_KEYWORD` | `True` | `None` | `Any` |

#### Source details

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

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `foreign_trade` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve foreign trade data for the given symbol.

### history

- Kind: `method`
- Signature: `(*args, **kwargs)`

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `args` | `VAR_POSITIONAL` | `True` | `None` | `` |
| `kwargs` | `VAR_KEYWORD` | `True` | `None` | `` |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `history` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `history` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

### insider_deal

- Kind: `method`
- Signature: `(*args: Any, **kwargs: Any) -> Any`
- Return type: `Any`
- Purpose: Retrieve insider deal data for the given symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `args` | `VAR_POSITIONAL` | `True` | `None` | `Any` |
| `kwargs` | `VAR_KEYWORD` | `True` | `None` | `Any` |

#### Source details

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

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `insider_deal` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve insider deal data for the given symbol.

### order_stats

- Kind: `method`
- Signature: `(*args: Any, **kwargs: Any) -> Any`
- Return type: `Any`
- Purpose: Retrieve order statistics for the given symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `args` | `VAR_POSITIONAL` | `True` | `None` | `Any` |
| `kwargs` | `VAR_KEYWORD` | `True` | `None` | `Any` |

#### Source details

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
- Signature: `(symbols_list: List[str], exchange: str = 'HOSE', show_log: Optional[bool] = False, get_all: bool = False) -> Any`
- Declared signature: `(*args: Any, **kwargs: Any) -> Any`
- Effective signature source: provider `kbs`
- Return type: `Any`
- Purpose: Retrieve the price board (order book) for a list of symbols.

#### Parameters

| Name | Kind | Required | Default | Annotation | Example | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `symbols_list` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `List[str]` |  | `VCB` | `ACB`, `VNM`, `HPG` | Danh sách mã chứng khoán (VD: ['ACB', 'VNM', 'HPG']). |
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

- Captured at: `2026-03-16T11:17:15.412655+00:00`
- Success: `True`
- Row count: `3`

```text
symbol
```
- Dtypes: `{'symbol': 'str'}`

```json
[
  {
    "symbol": "V"
  },
  {
    "symbol": "C"
  },
  {
    "symbol": "B"
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

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve the price board (order book) for a list of symbols.

### price_history

- Kind: `method`
- Signature: `(*args: Any, **kwargs: Any) -> Any`
- Return type: `Any`
- Purpose: Retrieve the price history for a list of symbols.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `args` | `VAR_POSITIONAL` | `True` | `None` | `Any` |
| `kwargs` | `VAR_KEYWORD` | `True` | `None` | `Any` |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `price_history` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `price_history` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve the price history for a list of symbols.

### prop_trade

- Kind: `method`
- Signature: `(*args: Any, **kwargs: Any) -> Any`
- Return type: `Any`
- Purpose: Retrieve property trade data for the given symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `args` | `VAR_POSITIONAL` | `True` | `None` | `Any` |
| `kwargs` | `VAR_KEYWORD` | `True` | `None` | `Any` |

#### Source details

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

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `prop_trade` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve property trade data for the given symbol.

### side_stats

- Kind: `method`
- Signature: `(*args: Any, **kwargs: Any) -> Any`
- Return type: `Any`
- Purpose: Retrieve bid/ask side statistics for the given symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `args` | `VAR_POSITIONAL` | `True` | `None` | `Any` |
| `kwargs` | `VAR_KEYWORD` | `True` | `None` | `Any` |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `side_stats` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `side_stats` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve bid/ask side statistics for the given symbol.

### trading_stats

- Kind: `method`
- Signature: `(*args: Any, **kwargs: Any) -> Any`
- Return type: `Any`
- Purpose: Retrieve trading statistics for the given symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `args` | `VAR_POSITIONAL` | `True` | `None` | `Any` |
| `kwargs` | `VAR_KEYWORD` | `True` | `None` | `Any` |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `trading_stats` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `trading_stats` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve trading statistics for the given symbol.
