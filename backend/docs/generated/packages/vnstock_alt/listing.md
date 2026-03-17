# Listing

- Qualified name: `app.lib.vnstock_alt.api.listing.Listing`
- Signature: `(source: str = 'kbs', random_agent: bool = False, show_log: bool = False)`
- Supported sources: `kbs, msn, vci`

Base adapter that uses ProviderRegistry to discover and instantiate

## Purpose

Base adapter that uses ProviderRegistry to discover and instantiate
providers from both explorer and connector packages.

## Members

### all_bonds

- Kind: `method`
- Signature: `(**kwargs: Any) -> Any`
- Effective signature source: provider `kbs`
- Return type: `Any`
- Purpose: Retrieve all bonds (group='BOND').

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `Optional[bool]` | `False` | Hiển thị log debug. Mặc định False. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.kbs.listing.Listing`
- Provider method: `all_bonds`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:09.945031+00:00`
- Success: `True`
- Row count: `None`

```json
"0     BAB123032\n1     BAB124015\n2     BAB124016\n3     BAB124024\n4     BAB124025\n        ...    \n78    VIC124003\n79    VIC124005\n80    VND125032\n81    VND125033\n82    VPI124001\nName: symbol, Length: 83, dtype: str"
```

##### Source `msn`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `all_bonds` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.vci.listing.Listing`
- Provider method: `all_bonds`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:10.219554+00:00`
- Success: `True`
- Row count: `None`

```json
"0     BAB123032\n1     BAB124015\n2     BAB124016\n3     BAB124024\n4     BAB124025\n        ...    \n78    VIC124003\n79    VIC124005\n80    VND125032\n81    VND125033\n82    VPI124001\nName: symbol, Length: 83, dtype: str"
```

#### Notes / caveats

Retrieve all bonds (group='BOND').

### all_covered_warrant

- Kind: `method`
- Signature: `(**kwargs: Any) -> Any`
- Effective signature source: provider `kbs`
- Return type: `Any`
- Purpose: Retrieve all covered warrants (group='CW').

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `Optional[bool]` | `False` | Hiển thị log debug. Mặc định False. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.kbs.listing.Listing`
- Provider method: `all_covered_warrant`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:10.361538+00:00`
- Success: `True`
- Row count: `None`

```json
"0      CACB2502\n1      CACB2510\n2      CACB2511\n3      CACB2514\n4      CACB2515\n         ...   \n298    CVRE2525\n299    CVRE2526\n300    CVRE2527\n301    CVRE2601\n302    CVRE2602\nName: symbol, Length: 303, dtype: str"
```

##### Source `msn`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `all_covered_warrant` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.vci.listing.Listing`
- Provider method: `all_covered_warrant`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:10.681052+00:00`
- Success: `True`
- Row count: `None`

```json
"0      CACB2502\n1      CACB2510\n2      CACB2511\n3      CACB2514\n4      CACB2515\n         ...   \n298    CVRE2525\n299    CVRE2526\n300    CVRE2527\n301    CVRE2601\n302    CVRE2602\nName: symbol, Length: 303, dtype: str"
```

#### Notes / caveats

Retrieve all covered warrants (group='CW').

### all_future_indices

- Kind: `method`
- Signature: `(**kwargs: Any) -> Any`
- Effective signature source: provider `kbs`
- Return type: `Any`
- Purpose: Retrieve all futures indices (group='FU_INDEX').

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `Optional[bool]` | `False` | Hiển thị log debug. Mặc định False. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.kbs.listing.Listing`
- Provider method: `all_future_indices`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:10.821642+00:00`
- Success: `True`
- Row count: `None`

```json
"0     41I1G3000\n1     41I1G4000\n2     41I1G6000\n3     41I1G9000\n4     41I2G3000\n5     41I2G4000\n6     41I2G6000\n7     41I2G9000\n8     41B5G6000\n9     41B5G9000\n10    41B5GC000\n11    41BAG3000\n12    41BAG6000\n13    41BAG9000\nName: symbol, dtype: str"
```

##### Source `msn`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `all_future_indices` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.vci.listing.Listing`
- Provider method: `all_future_indices`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:10.921729+00:00`
- Success: `True`
- Row count: `None`

```json
"0    41I1G3000\n1    41I1G4000\n2    41I1G6000\n3    41I1G9000\n4    41I2G3000\n5    41I2G4000\n6    41I2G6000\n7    41I2G9000\nName: symbol, dtype: str"
```

#### Notes / caveats

Retrieve all futures indices (group='FU_INDEX').

### all_government_bonds

- Kind: `method`
- Signature: `(**kwargs: Any) -> Any`
- Effective signature source: provider `kbs`
- Return type: `Any`
- Purpose: Retrieve all government bonds (group='FU_BOND').

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `Optional[bool]` | `False` |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.kbs.listing.Listing`
- Provider method: `all_government_bonds`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `msn`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `all_government_bonds` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.vci.listing.Listing`
- Provider method: `all_government_bonds`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:27.055598+00:00`
- Success: `True`
- Row count: `None`

```json
"0    41B5G6000\n1    41B5G9000\n2    41B5GC000\n3    41BAG3000\n4    41BAG6000\n5    41BAG9000\nName: symbol, dtype: str"
```

#### Notes / caveats

Retrieve all government bonds (group='FU_BOND').

### all_symbols

- Kind: `method`
- Signature: `(show_log: Optional[bool] = False) -> Any`
- Declared signature: `(*args: Any, **kwargs: Any) -> Any`
- Effective signature source: provider `kbs`
- Return type: `Any`
- Purpose: Retrieve all symbols (filtered to STOCK).

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `Optional[bool]` | `False` | Hiển thị log debug. Mặc định False. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.kbs.listing.Listing`
- Provider method: `all_symbols`

```text
symbol, organ_name
```
- Note: Derived from provider docstring column hints.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:27.305573+00:00`
- Success: `True`
- Row count: `1545`

```text
symbol, organ_name
```
- Dtypes: `{'symbol': 'str', 'organ_name': 'str'}`

```json
[
  {
    "symbol": "DPP",
    "organ_name": "CTCP Dược Đồng Nai"
  },
  {
    "symbol": "SDA",
    "organ_name": "CTCP Simco Sông Đà"
  },
  {
    "symbol": "CLH",
    "organ_name": "CTCP Xi măng La Hiên VVMI"
  }
]
```

##### Source `msn`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `all_symbols` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.vci.listing.Listing`
- Provider method: `all_symbols`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:26:37.678165+00:00`
- Success: `True`
- Row count: `1738`

```text
symbol, organ_name
```
- Dtypes: `{'symbol': 'str', 'organ_name': 'str'}`

```json
[
  {
    "symbol": "YTC",
    "organ_name": "Công ty Cổ phần Xuất nhập khẩu Y tế Thành phố Hồ Chí Minh"
  },
  {
    "symbol": "YEG",
    "organ_name": "Công ty Cổ phần Tập đoàn Yeah1"
  },
  {
    "symbol": "YBM",
    "organ_name": "Công ty Cổ phần Khoáng sản Công nghiệp Yên Bái"
  }
]
```

#### Notes / caveats

Retrieve all symbols (filtered to STOCK).

### industries_icb

- Kind: `method`
- Signature: `(show_log: Optional[bool] = False) -> Any`
- Declared signature: `(*args: Any, **kwargs: Any) -> Any`
- Effective signature source: provider `kbs`
- Return type: `Any`
- Purpose: Retrieve ICB code hierarchy and mapping.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `Optional[bool]` | `omitted; default False` |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.kbs.listing.Listing`
- Provider method: `industries_icb`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `msn`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `industries_icb` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_alt.explorer.vci.listing.Listing`
- Provider method: `industries_icb`

```text
icb_name, en_icb_name, icb_code, level
```
- Note: Derived from static analysis of provider DataFrame shaping logic.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:26:37.132089+00:00`
- Success: `True`
- Row count: `155`

```text
icb_name, en_icb_name, icb_code, level
```
- Dtypes: `{'icb_name': 'str', 'en_icb_name': 'str', 'icb_code': 'str', 'level': 'int64'}`

```json
[
  {
    "icb_name": "Sản xuất Dầu khí",
    "en_icb_name": "Oil & Gas Producers",
    "icb_code": "0530",
    "level": 3
  },
  {
    "icb_name": "Thiết bị, Dịch vụ và Phân phối Dầu khí",
    "en_icb_name": "Oil Equipment, Services & Distribution",
    "icb_code": "0570",
    "level": 3
  },
  {
    "icb_name": "Hóa chất",
    "en_icb_name": "Chemicals",
    "icb_code": "1350",
    "level": 3
  }
]
```

#### Notes / caveats

Retrieve ICB code hierarchy and mapping.

### symbols_by_exchange

- Kind: `method`
- Signature: `(get_all: Optional[bool] = False, show_log: Optional[bool] = False) -> Any`
- Declared signature: `(*args: Any, **kwargs: Any) -> Any`
- Effective signature source: provider `kbs`
- Return type: `Any`
- Purpose: Retrieve symbols by exchange/board.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `get_all` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `Optional[bool]` | `True` | Lấy tất cả các cột mà API cung cấp thay vì chỉ các cột chuẩn hoá. Mặc định False. |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `Optional[bool]` | `False` | Hiển thị log debug. Mặc định False. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_alt.explorer.kbs.listing.Listing`
- Provider method: `symbols_by_exchange`

```text
symbol, organ_name, en_organ_name, exchange, type, id
```
- Note: Derived from static analysis of provider DataFrame shaping logic.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:31.573451+00:00`
- Success: `True`
- Row count: `1967`

```text
symbol, organ_name, en_organ_name, exchange, type, id, re, ceiling, floor
```
- Dtypes: `{'symbol': 'str', 'organ_name': 'str', 'en_organ_name': 'str', 'exchange': 'str', 'type': 'str', 'id': 'int64', 're': 'float64', 'ceiling': 'float64', 'floor': 'float64'}`

```json
[
  {
    "symbol": "TCB",
    "organ_name": "Ngân hàng TMCP Kỹ thương Việt Nam",
    "en_organ_name": "Vietnam Technological and Commercial Joint Stock Bank",
    "exchange": "HOSE",
    "type": "stock",
    "id": 1,
    "re": 30200.0,
    "ceiling": 32300.0,
    "floor": 28100.0
  },
  {
    "symbol": "VGI",
    "organ_name": "Tổng Công ty cổ phần Đầu tư Quốc tế Viettel",
    "en_organ_name": "Viettel Global Investment Joint Stock Company",
    "exchange": "UPCOM",
    "type": "stock",
    "id": 1,
    "re": 84300.0,
    "ceiling": 96900.0,
    "floor": 71700.0
  },
  {
    "symbol": "VKP",
    "organ_name": "CTCP Nhựa Tân Hóa",
    "en_organ_name": "Viky Plastic Joint Stock Company",
    "exchange": "UPCOM",
    "type": "stock",
    "id": 1,
    "re": 500.0,
    "ceiling": 600.0,
    "floor": 400.0
  }
]
```

##### Source `msn`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `symbols_by_exchange` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.vci.listing.Listing`
- Provider method: `symbols_by_exchange`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:31.879716+00:00`
- Success: `True`
- Row count: `3308`

```text
symbol, exchange, type, organ_short_name, organ_name, product_grp_id, icb_code2
```
- Dtypes: `{'symbol': 'str', 'exchange': 'str', 'type': 'str', 'organ_short_name': 'str', 'organ_name': 'str', 'product_grp_id': 'str', 'icb_code2': 'str'}`

```json
[
  {
    "symbol": "YTC",
    "exchange": "UPCOM",
    "type": "STOCK",
    "organ_short_name": "XNK Y tế TP.HCM",
    "organ_name": "Công ty Cổ phần Xuất nhập khẩu Y tế Thành phố Hồ Chí Minh",
    "product_grp_id": "UPX",
    "icb_code2": "4500"
  },
  {
    "symbol": "YEG",
    "exchange": "HSX",
    "type": "STOCK",
    "organ_short_name": "Tập đoàn Yeah1",
    "organ_name": "Công ty Cổ phần Tập đoàn Yeah1",
    "product_grp_id": "STO",
    "icb_code2": "5500"
  },
  {
    "symbol": "YBM",
    "exchange": "HSX",
    "type": "STOCK",
    "organ_short_name": "Khoáng sản CN Yên Bái",
    "organ_name": "Công ty Cổ phần Khoáng sản Công nghiệp Yên Bái",
    "product_grp_id": "STO",
    "icb_code2": "1700"
  }
]
```

#### Notes / caveats

Retrieve symbols by exchange/board.

### symbols_by_group

- Kind: `method`
- Signature: `(group: str = 'VN30', show_log: Optional[bool] = False) -> Any`
- Declared signature: `(*args: Any, **kwargs: Any) -> Any`
- Effective signature source: provider `kbs`
- Return type: `Any`
- Purpose: Retrieve symbols by predefined group (VN30, HNX30, CW, etc.).

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `group` | `POSITIONAL_OR_KEYWORD` | `False` | `VN30` | `str` | `VN30` | `VN30`, `VN30`, `VN100`, `HOSE`, `HNX`, `UPCOM`, `ETF`, `BOND`, `CW`, `FU_INDEX` | Tên nhóm được hỗ trợ. Mặc định 'VN30'. Ví dụ: 'VN30', 'VN100', 'HOSE', 'HNX', 'UPCOM', 'ETF', 'BOND', 'CW', 'FU_INDEX'. |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `Optional[bool]` | `False` |  | Hiển thị log debug. Mặc định False. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.kbs.listing.Listing`
- Provider method: `symbols_by_group`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:32.016604+00:00`
- Success: `True`
- Row count: `None`

```json
"0     ACB\n1     BID\n2     CTG\n3     DGC\n4     FPT\n5     GAS\n6     GVR\n7     HDB\n8     HPG\n9     LPB\n10    MBB\n11    MSN\n12    MWG\n13    PLX\n14    SAB\n15    SHB\n16    SSB\n17    SSI\n18    STB\n19    TCB\n20    TPB\n21    VCB\n22    VHM\n23    VIB\n24    VIC\n25    VJC\n26    VNM\n27    VPB\n28    VPL\n29    VRE\nName: symbol, dtype: str"
```

##### Source `msn`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `symbols_by_group` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.vci.listing.Listing`
- Provider method: `symbols_by_group`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:26:37.367582+00:00`
- Success: `True`
- Row count: `None`

```json
"0     ACB\n1     BID\n2     CTG\n3     DGC\n4     FPT\n5     GAS\n6     GVR\n7     HDB\n8     HPG\n9     LPB\n10    MBB\n11    MSN\n12    MWG\n13    PLX\n14    SAB\n15    SHB\n16    SSB\n17    SSI\n18    STB\n19    TCB\n20    TPB\n21    VCB\n22    VHM\n23    VIB\n24    VIC\n25    VJC\n26    VNM\n27    VPB\n28    VPL\n29    VRE\nName: symbol, dtype: str"
```

#### Notes / caveats

Retrieve symbols by predefined group (VN30, HNX30, CW, etc.).

### symbols_by_industries

- Kind: `method`
- Signature: `(lang: str = 'vi', show_log: Optional[bool] = False) -> Any`
- Declared signature: `(*args: Any, **kwargs: Any) -> Any`
- Effective signature source: provider `kbs`
- Return type: `Any`
- Purpose: Retrieve symbols grouped by ICB industries.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lang` | `POSITIONAL_OR_KEYWORD` | `False` | `vi` | `str` | `vi` | `vi`, `en`, `vi` | Ngôn ngữ ('vi' hoặc 'en'). Mặc định 'vi'. |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `Optional[bool]` | `False` |  | Hiển thị log debug. Mặc định False. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.kbs.listing.Listing`
- Provider method: `symbols_by_industries`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:37.113965+00:00`
- Success: `True`
- Row count: `698`

```text
symbol, industry_code, industry_name
```
- Dtypes: `{'symbol': 'str', 'industry_code': 'int64', 'industry_name': 'str'}`

```json
[
  {
    "symbol": "AGR",
    "industry_code": 5,
    "industry_name": "Chứng khoán"
  },
  {
    "symbol": "APG",
    "industry_code": 5,
    "industry_name": "Chứng khoán"
  },
  {
    "symbol": "APS",
    "industry_code": 5,
    "industry_name": "Chứng khoán"
  }
]
```

##### Source `msn`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `symbols_by_industries` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.vci.listing.Listing`
- Provider method: `symbols_by_industries`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:26:37.271876+00:00`
- Success: `True`
- Row count: `1550`

```text
symbol, organ_name, icb_name3, icb_name2, icb_name4, com_type_code, icb_code1, icb_code2, icb_code3, icb_code4
```
- Dtypes: `{'symbol': 'str', 'organ_name': 'str', 'icb_name3': 'str', 'icb_name2': 'str', 'icb_name4': 'str', 'com_type_code': 'str', 'icb_code1': 'str', 'icb_code2': 'str', 'icb_code3': 'str', 'icb_code4': 'str'}`

```json
[
  {
    "symbol": "VLS",
    "organ_name": "Công ty Cổ phần Sản xuất Thép Việt Long",
    "icb_name3": "Kim loại",
    "icb_name2": "Tài nguyên Cơ bản",
    "icb_name4": "Thép và sản phẩm thép",
    "com_type_code": "CT",
    "icb_code1": "1000",
    "icb_code2": "1700",
    "icb_code3": "1750",
    "icb_code4": "1757"
  },
  {
    "symbol": "BQP",
    "organ_name": "Công ty Cổ phần Nhựa chất lượng cao Bình Thuận",
    "icb_name3": "Hóa chất",
    "icb_name2": "Hóa chất",
    "icb_name4": "Nhựa, cao su & sợi",
    "com_type_code": "CT",
    "icb_code1": "1000",
    "icb_code2": "1300",
    "icb_code3": "1350",
    "icb_code4": "1353"
  },
  {
    "symbol": "RYG",
    "organ_name": "Công ty Cổ phần Sản xuất và Đầu tư Hoàng Gia",
    "icb_name3": "Xây dựng và Vật liệu",
    "icb_name2": "Xây dựng và Vật liệu",
    "icb_name4": "Vật liệu xây dựng & Nội thất",
    "com_type_code": "CT",
    "icb_code1": "2000",
    "icb_code2": "2300",
    "icb_code3": "2350",
    "icb_code4": "2353"
  }
]
```

#### Notes / caveats

Retrieve symbols grouped by ICB industries.
