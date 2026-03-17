# Listing

- Qualified name: `app.lib.vnstock_data_alt.api.listing.Listing`
- Signature: `(source=None, random_agent=False, show_log=False)`
- Supported sources: `kbs, msn, vci, vnd`

Base adapter that uses ProviderRegistry to discover and instantiate

## Purpose

Base adapter that uses ProviderRegistry to discover and instantiate
providers from both explorer and connector packages.

## Members

### all_bonds

- Kind: `method`
- Signature: `(show_log = False) -> Series chứa mã trái phiếu.`
- Declared signature: `(**A)`
- Effective signature source: provider `kbs`
- Return type: `Series chứa mã trái phiếu.`
- Purpose: Retrieve all bonds (group='BOND').

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` | Hiển thị log debug. Mặc định False. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.listing.Listing`
- Provider method: `all_bonds`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:43.479584+00:00`
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
- Provider: `app.lib.vnstock_data_alt.explorer.vci.listing.Listing`
- Provider method: `all_bonds`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:43.671926+00:00`
- Success: `True`
- Row count: `None`

```json
"0     BAB123032\n1     BAB124015\n2     BAB124016\n3     BAB124024\n4     BAB124025\n        ...    \n78    VIC124003\n79    VIC124005\n80    VND125032\n81    VND125033\n82    VPI124001\nName: symbol, Length: 83, dtype: str"
```

##### Source `vnd`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `all_bonds` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve all bonds (group='BOND').

### all_covered_warrant

- Kind: `method`
- Signature: `(show_log = False) -> Series chứa mã chứng quyền.`
- Declared signature: `(**A)`
- Effective signature source: provider `kbs`
- Return type: `Series chứa mã chứng quyền.`
- Purpose: Retrieve all covered warrants (group='CW').

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` | Hiển thị log debug. Mặc định False. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.listing.Listing`
- Provider method: `all_covered_warrant`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:43.817247+00:00`
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
- Provider: `app.lib.vnstock_data_alt.explorer.vci.listing.Listing`
- Provider method: `all_covered_warrant`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:43.919871+00:00`
- Success: `True`
- Row count: `None`

```json
"0      CACB2502\n1      CACB2510\n2      CACB2511\n3      CACB2514\n4      CACB2515\n         ...   \n298    CVRE2525\n299    CVRE2526\n300    CVRE2527\n301    CVRE2601\n302    CVRE2602\nName: symbol, Length: 303, dtype: str"
```

##### Source `vnd`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `all_covered_warrant` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve all covered warrants (group='CW').

### all_etf

- Kind: `method`
- Signature: `(show_log = False) -> Series chứa mã ETF.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `Series chứa mã ETF.`
- Purpose: Retrieve all ETF (exchange-traded funds).

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` | Hiển thị log debug. Mặc định False. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.listing.Listing`
- Provider method: `all_etf`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:44.180642+00:00`
- Success: `True`
- Row count: `None`

```json
"0     E1VFVN30\n1     FUCTVGF3\n2     FUCTVGF4\n3     FUCTVGF5\n4     FUCVREIT\n5     FUEABVND\n6     FUEBFVND\n7     FUEDCMID\n8     FUEFCV50\n9     FUEIP100\n10    FUEKIV30\n11    FUEKIVFS\n12    FUEKIVND\n13    FUEMAV30\n14    FUEMAVND\n15    FUESSV30\n16    FUESSV50\n17    FUESSVFL\n18    FUETCC50\n19    FUETPVND\n20    FUEVFVND\n21    FUEVN100\nName: symbol, dtype: str"
```

##### Source `msn`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `all_etf` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `all_etf` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vnd`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `all_etf` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve all ETF (exchange-traded funds).

### all_future_indices

- Kind: `method`
- Signature: `(show_log = False) -> Series chứa mã phái sinh.`
- Declared signature: `(**A)`
- Effective signature source: provider `kbs`
- Return type: `Series chứa mã phái sinh.`
- Purpose: Retrieve all futures indices (group='FU_INDEX').

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` | Hiển thị log debug. Mặc định False. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.listing.Listing`
- Provider method: `all_future_indices`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:44.564717+00:00`
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
- Provider: `app.lib.vnstock_data_alt.explorer.vci.listing.Listing`
- Provider method: `all_future_indices`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:44.687445+00:00`
- Success: `True`
- Row count: `None`

```json
"0    41I1G3000\n1    41I1G4000\n2    41I1G6000\n3    41I1G9000\n4    41I2G3000\n5    41I2G4000\n6    41I2G6000\n7    41I2G9000\nName: symbol, dtype: str"
```

##### Source `vnd`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `all_future_indices` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve all futures indices (group='FU_INDEX').

### all_government_bonds

- Kind: `method`
- Signature: `(show_log = False)`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Purpose: Retrieve all government bonds.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.listing.Listing`
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
- Provider: `app.lib.vnstock_data_alt.explorer.vci.listing.Listing`
- Provider method: `all_government_bonds`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:48.816772+00:00`
- Success: `True`
- Row count: `None`

```json
"0    41B5G6000\n1    41B5G9000\n2    41B5GC000\n3    41BAG3000\n4    41BAG6000\n5    41BAG9000\nName: symbol, dtype: str"
```

##### Source `vnd`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `all_government_bonds` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve all government bonds.

### all_indices

- Kind: `method`
- Signature: `(show_log = False) -> DataFrame with columns`
- Declared signature: `(*B, **C)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame with columns`
- Purpose: Retrieve all standardized market indices with metadata.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.listing.Listing`
- Provider method: `all_indices`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:48.818943+00:00`
- Success: `True`
- Row count: `21`

```text
symbol, name, description, full_name, group, index_id, sector_id
```
- Dtypes: `{'symbol': 'str', 'name': 'str', 'description': 'str', 'full_name': 'str', 'group': 'str', 'index_id': 'int64', 'sector_id': 'float64'}`

```json
[
  {
    "symbol": "VN30",
    "name": "VN30",
    "description": "30 cổ phiếu vốn hóa lớn nhất & thanh khoản tốt nhất HOSE",
    "full_name": "VN30 Index",
    "group": "HOSE Indices",
    "index_id": 5,
    "sector_id": NaN
  },
  {
    "symbol": "VNMID",
    "name": "VNMID",
    "description": "Mid-Cap Index - nhóm cổ phiếu vốn hóa trung bình",
    "full_name": "VNMidCap Index",
    "group": "HOSE Indices",
    "index_id": 6,
    "sector_id": NaN
  },
  {
    "symbol": "VNSML",
    "name": "VNSML",
    "description": "Small-Cap Index - nhóm cổ phiếu vốn hóa nhỏ",
    "full_name": "VNSmallCap Index",
    "group": "HOSE Indices",
    "index_id": 7,
    "sector_id": NaN
  }
]
```

##### Source `msn`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `all_indices` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `all_indices` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vnd`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `all_indices` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve all standardized market indices with metadata.

    DataFrame with columns:
    - symbol: Index symbol (VN30, VNIT, etc.)
    - name: Index name
    - description: Vietnamese description
    - full_name: Full English name
    - group: Index group (HOSE Indices, Sector Indices, etc.)
    - index_id: Unique index ID
    - sector_id: ICB sector ID (for sector indices only)

### all_symbols

- Kind: `method`
- Signature: `(show_log = False) -> DataFrame với 2 cột: symbol, organ_name.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame với 2 cột: symbol, organ_name.`
- Purpose: Retrieve all symbols (filtered to STOCK).

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` | Hiển thị log debug. Mặc định False. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.listing.Listing`
- Provider method: `all_symbols`

```text
symbol, organ_name
```
- Note: Derived from provider docstring column hints.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:49.562807+00:00`
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
- Provider: `app.lib.vnstock_data_alt.explorer.vci.listing.Listing`
- Provider method: `all_symbols`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:49.940531+00:00`
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

##### Source `vnd`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vnd.listing.Listing`
- Provider method: `all_symbols`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:50.529144+00:00`
- Success: `True`
- Row count: `1980`

```text
symbol, type, exchange, isin, status, company_name, company_name_eng, short_name, short_name_eng, listed_date, company_id, fund_type, delisted_date, tax_code, index_code
```
- Dtypes: `{'symbol': 'str', 'type': 'str', 'exchange': 'str', 'isin': 'str', 'status': 'str', 'company_name': 'str', 'company_name_eng': 'str', 'short_name': 'str', 'short_name_eng': 'str', 'listed_date': 'str', 'company_id': 'str', 'fund_type': 'str', 'delisted_date': 'str', 'tax_code': 'str', 'index_code': 'str'}`

```json
[
  {
    "symbol": "DCDS",
    "type": "IFC",
    "exchange": "UPCOM",
    "isin": "VN000000DCV0",
    "status": "listed",
    "company_name": "Cổ phiếu CTCP Quản lý quỹ Đầu tư Dragon Capital Việt Nam",
    "company_name_eng": "Dragon Capital Vietfund Management Joint stock company",
    "short_name": "DCV",
    "short_name_eng": "DCV",
    "listed_date": "2026-01-19",
    "company_id": "88",
    "fund_type": "STOCK_FUND",
    "delisted_date": NaN,
    "tax_code": NaN,
    "index_code": NaN
  },
  {
    "symbol": "ENF",
    "type": "IFC",
    "exchange": "UPCOM",
    "isin": NaN,
    "status": "delisted",
    "company_name": "Quỹ Đầu tư Năng động Eastspring Investments Việt Nam",
    "company_name_eng": "Eastspring Investments Vietnam Navigator Fund",
    "short_name": "Quỹ đầu tư ENF",
    "short_name_eng": NaN,
    "listed_date": "2001-01-01",
    "company_id": "3903",
    "fund_type": "BALANCED_MUTUAL_FUND",
    "delisted_date": "2001-01-01",
    "tax_code": NaN,
    "index_code": NaN
  },
  {
    "symbol": "FUCTVGF5",
    "type": "IFC",
    "exchange": "HOSE",
    "isin": "VN0FUCTVGF54",
    "status": "listed",
    "company_name": "Quỹ đầu tư tăng trưởng Thiên Việt 5",
    "company_name_eng": "Thien Viet Growth Fund 5",
    "short_name": "Quỹ Thiên Việt 5",
    "short_name_eng": "TVGF5",
    "listed_date": "2023-12-29",
    "company_id": "15034",
    "fund_type": NaN,
    "delisted_date": NaN,
    "tax_code": NaN,
    "index_code": NaN
  }
]
```

#### Notes / caveats

Retrieve all symbols (filtered to STOCK).

### indices_by_group

- Kind: `method`
- Signature: `(group, show_log = False) -> DataFrame with index information filtered by group,`
- Declared signature: `(*B, **C)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame with index information filtered by group,`
- Purpose: Retrieve standardized market indices by group/category.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `group` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` | `VN30` | `HOSE Indices`, `Sector Indices` | Tên nhóm (VD: 'HOSE Indices', 'Sector Indices', etc.) |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` |  | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.listing.Listing`
- Provider method: `indices_by_group`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:50.531146+00:00`
- Success: `True`
- Row count: `None`

##### Source `msn`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `indices_by_group` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `indices_by_group` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vnd`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `indices_by_group` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve standardized market indices by group/category.

        enum values, and full names:
        - 'HOSE', IndexGroup.HOSE, or 'HOSE Indices' - HOSE market indices
        - 'SECTOR', IndexGroup.SECTOR, or 'Sector Indices' - Sector indices
        - 'INVESTMENT', IndexGroup.INVESTMENT, or 'Investment Indices' - Investment indices
        - 'VNX', IndexGroup.VNX, or 'VNX Indices' - VNX market indices

**Examples**
    >>> from app.lib.vnstock_data_alt.api.listing import Listing
    >>> from app.lib.vnstock_data_alt.enums import IndexGroup
    >>> lst = Listing()
    >>> # Using string short names
    >>> lst.indices_by_group('HOSE')
    >>> # Using enum values
    >>> lst.indices_by_group(IndexGroup.SECTOR)
    >>> # Using full names
    >>> lst.indices_by_group('HOSE Indices')

### industries_icb

- Kind: `method`
- Signature: `(show_log = False)`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Purpose: Retrieve ICB code hierarchy and mapping.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.listing.Listing`
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
- Provider: `app.lib.vnstock_data_alt.explorer.vci.listing.Listing`
- Provider method: `industries_icb`

```text
icb_name, en_icb_name, icb_code, level
```
- Note: Derived from static analysis of provider DataFrame shaping logic.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:54.704881+00:00`
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

##### Source `vnd`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `industries_icb` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve ICB code hierarchy and mapping.

### symbols_by_exchange

- Kind: `method`
- Signature: `(get_all = False, show_log = False) -> DataFrame chứa các cột từ API KBS: symbol, organ_name, en_organ_name,`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa các cột từ API KBS: symbol, organ_name, en_organ_name,`
- Purpose: Retrieve symbols by exchange/board.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `get_all` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `True` | Lấy tất cả các cột mà API cung cấp thay vì chỉ các cột chuẩn hoá. Mặc định False. |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` | Hiển thị log debug. Mặc định False. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.listing.Listing`
- Provider method: `symbols_by_exchange`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:55.030535+00:00`
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
- Provider: `app.lib.vnstock_data_alt.explorer.vci.listing.Listing`
- Provider method: `symbols_by_exchange`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:55.618311+00:00`
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

##### Source `vnd`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `symbols_by_exchange` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve symbols by exchange/board.

### symbols_by_group

- Kind: `method`
- Signature: `(group = 'VN30', show_log = False) -> Series chứa mã chứng khoán theo nhóm.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `Series chứa mã chứng khoán theo nhóm.`
- Purpose: Retrieve symbols by predefined group (VN30, HNX30, CW, etc.).

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `group` | `POSITIONAL_OR_KEYWORD` | `False` | `VN30` | `` | `VN30` | `VN30`, `VN30`, `VN100`, `HOSE`, `HNX`, `UPCOM`, `ETF`, `BOND`, `CW`, `FU_INDEX` | Tên nhóm được hỗ trợ. Mặc định 'VN30'. Ví dụ: 'VN30', 'VN100', 'HOSE', 'HNX', 'UPCOM', 'ETF', 'BOND', 'CW', 'FU_INDEX'. |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` |  | Hiển thị log debug. Mặc định False. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.listing.Listing`
- Provider method: `symbols_by_group`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:55.757092+00:00`
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
- Provider: `app.lib.vnstock_data_alt.explorer.vci.listing.Listing`
- Provider method: `symbols_by_group`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:55.901608+00:00`
- Success: `True`
- Row count: `None`

```json
"0     ACB\n1     BID\n2     CTG\n3     DGC\n4     FPT\n5     GAS\n6     GVR\n7     HDB\n8     HPG\n9     LPB\n10    MBB\n11    MSN\n12    MWG\n13    PLX\n14    SAB\n15    SHB\n16    SSB\n17    SSI\n18    STB\n19    TCB\n20    TPB\n21    VCB\n22    VHM\n23    VIB\n24    VIC\n25    VJC\n26    VNM\n27    VPB\n28    VPL\n29    VRE\nName: symbol, dtype: str"
```

##### Source `vnd`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `symbols_by_group` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve symbols by predefined group (VN30, HNX30, CW, etc.).

### symbols_by_industries

- Kind: `method`
- Signature: `(lang = 'vi', show_log = False) -> DataFrame chứa thông tin mã chứng khoán theo ngành.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa thông tin mã chứng khoán theo ngành.`
- Purpose: Retrieve symbols grouped by ICB industries.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lang` | `POSITIONAL_OR_KEYWORD` | `False` | `vi` | `` | `vi` | `vi`, `en`, `vi` | Ngôn ngữ ('vi' hoặc 'en'). Mặc định 'vi'. |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` |  | Hiển thị log debug. Mặc định False. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.listing.Listing`
- Provider method: `symbols_by_industries`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:01.406896+00:00`
- Success: `True`
- Row count: `625`

```text
symbol, industry_code, industry_name
```
- Dtypes: `{'symbol': 'object', 'industry_code': 'int64', 'industry_name': 'str'}`

```json
[
  {
    "symbol": {
      "name": "SX Phụ trợ",
      "code": 26,
      "change": -3.749553
    },
    "industry_code": 5,
    "industry_name": "Chứng khoán"
  },
  {
    "symbol": {
      "name": "Khai khoáng",
      "code": 10,
      "change": -0.434627
    },
    "industry_code": 5,
    "industry_name": "Chứng khoán"
  },
  {
    "symbol": {
      "name": "Dịch vụ lưu trú, ăn uống, giải trí",
      "code": 25,
      "change": -0.246161
    },
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
- Provider: `app.lib.vnstock_data_alt.explorer.vci.listing.Listing`
- Provider method: `symbols_by_industries`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:01.747137+00:00`
- Success: `True`
- Row count: `6200`

```text
symbol, organ_name, com_type_code, icb_level, icb_code, icb_name
```
- Dtypes: `{'symbol': 'str', 'organ_name': 'str', 'com_type_code': 'str', 'icb_level': 'int64', 'icb_code': 'str', 'icb_name': 'object'}`

```json
[
  {
    "symbol": "A32",
    "organ_name": "Công ty Cổ phần 32",
    "com_type_code": "CT",
    "icb_level": 1,
    "icb_code": "3000",
    "icb_name": null
  },
  {
    "symbol": "A32",
    "organ_name": "Công ty Cổ phần 32",
    "com_type_code": "CT",
    "icb_level": 2,
    "icb_code": "3700",
    "icb_name": "Hàng cá nhân & Gia dụng"
  },
  {
    "symbol": "A32",
    "organ_name": "Công ty Cổ phần 32",
    "com_type_code": "CT",
    "icb_level": 3,
    "icb_code": "3760",
    "icb_name": "Hàng cá nhân"
  }
]
```

##### Source `vnd`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `symbols_by_industries` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve symbols grouped by ICB industries.
