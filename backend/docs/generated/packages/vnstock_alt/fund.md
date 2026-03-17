# Fund

- Qualified name: `app.lib.vnstock_alt.explorer.fmarket.fund.Fund`
- Signature: `(random_agent: bool = False) -> None`

## Members

### asset_holding

- Kind: `method`
- Signature: `(fundId: int = 23) -> pandas.DataFrame`
- Return type: `<class 'pandas.DataFrame'>`
- Purpose: Retrieve list of assets holding allocation for specific fundID. Live data is retrieved from the Fmarket API.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `fundId` | `POSITIONAL_OR_KEYWORD` | `False` | `23` | `int` | `11` |

#### Source details

_No source-specific output contract derived._

###### Live-observed sample

- Captured at: `2026-03-17T05:26:46.126963+00:00`
- Success: `True`
- Row count: `2`

```text
asset_percent, asset_type
```
- Dtypes: `{'asset_percent': 'float64', 'asset_type': 'str'}`

```json
[
  {
    "asset_percent": 96.67,
    "asset_type": "Cổ phiếu"
  },
  {
    "asset_percent": 3.33,
    "asset_type": "Tiền và tương đương tiền"
  }
]
```

#### Notes / caveats

Retrieve list of assets holding allocation for specific fundID. Live data is retrieved from the Fmarket API.

### filter

- Kind: `method`
- Signature: `(symbol: str = '') -> pandas.DataFrame`
- Return type: `<class 'pandas.DataFrame'>`
- Purpose: Truy xuất danh sách quỹ theo tên viết tắt (short_name) và mã id của quỹ. Mặc định là rỗng để liệt kê tất cả các quỹ.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `False` | `` | `str` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Truy xuất danh sách quỹ theo tên viết tắt (short_name) và mã id của quỹ. Mặc định là rỗng để liệt kê tất cả các quỹ.

### industry_holding

- Kind: `method`
- Signature: `(fundId: int = 23) -> pandas.DataFrame`
- Return type: `<class 'pandas.DataFrame'>`
- Purpose: Retrieve list of industries and fund distribution for specific fundID. Live data is retrieved from the Fmarket API.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `fundId` | `POSITIONAL_OR_KEYWORD` | `False` | `23` | `int` | `11` |

#### Source details

_No source-specific output contract derived._

###### Live-observed sample

- Captured at: `2026-03-17T05:26:45.906873+00:00`
- Success: `True`
- Row count: `14`

```text
industry, net_asset_percent
```
- Dtypes: `{'industry': 'str', 'net_asset_percent': 'float64'}`

```json
[
  {
    "industry": "Ngân hàng",
    "net_asset_percent": 34.59
  },
  {
    "industry": "Bất động sản",
    "net_asset_percent": 11.9
  },
  {
    "industry": "Bán lẻ",
    "net_asset_percent": 11.2
  }
]
```

#### Notes / caveats

Retrieve list of industries and fund distribution for specific fundID. Live data is retrieved from the Fmarket API.

### listing

- Kind: `method`
- Signature: `(fund_type: str = '') -> pandas.DataFrame`
- Return type: `<class 'pandas.DataFrame'>`
- Purpose: Truy xuất danh sách tất cả các quỹ mở hiện có trên Fmarket thông qua API. Xem trực tiếp tại https://fmarket.vn

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `fund_type` | `POSITIONAL_OR_KEYWORD` | `False` | `` | `str` | `omitted in live probe` |

#### Source details

_No source-specific output contract derived._

###### Live-observed sample

- Captured at: `2026-03-17T05:26:45.235502+00:00`
- Success: `True`
- Row count: `62`

```text
short_name, name, fund_type, fund_owner_name, management_fee, inception_date, nav, nav_change_previous, nav_change_last_year, nav_change_inception, nav_change_1m, nav_change_3m, nav_change_6m, nav_change_12m, nav_change_24m, nav_change_36m, nav_change_36m_annualized, nav_update_at, fund_id_fmarket, fund_code, vsd_fee_id
```
- Dtypes: `{'short_name': 'str', 'name': 'str', 'fund_type': 'str', 'fund_owner_name': 'str', 'management_fee': 'float64', 'inception_date': 'str', 'nav': 'float64', 'nav_change_previous': 'float64', 'nav_change_last_year': 'float64', 'nav_change_inception': 'float64', 'nav_change_1m': 'float64', 'nav_change_3m': 'float64', 'nav_change_6m': 'float64', 'nav_change_12m': 'float64', 'nav_change_24m': 'float64', 'nav_change_36m': 'float64', 'nav_change_36m_annualized': 'float64', 'nav_update_at': 'str', 'fund_id_fmarket': 'int64', 'fund_code': 'str', 'vsd_fee_id': 'str'}`

```json
[
  {
    "short_name": "DCDS",
    "name": "QUỸ ĐẦU TƯ CHỨNG KHOÁN NĂNG ĐỘNG DC",
    "fund_type": "Quỹ cổ phiếu",
    "fund_owner_name": "CÔNG TY CỔ PHẦN QUẢN LÝ QUỸ DRAGON CAPITAL VIỆT NAM",
    "management_fee": 1.95,
    "inception_date": "2004-05-19",
    "nav": 101906.47,
    "nav_change_previous": -1.12,
    "nav_change_last_year": -4.87,
    "nav_change_inception": 919.06,
    "nav_change_1m": -5.98,
    "nav_change_3m": 2.62,
    "nav_change_6m": -6.88,
    "nav_change_12m": 25.23,
    "nav_change_24m": 36.89,
    "nav_change_36m": 90.48,
    "nav_change_36m_annualized": 23.96,
    "nav_update_at": "2026-03-17",
    "fund_id_fmarket": 28,
    "fund_code": "VFMVF1",
    "vsd_fee_id": "VFMVF1N001"
  },
  {
    "short_name": "SSISCA",
    "name": "QUỸ ĐẦU TƯ LỢI THẾ CẠNH TRANH BỀN VỮNG SSI",
    "fund_type": "Quỹ cổ phiếu",
    "fund_owner_name": "CÔNG TY TNHH QUẢN LÝ QUỸ SSI",
    "management_fee": 1.75,
    "inception_date": "2014-09-25",
    "nav": 44898.19,
    "nav_change_previous": -0.75,
    "nav_change_last_year": -1.47,
    "nav_change_inception": 348.98,
    "nav_change_1m": -7.04,
    "nav_change_3m": 4.46,
    "nav_change_6m": -5.7,
    "nav_change_12m": 8.04,
    "nav_change_24m": 25.83,
    "nav_change_36m": 79.7,
    "nav_change_36m_annualized": 21.58,
    "nav_update_at": "2026-03-17",
    "fund_id_fmarket": 11,
    "fund_code": "SSISCA",
    "vsd_fee_id": "SSISCAN001"
  },
  {
    "short_name": "VCBF-MGF",
    "name": "QUỸ ĐẦU TƯ CỔ PHIẾU TĂNG TRƯỞNG VCBF",
    "fund_type": "Quỹ cổ phiếu",
    "fund_owner_name": "CÔNG TY TNHH QUẢN LÝ QUỸ ĐẦU TƯ CHỨNG KHOÁN VIETCOMBANK",
    "management_fee": 1.9,
    "inception_date": "2021-10-28",
    "nav": 14689.31,
    "nav_change_previous": -0.54,
    "nav_change_last_year": 2.57,
    "nav_change_inception": 46.89,
    "nav_change_1m": -5.26,
    "nav_change_3m": 7.87,
    "nav_change_6m": -3.33,
    "nav_change_12m": 6.27,
    "nav_change_24m": 21.47,
    "nav_change_36m": 77.48,
    "nav_change_36m_annualized": 21.07,
    "nav_update_at": "2026-03-17",
    "fund_id_fmarket": 46,
    "fund_code": "VCBFMGF",
    "vsd_fee_id": "VCBFMGFN001"
  }
]
```

#### Notes / caveats

Truy xuất danh sách tất cả các quỹ mở hiện có trên Fmarket thông qua API. Xem trực tiếp tại https://fmarket.vn

### nav_report

- Kind: `method`
- Signature: `(fundId: int = 23) -> pandas.DataFrame`
- Return type: `<class 'pandas.DataFrame'>`
- Purpose: Retrieve all available daily NAV data point of the specified fund. Live data is retrieved from the Fmarket API.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `fundId` | `POSITIONAL_OR_KEYWORD` | `False` | `23` | `int` | `11` |

#### Source details

_No source-specific output contract derived._

###### Live-observed sample

- Captured at: `2026-03-17T05:26:45.524608+00:00`
- Success: `True`
- Row count: `2087`

```text
date, nav_per_unit
```
- Dtypes: `{'date': 'str', 'nav_per_unit': 'float64'}`

```json
[
  {
    "date": "2014-09-26",
    "nav_per_unit": 10000.0
  },
  {
    "date": "2014-09-30",
    "nav_per_unit": 10403.38
  },
  {
    "date": "2014-10-01",
    "nav_per_unit": 10498.19
  }
]
```

#### Notes / caveats

Retrieve all available daily NAV data point of the specified fund. Live data is retrieved from the Fmarket API.

### top_holding

- Kind: `method`
- Signature: `(fundId: int = 23) -> pandas.DataFrame`
- Return type: `<class 'pandas.DataFrame'>`
- Purpose: Retrieve list of top 10 holdings in the specified fund. Live data is retrieved from the Fmarket API.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `fundId` | `POSITIONAL_OR_KEYWORD` | `False` | `23` | `int` | `11` |

#### Source details

_No source-specific output contract derived._

###### Live-observed sample

- Captured at: `2026-03-17T05:26:45.733947+00:00`
- Success: `True`
- Row count: `10`

```text
stock_code, industry, net_asset_percent, type_asset, update_at, fundId
```
- Dtypes: `{'stock_code': 'str', 'industry': 'str', 'net_asset_percent': 'float64', 'type_asset': 'str', 'update_at': 'str', 'fundId': 'int64'}`

```json
[
  {
    "stock_code": "MBB",
    "industry": "Ngân hàng",
    "net_asset_percent": 7.97,
    "type_asset": "STOCK",
    "update_at": "2026-03-06",
    "fundId": 11
  },
  {
    "stock_code": "CTG",
    "industry": "Ngân hàng",
    "net_asset_percent": 7.56,
    "type_asset": "STOCK",
    "update_at": "2026-03-06",
    "fundId": 11
  },
  {
    "stock_code": "MWG",
    "industry": "Bán lẻ",
    "net_asset_percent": 7.18,
    "type_asset": "STOCK",
    "update_at": "2026-03-06",
    "fundId": 11
  }
]
```

#### Notes / caveats

Retrieve list of top 10 holdings in the specified fund. Live data is retrieved from the Fmarket API.
