# Fund

- Qualified name: `app.lib.vnstock_data_alt.explorer.fmarket.fund.Fund`
- Signature: `(random_agent=False)`

## Members

### asset_holding

- Kind: `method`
- Signature: `(fundId=23)`
- Purpose: Retrieve list of assets holding allocation for specific fundID. Live data is retrieved from the Fmarket API.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `fundId` | `POSITIONAL_OR_KEYWORD` | `False` | `23` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Retrieve list of assets holding allocation for specific fundID. Live data is retrieved from the Fmarket API.

### filter

- Kind: `method`
- Signature: `(symbol='')`
- Purpose: Truy xuất danh sách quỹ theo tên viết tắt (short_name) và mã id của quỹ. Mặc định là rỗng để liệt kê tất cả các quỹ.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `False` | `` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Truy xuất danh sách quỹ theo tên viết tắt (short_name) và mã id của quỹ. Mặc định là rỗng để liệt kê tất cả các quỹ.

### industry_holding

- Kind: `method`
- Signature: `(fundId=23)`
- Purpose: Retrieve list of industries and fund distribution for specific fundID. Live data is retrieved from the Fmarket API.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `fundId` | `POSITIONAL_OR_KEYWORD` | `False` | `23` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Retrieve list of industries and fund distribution for specific fundID. Live data is retrieved from the Fmarket API.

### listing

- Kind: `method`
- Signature: `(fund_type='')`
- Purpose: Truy xuất danh sách tất cả các quỹ mở hiện có trên Fmarket thông qua API. Xem trực tiếp tại https://fmarket.vn

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `fund_type` | `POSITIONAL_OR_KEYWORD` | `False` | `` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Truy xuất danh sách tất cả các quỹ mở hiện có trên Fmarket thông qua API. Xem trực tiếp tại https://fmarket.vn

### nav_report

- Kind: `method`
- Signature: `(fundId=23)`
- Purpose: Retrieve all available daily NAV data point of the specified fund. Live data is retrieved from the Fmarket API.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `fundId` | `POSITIONAL_OR_KEYWORD` | `False` | `23` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Retrieve all available daily NAV data point of the specified fund. Live data is retrieved from the Fmarket API.

### top_holding

- Kind: `method`
- Signature: `(fundId=23)`
- Purpose: Retrieve list of top 10 holdings in the specified fund. Live data is retrieved from the Fmarket API.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `fundId` | `POSITIONAL_OR_KEYWORD` | `False` | `23` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Retrieve list of top 10 holdings in the specified fund. Live data is retrieved from the Fmarket API.
