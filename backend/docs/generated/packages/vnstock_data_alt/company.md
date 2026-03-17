# Company

- Qualified name: `app.lib.vnstock_data_alt.api.company.Company`
- Signature: `(source='KBS', symbol=None, random_agent=False, show_log=False)`
- Supported sources: `kbs, vci`

Base adapter that uses ProviderRegistry to discover and instantiate

## Purpose

Base adapter that uses ProviderRegistry to discover and instantiate
providers from both explorer and connector packages.

## Members

### affiliate

- Kind: `method`
- Signature: `(show_log = False) -> DataFrame chứa thông tin công ty liên kết.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa thông tin công ty liên kết.`
- Purpose: Retrieve company affiliate data.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `affiliate`

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
- Provider: `app.lib.vnstock_data_alt.explorer.vci.company.Company`
- Provider method: `affiliate`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:47.985484+00:00`
- Success: `True`
- Row count: `4`

```text
id, sub_organ_code, organ_name, ownership_percent
```
- Dtypes: `{'id': 'str', 'sub_organ_code': 'str', 'organ_name': 'str', 'ownership_percent': 'float64'}`

```json
[
  {
    "id": "28550532",
    "sub_organ_code": "VCBB",
    "organ_name": "Công Ty Liên Doanh Trách Nhiệm Hữu Hạn Vietcombank-bonday-benthanh",
    "ownership_percent": 0.52
  },
  {
    "id": "28550534",
    "sub_organ_code": "VCBBONDAY",
    "organ_name": "Công ty Liên Doanh Hữu Hạn Vietcombank-bonday",
    "ownership_percent": 0.16
  },
  {
    "id": "28550533",
    "sub_organ_code": "VCBF",
    "organ_name": "Công ty TNHH Quản lý quỹ đầu tư chứng khoán Vietcombank",
    "ownership_percent": 0.51
  }
]
```

#### Notes / caveats

Retrieve company affiliate data.

### capital_history

- Kind: `method`
- Signature: `(show_log = False) -> DataFrame chứa lịch sử vốn điều lệ.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa lịch sử vốn điều lệ.`
- Purpose: Retrieve company charter capital history.

#### Parameters

| Name | Kind | Required | Default | Annotation | Description |
| --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `capital_history`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `capital_history` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve company charter capital history.

### events

- Kind: `method`
- Signature: `(event_type = None, page = 1, page_size = 10, show_log = False) -> DataFrame chứa danh sách sự kiện.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa danh sách sự kiện.`
- Purpose: Retrieve company events.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `event_type` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` | Loại sự kiện (1-5). None để lấy tất cả. |
| `page` | `POSITIONAL_OR_KEYWORD` | `False` | `1` | `` | `1` | Số trang. Mặc định 1. |
| `page_size` | `POSITIONAL_OR_KEYWORD` | `False` | `10` | `` | `5` | Số lượng bản ghi mỗi trang. Mặc định 10. |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `events`

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
- Provider: `app.lib.vnstock_data_alt.explorer.vci.company.Company`
- Provider method: `events`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:27:56.112560+00:00`
- Success: `True`
- Row count: `32`

```text
id, event_title, en__event_title, public_date, issue_date, source_url, event_list_code, ratio, value, record_date, exright_date, event_list_name, en__event_list_name
```
- Dtypes: `{'id': 'str', 'event_title': 'str', 'en__event_title': 'str', 'public_date': 'str', 'issue_date': 'str', 'source_url': 'str', 'event_list_code': 'str', 'ratio': 'float64', 'value': 'float64', 'record_date': 'str', 'exright_date': 'str', 'event_list_name': 'str', 'en__event_list_name': 'str'}`

```json
[
  {
    "id": "34191",
    "event_title": "VCB-Niêm yết bổ sung 719.276.804",
    "en__event_title": "VCB-Lists 719,276,804 additional shares ",
    "public_date": "2016-11-22",
    "issue_date": "1753-01-01",
    "source_url": "http://fiinpro.com/News/Detail/323795?lang=vi-VN",
    "event_list_code": "AIS",
    "ratio": NaN,
    "value": NaN,
    "record_date": "1753-01-01",
    "exright_date": "1753-01-01",
    "event_list_name": "Niêm yết thêm",
    "en__event_list_name": "Additional Listing"
  },
  {
    "id": "27101879",
    "event_title": "VCB - Niêm yết bổ sung 856,574,691 cổ phiếu",
    "en__event_title": "VCB- Lists additional 856,574,691 shares",
    "public_date": "2023-08-31",
    "issue_date": "2023-08-30",
    "source_url": "http://fiinpro.com/News/Detail/10915684?lang=vi-vn",
    "event_list_code": "AIS",
    "ratio": 0.0,
    "value": 0.0,
    "record_date": "1753-01-01",
    "exright_date": "1753-01-01",
    "event_list_name": "Niêm yết thêm",
    "en__event_list_name": "Additional Listing"
  },
  {
    "id": "34209",
    "event_title": "VCB-Niêm yết bổ sung 213.471.437 cổ phiếu",
    "en__event_title": "VCB-Lists 213,471,437 additional shares",
    "public_date": "2016-11-22",
    "issue_date": "2016-11-30",
    "source_url": "http://fiinpro.com/News/Detail/323795?lang=vi-VN",
    "event_list_code": "AIS",
    "ratio": NaN,
    "value": NaN,
    "record_date": "1753-01-01",
    "exright_date": "1753-01-01",
    "event_list_name": "Niêm yết thêm",
    "en__event_list_name": "Additional Listing"
  }
]
```

#### Notes / caveats

Retrieve company events.

### insider_trading

- Kind: `method`
- Signature: `(page = 1, page_size = 10, show_log = False) -> DataFrame chứa thông tin giao dịch nội bộ.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa thông tin giao dịch nội bộ.`
- Purpose: Retrieve insider trading data for the given symbol.

#### Parameters

| Name | Kind | Required | Default | Annotation | Description |
| --- | --- | --- | --- | --- | --- |
| `page` | `POSITIONAL_OR_KEYWORD` | `False` | `1` | `` | Số trang. Mặc định 1. |
| `page_size` | `POSITIONAL_OR_KEYWORD` | `False` | `10` | `` | Số lượng bản ghi mỗi trang. Mặc định 10. |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `insider_trading`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `insider_trading` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

#### Notes / caveats

Retrieve insider trading data for the given symbol.

### news

- Kind: `method`
- Signature: `(page = 1, page_size = 10, show_log = False) -> DataFrame chứa danh sách tin tức.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa danh sách tin tức.`
- Purpose: Retrieve company news.

#### Parameters

| Name | Kind | Required | Default | Annotation | Example | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `page` | `POSITIONAL_OR_KEYWORD` | `False` | `1` | `` |  | `1` | Số trang. Mặc định 1. |
| `page_size` | `POSITIONAL_OR_KEYWORD` | `False` | `10` | `` | `20` | `5` | Số lượng bản ghi mỗi trang. Mặc định 10. |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` |  | `False` | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `news`

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
- Provider: `app.lib.vnstock_data_alt.explorer.vci.company.Company`
- Provider method: `news`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:04.269556+00:00`
- Success: `True`
- Row count: `10`

```text
id, news_title, news_sub_title, friendly_sub_title, news_image_url, news_source_link, created_at, public_date, updated_at, lang_code, news_id, news_short_content, news_full_content, close_price, ref_price, floor, ceiling, price_change_pct
```
- Dtypes: `{'id': 'str', 'news_title': 'str', 'news_sub_title': 'str', 'friendly_sub_title': 'str', 'news_image_url': 'str', 'news_source_link': 'str', 'created_at': 'object', 'public_date': 'int64', 'updated_at': 'object', 'lang_code': 'str', 'news_id': 'str', 'news_short_content': 'str', 'news_full_content': 'str', 'close_price': 'int64', 'ref_price': 'int64', 'floor': 'int64', 'ceiling': 'int64', 'price_change_pct': 'float64'}`

```json
[
  {
    "id": "9549222",
    "news_title": "VCB: Nghị quyết HĐQT về việc phê duyệt chủ trương thay đổi địa điểm trụ sở Chi nhánh Sở giao dịch",
    "news_sub_title": "",
    "friendly_sub_title": "",
    "news_image_url": "https://cdn.fiingroup.vn/medialib/127889/I/2024/11/25/16184745330570700_VCB.png",
    "news_source_link": "https://www.hsx.vn/vi/tin-tuc/vcb-nghi-quyet-hdqt-ve-viec-phe-duyet-chu-truong-thay-doi-dia-diem-tru-so-chi-nhanh-so-giao-dich/2442948",
    "created_at": null,
    "public_date": 1773079639000,
    "updated_at": null,
    "lang_code": "vi",
    "news_id": "11910535",
    "news_short_content": "Ngân hàng Thương mại Cổ phần Ngoại Thương Việt Nam thông báo Nghị quyết HĐQT về việc phê duyệt chủ trương thay đổi địa điểm trụ sở Chi nhánh Sở giao dịch như sau:",
    "news_full_content": "<p>Ngân hàng Thương mại Cổ phần Ngoại Thương Việt Nam thông báo Nghị quyết HĐQT về việc phê duyệt chủ trương thay đổi địa điểm trụ sở Chi nhánh Sở giao dịch như sau:</p><table width=\"100%\" style='text-align: left;border=0'><tr><td colspan='2'><hr /></td></tr><tr><td colspan='2'>Tài liệu đính kèm</td></tr><tr><td> </td><td><a href=\"https://cmsv5.fiingroup.vn/medialib/FG/2026/2026-03/2026-03-09/VCB/20260309--VCB--NQ-HDQT-phe-duyet-chu-truong-thay-doi-dia-diem-Chi-nhanh-VCB-So-Giao-Dich.pdf\" title=\"Tải về\" download>20260309--VCB--NQ-HDQT-phe-duyet-chu-truong-thay-doi-dia-diem-Chi-nhanh-VCB-So-Giao-Dich.pdf</a></td></tr></table>",
    "close_price": 57300,
    "ref_price": 61600,
    "floor": 57300,
    "ceiling": 65900,
    "price_change_pct": -0.06980519
  },
  {
    "id": "9516176",
    "news_title": "VCB: Thông báo về ngày đăng ký cuối cùng tổ chức ĐHĐCĐ thường niên năm 2026",
    "news_sub_title": "",
    "friendly_sub_title": "",
    "news_image_url": "https://cdn.fiingroup.vn/medialib/127889/I/2024/11/25/16184745330570700_VCB.png",
    "news_source_link": "https://www.hsx.vn/vi/tin-tuc/vcb-thong-bao-ve-ngay-dang-ky-cuoi-cung-to-chuc-dhdcd-thuong-nien-nam-2026/2442473",
    "created_at": null,
    "public_date": 1772816352000,
    "updated_at": null,
    "lang_code": "vi",
    "news_id": "11908164",
    "news_short_content": "Sở Giao dịch Chứng khoán TP.HCM thông báo về ngày đăng ký cuối cùng tổ chức ĐHĐCĐ thường niên năm 2026 của Ngân hàng Thương mại Cổ phần Ngoại Thương Việt Nam như sau:",
    "news_full_content": "<p>Sở Giao dịch Chứng khoán TP.HCM thông báo về ngày đăng ký cuối cùng tổ chức ĐHĐCĐ thường niên năm 2026 của Ngân hàng Thương mại Cổ phần Ngoại Thương Việt Nam như sau:</p><table width=\"100%\" style='text-align: left;border=0'><tr><td colspan='2'><hr /></td></tr><tr><td colspan='2'>Tài liệu đính kèm</td></tr><tr><td> </td><td><a href=\"https://cmsv5.fiingroup.vn/medialib/FG/2026/2026-03/2026-03-06/VCB/20260306--VCB--TB-ngay-DKCC-to-chuc-DHDCD-thuong-nien-2026.pdf\" title=\"Tải về\" download>20260306--VCB--TB-ngay-DKCC-to-chuc-DHDCD-thuong-nien-2026.pdf</a></td></tr></table>",
    "close_price": 61600,
    "ref_price": 62500,
    "floor": 58200,
    "ceiling": 66800,
    "price_change_pct": -0.0144
  },
  {
    "id": "9506985",
    "news_title": "VCB: Báo cáo kết quả giao dịch cổ phiếu của Người nội bộ Nguyễn Tuấn Anh",
    "news_sub_title": "",
    "friendly_sub_title": "",
    "news_image_url": "https://cdn.fiingroup.vn/medialib/127889/I/2024/11/25/16184745330570700_VCB.png",
    "news_source_link": "https://www.hsx.vn/vi/tin-tuc/vcb-bao-cao-ket-qua-giao-dich-co-phieu-cua-nguoi-noi-bo-nguyen-tuan-anh/2441877",
    "created_at": null,
    "public_date": 1772646800000,
    "updated_at": null,
    "lang_code": "vi",
    "news_id": "11905193",
    "news_short_content": "Nguyễn Tuấn Anh báo cáo kết quả giao dịch cổ phiếu của Người nội bộ Ngân hàng Thương mại Cổ phần Ngoại Thương Việt Nam như sau:",
    "news_full_content": "<p>Nguyễn Tuấn Anh báo cáo kết quả giao dịch cổ phiếu của Người nội bộ Ngân hàng Thương mại Cổ phần Ngoại Thương Việt Nam như sau:</p><table width=\"100%\" style='text-align: left;border=0'><tr><td colspan='2'><hr /></td></tr><tr><td colspan='2'>Tài liệu đính kèm</td></tr><tr><td> </td><td><a href=\"https://cmsv5.fiingroup.vn/medialib/FG/2026/2026-03/2026-03-04/VCB/20260304--VCB--Bao-cao-ket-qua-giao-dich-co-phieu-cua-nguoi-noi-bo--Nguyen-Tuan-Anh--Ban-CBTT.pdf\" title=\"Tải về\" download>20260304--VCB--Bao-cao-ket-qua-giao-dich-co-phieu-cua-nguoi-noi-bo--Nguyen-Tuan-Anh--Ban-CBTT.pdf</a></td></tr></table>",
    "close_price": 63000,
    "ref_price": 61600,
    "floor": 57300,
    "ceiling": 65900,
    "price_change_pct": 0.02272727
  }
]
```

#### Notes / caveats

Retrieve company news.

### officers

- Kind: `method`
- Signature: `(show_log = False) -> DataFrame chứa thông tin lãnh đạo.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa thông tin lãnh đạo.`
- Purpose: Retrieve company officers data.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `officers`

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
- Provider: `app.lib.vnstock_data_alt.explorer.vci.company.Company`
- Provider method: `officers`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:08.466821+00:00`
- Success: `True`
- Row count: `21`

```text
id, officer_name, officer_position, position_short_name, update_date, officer_own_percent, quantity
```
- Dtypes: `{'id': 'str', 'officer_name': 'str', 'officer_position': 'str', 'position_short_name': 'str', 'update_date': 'str', 'officer_own_percent': 'float64', 'quantity': 'int64'}`

```json
[
  {
    "id": "21",
    "officer_name": "Phùng Nguyễn Hải Yến",
    "officer_position": "Phụ trách Công bố thông tin/Phó Tổng Giám đốc",
    "position_short_name": "Phụ trách CBTT/Phó TGĐ",
    "update_date": "2026-02-24",
    "officer_own_percent": 6.3e-06,
    "quantity": 52339
  },
  {
    "id": "10",
    "officer_name": "Nguyễn Thanh Tùng",
    "officer_position": "Phó Tổng Giám đốc",
    "position_short_name": "Phó TGĐ",
    "update_date": "2026-02-02",
    "officer_own_percent": 2.7e-06,
    "quantity": 22324
  },
  {
    "id": "1",
    "officer_name": "Đào Minh Tuấn",
    "officer_position": "Phó Tổng Giám đốc",
    "position_short_name": "Phó TGĐ",
    "update_date": "2015-09-14",
    "officer_own_percent": 2e-06,
    "quantity": 5810
  }
]
```

#### Notes / caveats

Retrieve company officers data.
Supports kwargs like filter_by='working'|'resigned'|'all'.

### overview

- Kind: `method`
- Signature: `(show_log = False) -> DataFrame chứa thông tin tổng quan công ty.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa thông tin tổng quan công ty.`
- Purpose: Retrieve company overview data.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `overview`

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
- Provider: `app.lib.vnstock_data_alt.explorer.vci.company.Company`
- Provider method: `overview`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:12.644684+00:00`
- Success: `True`
- Row count: `1`

```text
symbol, id, issue_share, history, company_profile, icb_name3, icb_name2, icb_name4, financial_ratio_issue_share, charter_capital
```
- Dtypes: `{'symbol': 'str', 'id': 'str', 'issue_share': 'int64', 'history': 'str', 'company_profile': 'str', 'icb_name3': 'str', 'icb_name2': 'str', 'icb_name4': 'str', 'financial_ratio_issue_share': 'int64', 'charter_capital': 'int64'}`

```json
[
  {
    "symbol": "VCB",
    "id": "75836",
    "issue_share": 8355675094,
    "history": " - Ngày 30/10/1962: Ngân hàng Ngoại thương Việt Nam (Vietcombank) được thành lập có tiền thân là Cục Ngoại Hối trực thuộc Ngân Hàng Quốc Gia Việt Nam;  - Ngày 01/04/1963: Vietcombank chính thức đi vào hoạt động;  - Năm 1990: Vietcombank chuyển thành một NHTM nhà nước hoạt động đa năng;  - Năm 2007: Vietcombank và NHTMCP SeaBank ký kết Hợp đồng với đối tác Cardif thành lập Công ty TNHH Bảo hiểm Nhân thọ Vietcombank – Cardif (VCLI);  - Ngày 26/12/2007: Vietcombank phát hành đợt cổ phiếu đầu tiên ra công chúng.  - Năm 2008: Ngân hàng ngoại thương Việt Nam chính thức chuyển đổi thành Ngân Hàng TMCP Ngoại Thương Việt Nam;  - Năm 2009: Cổ phiếu của Vietcombank chính thức được niêm yết trên Sở Giao dịch Chứng khoán Thành phố Hồ Chí Minh (HOSE);  - Ngày 30/09/2011: Ngân hàng Mizuho (MHCB) đã chính thức trở thành nhà đầu tư chiến lược vào Vietcombank, nắm giữ 15% vốn điều lệ của Vietcombank;  - Ngày 16/01/2019: Tăng vốn điều lệ lên 37.088.774.480.000 đồng;  - Ngày 10/03/2022: Tăng vốn điều lệ lên 47.325.165.710.000 đồng;  - Ngày 05/10/2023: Tăng vốn điều lệ lên 55.890.912.620.000 đồng;  - Ngày 28/04/2025: Tăng vốn điều lệ lên 83.556.750.940.000 đồng do phát hành cổ phiếu trả cổ tức;",
    "company_profile": "Ngân hàng Thương mại Cổ phần Ngoại thương Việt Nam (Vietcombank) chính thức đi vào hoạt động ngày 01/04/1963. Là ngân hàng thương mại nhà nước đầu tiên được Chính phủ lựa chọn thực hiện thí điểm cổ phần hoá, Ngân hàng Ngoại thương Việt Nam chính thức hoạt động với tư cách là một Ngân hàng Thương mại Cổ phần từ ngày 02/06/2008 sau khi thực hiện thành công kế hoạch cổ phần hóa thông qua việc phát hành cổ phiếu lần đầu ra công chúng. Năm 2024, so với cùng kỳ, biên lãi thuần (NIM) ở mức 2.86%, giảm 0.15%. Tỷ lệ nợ xấu ở mức 0.96%, giảm 0.02%. Tỷ lệ bao phủ nợ xấu ở mức 223.31%, giảm 6.99%. Lợi nhuận sau thuế công ty mẹ có giá trị bằng 33,8 nghìn tỷ đồng, tăng 2.42%. Tỷ suất lợi nhuận trên vốn chủ sở hữu (ROE) ở mức 18.74%, giảm 3.25%. VCB chính thức niêm yết và giao dịch trên Sở Giao dịch Chứng khoán Thành phố Hồ Chí Minh từ năm 2009.",
    "icb_name3": "Ngân hàng",
    "icb_name2": "Ngân hàng",
    "icb_name4": "Ngân hàng",
    "financial_ratio_issue_share": 8355675094,
    "charter_capital": 83556750940000
  }
]
```

#### Notes / caveats

Retrieve company overview data.

### shareholders

- Kind: `method`
- Signature: `(show_log = False) -> DataFrame chứa thông tin cổ đông.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa thông tin cổ đông.`
- Purpose: Retrieve company shareholders data.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `shareholders`

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
- Provider: `app.lib.vnstock_data_alt.explorer.vci.company.Company`
- Provider method: `shareholders`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:16.985570+00:00`
- Success: `True`
- Row count: `47`

```text
id, share_holder, quantity, share_own_percent, update_date
```
- Dtypes: `{'id': 'str', 'share_holder': 'str', 'quantity': 'int64', 'share_own_percent': 'float64', 'update_date': 'str'}`

```json
[
  {
    "id": "100755807",
    "share_holder": "Ngân Hàng Nhà Nước Việt Nam",
    "quantity": 6250338579,
    "share_own_percent": 0.748,
    "update_date": "2026-02-02"
  },
  {
    "id": "100755752",
    "share_holder": "Mizuho Bank Limited",
    "quantity": 1253366534,
    "share_own_percent": 0.15,
    "update_date": "2025-11-21"
  },
  {
    "id": "100768368",
    "share_holder": "Quỹ Đầu tư Chính phủ Singapore (GIC)",
    "quantity": 84503639,
    "share_own_percent": 0.0101,
    "update_date": "2025-10-05"
  }
]
```

#### Notes / caveats

Retrieve company shareholders data.

### subsidiaries

- Kind: `method`
- Signature: `(show_log = False) -> DataFrame chứa thông tin công ty con và công ty liên kết.`
- Declared signature: `(*A, **B)`
- Effective signature source: provider `kbs`
- Return type: `DataFrame chứa thông tin công ty con và công ty liên kết.`
- Purpose: Retrieve company subsidiaries data.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `subsidiaries`

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
- Provider: `app.lib.vnstock_data_alt.explorer.vci.company.Company`
- Provider method: `subsidiaries`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:28:21.478130+00:00`
- Success: `True`
- Row count: `12`

```text
id, sub_organ_code, ownership_percent, organ_name, type
```
- Dtypes: `{'id': 'str', 'sub_organ_code': 'str', 'ownership_percent': 'float64', 'organ_name': 'str', 'type': 'str'}`

```json
[
  {
    "id": "28550524",
    "sub_organ_code": "2646966",
    "ownership_percent": 0.875,
    "organ_name": "Công ty Chuyển tiền Vietcombank",
    "type": "công ty con"
  },
  {
    "id": "28550525",
    "sub_organ_code": "TB",
    "ownership_percent": 1.0,
    "organ_name": "Ngân hàng Thương mại TNHH MTV Ngoại thương Công nghệ Số",
    "type": "công ty con"
  },
  {
    "id": "28550526",
    "sub_organ_code": "VCB198",
    "ownership_percent": 0.7,
    "organ_name": "Công ty TNHH Cao Ốc Vietcombank 198",
    "type": "công ty con"
  }
]
```

#### Notes / caveats

Retrieve company subsidiaries data.
Supports kwargs like filter_by='all'|'subsidiary'.
