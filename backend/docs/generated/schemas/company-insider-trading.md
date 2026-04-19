# company.insider_trading

- Class: `CompanyReference`
- Method: `insider_trading`
- Signature: `(page = 1, page_size = 10, show_log = False) -> DataFrame chứa thông tin giao dịch nội bộ.`
- Return type: `DataFrame chứa thông tin giao dịch nội bộ.`
- Normalization mode: `contractual`
- Supported sources: `kbs`
- Declared signature: `()`
- Default route source: `kbs`
- Default provider: `company.Company.insider_trading`

Get insider trading data.

## Purpose

Get insider trading data.

## Parameters

| Name | Kind | Required | Default | Annotation | Description |
| --- | --- | --- | --- | --- | --- |
| `page` | `POSITIONAL_OR_KEYWORD` | `False` | `1` | `` | Số trang. Mặc định 1. |
| `page_size` | `POSITIONAL_OR_KEYWORD` | `False` | `10` | `` | Số lượng bản ghi mỗi trang. Mặc định 10. |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | Hiển thị log debug. |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `not-available`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `insider_trading`

_No raw columns derived for this source._

#### Normalized output schema

- Coverage: `not-available`

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
