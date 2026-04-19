# company.news

- Class: `CompanyReference`
- Method: `news`
- Signature: `(page = 1, page_size = 10, show_log = False) -> DataFrame chứa danh sách tin tức.`
- Return type: `DataFrame chứa danh sách tin tức.`
- Normalization mode: `contractual`
- Supported sources: `kbs`
- Declared signature: `()`
- Default route source: `kbs`
- Default provider: `company.Company.news`

Get company news.

## Purpose

Get company news.

## Parameters

| Name | Kind | Required | Default | Annotation | Example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `page` | `POSITIONAL_OR_KEYWORD` | `False` | `1` | `` |  | Số trang. Mặc định 1. |
| `page_size` | `POSITIONAL_OR_KEYWORD` | `False` | `10` | `` | `20` | Số lượng bản ghi mỗi trang. Mặc định 10. |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` |  | Hiển thị log debug. |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `news`

```text
head, article_id, title, publish_time, url
```
- Note: No source-specific raw mapping declared; using normalized columns as a best-effort approximation.

#### Normalized output schema

- Coverage: `declared`

```text
head, article_id, title, publish_time, url
```

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
