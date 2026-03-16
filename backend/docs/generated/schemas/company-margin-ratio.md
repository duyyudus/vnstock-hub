# company.margin_ratio

- Class: `CompanyReference`
- Method: `margin_ratio`
- Signature: `(show_log = False) -> DataFrame chứa thông tin tỷ lệ margin.`
- Return type: `DataFrame chứa thông tin tỷ lệ margin.`
- Normalization mode: `contractual`
- Supported sources: `kbs`
- Declared signature: `()`
- Default route source: `kbs`
- Default provider: `company.Company.margin_ratio`

Get margin lending ratio for the company across brokers.

## Purpose

Get margin lending ratio for the company across brokers.

## Parameters

| Name | Kind | Required | Default | Annotation | Description |
| --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | Hiển thị log debug. |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `margin_ratio`

```text
CompanyCode, Name, MarginRate, PrevMarginRate, ClosedDate, MarginPer
```

| Raw | Normalized |
| --- | --- |
| `CompanyCode` | `broker_code` |
| `Name` | `broker_name` |
| `MarginRate` | `margin_rate` |
| `PrevMarginRate` | `prev_margin_rate` |
| `ClosedDate` | `updated_at` |
| `MarginPer` | `margin_per` |

#### Normalized output schema

- Coverage: `declared`

```text
broker_code, broker_name, margin_rate, prev_margin_rate, margin_per, updated_at
```

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
