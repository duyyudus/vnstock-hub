# company.capital_history

- Class: `CompanyReference`
- Method: `capital_history`
- Signature: `(show_log = False) -> DataFrame chứa lịch sử vốn điều lệ.`
- Return type: `DataFrame chứa lịch sử vốn điều lệ.`
- Normalization mode: `contractual`
- Supported sources: `kbs`
- Declared signature: `()`
- Default route source: `kbs`
- Default provider: `company.Company.capital_history`

Get company charter capital history.

## Purpose

Get company charter capital history.

## Parameters

| Name | Kind | Required | Default | Annotation | Description |
| --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | Hiển thị log debug. |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `capital_history`

```text
date, charter_capital, currency
```
- Note: No source-specific raw mapping declared; using normalized columns as a best-effort approximation.

#### Normalized output schema

- Coverage: `declared`

```text
date, charter_capital, currency
```

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
