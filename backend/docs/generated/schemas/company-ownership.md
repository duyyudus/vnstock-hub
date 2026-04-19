# company.ownership

- Class: `CompanyReference`
- Method: `ownership`
- Signature: `(show_log = False) -> DataFrame chứa cơ cấu cổ đông.`
- Return type: `DataFrame chứa cơ cấu cổ đông.`
- Normalization mode: `contractual`
- Supported sources: `kbs`
- Declared signature: `()`
- Default route source: `kbs`
- Default provider: `company.Company.ownership`

Get company ownership composition.

## Purpose

Get company ownership composition.

## Parameters

| Name | Kind | Required | Default | Annotation | Description |
| --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | Hiển thị log debug. |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `ownership`

```text
owner_type, ownership_percentage, shares_owned, update_date
```
- Note: No source-specific raw mapping declared; using normalized columns as a best-effort approximation.

#### Normalized output schema

- Coverage: `declared`

```text
owner_type, ownership_percentage, shares_owned, update_date
```

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
