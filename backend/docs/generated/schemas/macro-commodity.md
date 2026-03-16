# macro.commodity

- Class: `_unknown_`
- Method: `commodity`
- Signature: `()`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `spl`

## Parameters

_None._

## Source details

### Source `spl`

#### Raw output contract

- Coverage: `partially-derived`

```text
report_time, open, high, low, close, volume, buy, sell, ron95, ron92, oil_do
```
- Note: No source-specific raw mapping declared; using normalized columns as a best-effort approximation.

#### Normalized output schema

- Coverage: `declared`

```text
report_time, open, high, low, close, volume, buy, sell, ron95, ron92, oil_do
```

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
