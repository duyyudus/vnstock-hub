# equity.list

- Class: `EquityReference`
- Method: `list`
- Signature: `(show_log: bool = False, to_df: bool = True)`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `kbs, vci`
- Declared signature: `()`
- Default route source: `vci`
- Default provider: `listing.Listing.all_symbols`

List all equity symbols.

## Purpose

List all equity symbols.

## Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `bool` |
| `to_df` | `POSITIONAL_OR_KEYWORD` | `False` | `True` | `bool` |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `partially-derived`

```text
symbol, org_name, exchange, icb_name, listing_type
```
- Note: No source-specific raw mapping declared; using normalized columns as a best-effort approximation.

#### Normalized output schema

- Coverage: `declared`

```text
symbol, org_name, exchange, icb_name, listing_type
```

Enum/value normalization:

- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

### Source `vci`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.listing.Listing`
- Provider method: `all_symbols`

```text
symbol, organ_name, icb_name3, exchange, type
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `organ_name` | `org_name` |
| `icb_name3` | `icb_name` |
| `exchange` | `exchange` |
| `type` | `listing_type` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, org_name, exchange, icb_name, listing_type
```

Enum/value normalization:

- `exchange`: {'VNINDEX': 'HOSE', 'HNXINDEX': 'HNX', 'UPCOMINDEX': 'UPCOM', 'HSX': 'HOSE'}

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
