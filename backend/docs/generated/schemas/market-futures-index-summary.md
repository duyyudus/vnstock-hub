# market.futures.index_summary

- Class: `_unknown_`
- Method: `index_summary`
- Signature: `()`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `kbs, msn`

## Parameters

_None._

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `declared`

```text
symbol, timestamp, close_price, price_change, percent_change, open_price, high_price, low_price, reference_price, advances, declines, no_change, accumulated_volume, accumulated_value, total_volume, put_through_volume, put_through_value, previous_close
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `timestamp` | `time` |
| `close_price` | `close_price` |
| `price_change` | `price_change` |
| `percent_change` | `percent_change` |
| `open_price` | `open_price` |
| `high_price` | `high_price` |
| `low_price` | `low_price` |
| `reference_price` | `reference_price` |
| `advances` | `advances` |
| `declines` | `declines` |
| `no_change` | `no_change` |
| `accumulated_volume` | `accumulated_volume` |
| `accumulated_value` | `accumulated_value` |
| `total_volume` | `total_volume` |
| `put_through_volume` | `put_through_volume` |
| `put_through_value` | `put_through_value` |
| `previous_close` | `previous_close` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, time, close_price, price_change, percent_change, open_price, high_price, low_price, reference_price, advances, declines, no_change, accumulated_volume, accumulated_value, total_volume, put_through_volume, put_through_value, previous_close
```

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

### Source `msn`

#### Raw output contract

- Coverage: `declared`

```text
symbol, timestamp, close_price, price_change, percent_change, open_price, high_price, low_price, reference_price, advances, declines, no_change, accumulated_volume, accumulated_value, total_volume, put_through_volume, put_through_value, previous_close
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `timestamp` | `time` |
| `close_price` | `close_price` |
| `price_change` | `price_change` |
| `percent_change` | `percent_change` |
| `open_price` | `open_price` |
| `high_price` | `high_price` |
| `low_price` | `low_price` |
| `reference_price` | `reference_price` |
| `advances` | `advances` |
| `declines` | `declines` |
| `no_change` | `no_change` |
| `accumulated_volume` | `accumulated_volume` |
| `accumulated_value` | `accumulated_value` |
| `total_volume` | `total_volume` |
| `put_through_volume` | `put_through_volume` |
| `put_through_value` | `put_through_value` |
| `previous_close` | `previous_close` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, time, close_price, price_change, percent_change, open_price, high_price, low_price, reference_price, advances, declines, no_change, accumulated_volume, accumulated_value, total_volume, put_through_volume, put_through_value, previous_close
```

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
