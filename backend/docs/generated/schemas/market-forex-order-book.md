# market.forex.order_book

- Class: `_unknown_`
- Method: `order_book`
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
bid_price_1, bid_vol_1, ask_price_1, ask_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, bid_price_4, bid_vol_4, bid_price_5, bid_vol_5, bid_price_6, bid_vol_6, bid_price_7, bid_vol_7, bid_price_8, bid_vol_8, bid_price_9, bid_vol_9, bid_price_10, bid_vol_10, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, ask_price_4, ask_vol_4, ask_price_5, ask_vol_5, ask_price_6, ask_vol_6, ask_price_7, ask_vol_7, ask_price_8, ask_vol_8, ask_price_9, ask_vol_9, ask_price_10, ask_vol_10
```

| Raw | Normalized |
| --- | --- |
| `bid_price_1` | `bid_price_1` |
| `bid_vol_1` | `bid_vol_1` |
| `ask_price_1` | `ask_price_1` |
| `ask_vol_1` | `ask_vol_1` |
| `bid_price_2` | `bid_price_2` |
| `bid_vol_2` | `bid_vol_2` |
| `bid_price_3` | `bid_price_3` |
| `bid_vol_3` | `bid_vol_3` |
| `bid_price_4` | `bid_price_4` |
| `bid_vol_4` | `bid_vol_4` |
| `bid_price_5` | `bid_price_5` |
| `bid_vol_5` | `bid_vol_5` |
| `bid_price_6` | `bid_price_6` |
| `bid_vol_6` | `bid_vol_6` |
| `bid_price_7` | `bid_price_7` |
| `bid_vol_7` | `bid_vol_7` |
| `bid_price_8` | `bid_price_8` |
| `bid_vol_8` | `bid_vol_8` |
| `bid_price_9` | `bid_price_9` |
| `bid_vol_9` | `bid_vol_9` |
| `bid_price_10` | `bid_price_10` |
| `bid_vol_10` | `bid_vol_10` |
| `ask_price_2` | `ask_price_2` |
| `ask_vol_2` | `ask_vol_2` |
| `ask_price_3` | `ask_price_3` |
| `ask_vol_3` | `ask_vol_3` |
| `ask_price_4` | `ask_price_4` |
| `ask_vol_4` | `ask_vol_4` |
| `ask_price_5` | `ask_price_5` |
| `ask_vol_5` | `ask_vol_5` |
| `ask_price_6` | `ask_price_6` |
| `ask_vol_6` | `ask_vol_6` |
| `ask_price_7` | `ask_price_7` |
| `ask_vol_7` | `ask_vol_7` |
| `ask_price_8` | `ask_price_8` |
| `ask_vol_8` | `ask_vol_8` |
| `ask_price_9` | `ask_price_9` |
| `ask_vol_9` | `ask_vol_9` |
| `ask_price_10` | `ask_price_10` |
| `ask_vol_10` | `ask_vol_10` |

#### Normalized output schema

- Coverage: `declared`

```text
bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, bid_price_4, bid_vol_4, bid_price_5, bid_vol_5, bid_price_6, bid_vol_6, bid_price_7, bid_vol_7, bid_price_8, bid_vol_8, bid_price_9, bid_vol_9, bid_price_10, bid_vol_10, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, ask_price_4, ask_vol_4, ask_price_5, ask_vol_5, ask_price_6, ask_vol_6, ask_price_7, ask_vol_7, ask_price_8, ask_vol_8, ask_price_9, ask_vol_9, ask_price_10, ask_vol_10
```

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

### Source `msn`

#### Raw output contract

- Coverage: `declared`

```text
bid_price_1, bid_vol_1, ask_price_1, ask_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, bid_price_4, bid_vol_4, bid_price_5, bid_vol_5, bid_price_6, bid_vol_6, bid_price_7, bid_vol_7, bid_price_8, bid_vol_8, bid_price_9, bid_vol_9, bid_price_10, bid_vol_10, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, ask_price_4, ask_vol_4, ask_price_5, ask_vol_5, ask_price_6, ask_vol_6, ask_price_7, ask_vol_7, ask_price_8, ask_vol_8, ask_price_9, ask_vol_9, ask_price_10, ask_vol_10
```

| Raw | Normalized |
| --- | --- |
| `bid_price_1` | `bid_price_1` |
| `bid_vol_1` | `bid_vol_1` |
| `ask_price_1` | `ask_price_1` |
| `ask_vol_1` | `ask_vol_1` |
| `bid_price_2` | `bid_price_2` |
| `bid_vol_2` | `bid_vol_2` |
| `bid_price_3` | `bid_price_3` |
| `bid_vol_3` | `bid_vol_3` |
| `bid_price_4` | `bid_price_4` |
| `bid_vol_4` | `bid_vol_4` |
| `bid_price_5` | `bid_price_5` |
| `bid_vol_5` | `bid_vol_5` |
| `bid_price_6` | `bid_price_6` |
| `bid_vol_6` | `bid_vol_6` |
| `bid_price_7` | `bid_price_7` |
| `bid_vol_7` | `bid_vol_7` |
| `bid_price_8` | `bid_price_8` |
| `bid_vol_8` | `bid_vol_8` |
| `bid_price_9` | `bid_price_9` |
| `bid_vol_9` | `bid_vol_9` |
| `bid_price_10` | `bid_price_10` |
| `bid_vol_10` | `bid_vol_10` |
| `ask_price_2` | `ask_price_2` |
| `ask_vol_2` | `ask_vol_2` |
| `ask_price_3` | `ask_price_3` |
| `ask_vol_3` | `ask_vol_3` |
| `ask_price_4` | `ask_price_4` |
| `ask_vol_4` | `ask_vol_4` |
| `ask_price_5` | `ask_price_5` |
| `ask_vol_5` | `ask_vol_5` |
| `ask_price_6` | `ask_price_6` |
| `ask_vol_6` | `ask_vol_6` |
| `ask_price_7` | `ask_price_7` |
| `ask_vol_7` | `ask_vol_7` |
| `ask_price_8` | `ask_price_8` |
| `ask_vol_8` | `ask_vol_8` |
| `ask_price_9` | `ask_price_9` |
| `ask_vol_9` | `ask_vol_9` |
| `ask_price_10` | `ask_price_10` |
| `ask_vol_10` | `ask_vol_10` |

#### Normalized output schema

- Coverage: `declared`

```text
bid_price_1, bid_vol_1, bid_price_2, bid_vol_2, bid_price_3, bid_vol_3, bid_price_4, bid_vol_4, bid_price_5, bid_vol_5, bid_price_6, bid_vol_6, bid_price_7, bid_vol_7, bid_price_8, bid_vol_8, bid_price_9, bid_vol_9, bid_price_10, bid_vol_10, ask_price_1, ask_vol_1, ask_price_2, ask_vol_2, ask_price_3, ask_vol_3, ask_price_4, ask_vol_4, ask_price_5, ask_vol_5, ask_price_6, ask_vol_6, ask_price_7, ask_vol_7, ask_price_8, ask_vol_8, ask_price_9, ask_vol_9, ask_price_10, ask_vol_10
```

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
