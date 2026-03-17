# CommodityPrice

- Qualified name: `app.lib.vnstock_data_alt.api.commodity.CommodityPrice`
- Signature: `(source='spl', start=None, end=None, length=None, show_log=False)`
- Supported sources: `spl`

Adapter for commodity prices from various SPL‐based providers.

## Purpose

Adapter for commodity prices from various SPL‐based providers.

Usage:
    from app.lib.vnstock_data_alt.api.commodity import CommodityPrice
    c = CommodityPrice(source="spl", start="2024-01-01", end="2024-04-01", show_log=False)
    df = c.gold_vn()

## Members

### coke

- Kind: `method`
- Signature: `(start=None, end=None, length=None)`
- Purpose: Coke price.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-07` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |

#### Source details

##### Source `spl`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.spl.commodity.CommodityPrice`
- Provider method: `coke`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:07.368083+00:00`
- Success: `True`
- Row count: `5`

```text
open, high, low, close, volume
```
- Dtypes: `{'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'}`

```json
[
  {
    "open": 103.0,
    "high": 103.0,
    "low": 102.95,
    "close": 102.95,
    "volume": 65.0
  },
  {
    "open": 103.45,
    "high": 103.6,
    "low": 103.0,
    "close": 103.6,
    "volume": 121.0
  },
  {
    "open": 101.05,
    "high": 105.15,
    "low": 101.05,
    "close": 105.15,
    "volume": 20.0
  }
]
```

#### Notes / caveats

Coke price.

### corn

- Kind: `method`
- Signature: `(start=None, end=None, length=None)`
- Purpose: Corn price.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-07` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |

#### Source details

##### Source `spl`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.spl.commodity.CommodityPrice`
- Provider method: `corn`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:07.659770+00:00`
- Success: `True`
- Row count: `5`

```text
open, high, low, close, volume
```
- Dtypes: `{'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'}`

```json
[
  {
    "open": 469.75,
    "high": 472.75,
    "low": 454.5,
    "close": 456.25,
    "volume": 353207.0
  },
  {
    "open": 456.0,
    "high": 459.25,
    "low": 442.5,
    "close": 451.5,
    "volume": 391274.0
  },
  {
    "open": 454.5,
    "high": 460.5,
    "low": 448.5,
    "close": 455.75,
    "volume": 277827.0
  }
]
```

#### Notes / caveats

Corn price.

### fertilizer_ure

- Kind: `method`
- Signature: `(start=None, end=None, length=None)`
- Purpose: Urea fertilizer price.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-07` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |

#### Source details

##### Source `spl`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.spl.commodity.CommodityPrice`
- Provider method: `fertilizer_ure`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:07.837837+00:00`
- Success: `True`
- Row count: `5`

```text
open, high, low, close, volume
```
- Dtypes: `{'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'}`

```json
[
  {
    "open": 405.0,
    "high": 405.0,
    "low": 405.0,
    "close": 405.0,
    "volume": 0.0
  },
  {
    "open": 406.0,
    "high": 406.0,
    "low": 406.0,
    "close": 406.0,
    "volume": 0.0
  },
  {
    "open": 402.5,
    "high": 402.5,
    "low": 402.5,
    "close": 402.5,
    "volume": 0.0
  }
]
```

#### Notes / caveats

Urea fertilizer price.

### gas_natural

- Kind: `method`
- Signature: `(start=None, end=None, length=None)`
- Purpose: Natural gas price.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-07` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |

#### Source details

##### Source `spl`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.spl.commodity.CommodityPrice`
- Provider method: `gas_natural`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:08.035844+00:00`
- Success: `True`
- Row count: `5`

```text
open, high, low, close, volume
```
- Dtypes: `{'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'}`

```json
[
  {
    "open": 3.78,
    "high": 4.17,
    "low": 3.745,
    "close": 4.12,
    "volume": 7914.0
  },
  {
    "open": 4.165,
    "high": 4.55,
    "low": 4.05,
    "close": 4.35,
    "volume": 10754.0
  },
  {
    "open": 4.31,
    "high": 4.52,
    "low": 4.225,
    "close": 4.45,
    "volume": 8547.0
  }
]
```

#### Notes / caveats

Natural gas price.

### gas_vn

- Kind: `method`
- Signature: `(start=None, end=None, length=None)`
- Purpose: Vietnam gasoline & diesel prices.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-07` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |

#### Source details

##### Source `spl`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.spl.commodity.CommodityPrice`
- Provider method: `gas_vn`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:09.754873+00:00`
- Success: `True`
- Row count: `7`

```text
ron95, ron92, oil_do
```
- Dtypes: `{'ron95': 'float64', 'ron92': 'float64', 'oil_do': 'float64'}`

```json
[
  {
    "ron95": 21.11,
    "ron92": 20.65,
    "oil_do": 18.95
  },
  {
    "ron95": 21.11,
    "ron92": 20.65,
    "oil_do": 18.95
  },
  {
    "ron95": 21.11,
    "ron92": 20.65,
    "oil_do": 18.95
  }
]
```

#### Notes / caveats

Vietnam gasoline & diesel prices.

### gold_global

- Kind: `method`
- Signature: `(start=None, end=None, length=None)`
- Purpose: Global gold price.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-07` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |

#### Source details

##### Source `spl`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.spl.commodity.CommodityPrice`
- Provider method: `gold_global`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:09.926769+00:00`
- Success: `True`
- Row count: `5`

```text
open, high, low, close, volume
```
- Dtypes: `{'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'}`

```json
[
  {
    "open": 2872.0,
    "high": 2906.4,
    "low": 2866.3,
    "close": 2901.1,
    "volume": 177018.0
  },
  {
    "open": 2904.2,
    "high": 2939.8,
    "low": 2892.5,
    "close": 2920.6,
    "volume": 188515.0
  },
  {
    "open": 2929.0,
    "high": 2941.3,
    "low": 2903.4,
    "close": 2926.0,
    "volume": 183016.0
  }
]
```

#### Notes / caveats

Global gold price.

### gold_vn

- Kind: `method`
- Signature: `(start=None, end=None, length=None)`
- Purpose: Vietnam gold prices (buy & sell).

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-07` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |

#### Source details

##### Source `spl`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.spl.commodity.CommodityPrice`
- Provider method: `gold_vn`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:10.403077+00:00`
- Success: `True`
- Row count: `6`

```text
buy, sell
```
- Dtypes: `{'buy': 'float64', 'sell': 'float64'}`

```json
[
  {
    "buy": 88500.0,
    "sell": 90500.0
  },
  {
    "buy": 89000.0,
    "sell": 91000.0
  },
  {
    "buy": 89600.0,
    "sell": 91600.0
  }
]
```

#### Notes / caveats

Vietnam gold prices (buy & sell).

### iron_ore

- Kind: `method`
- Signature: `(start=None, end=None, length=None)`
- Purpose: Iron ore price.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-07` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |

#### Source details

##### Source `spl`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.spl.commodity.CommodityPrice`
- Provider method: `iron_ore`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:10.605375+00:00`
- Success: `True`
- Row count: `5`

```text
open, high, low, close, volume
```
- Dtypes: `{'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'}`

```json
[
  {
    "open": 100.81,
    "high": 100.81,
    "low": 100.81,
    "close": 100.81,
    "volume": 0.0
  },
  {
    "open": 101.61,
    "high": 101.61,
    "low": 101.61,
    "close": 101.61,
    "volume": 0.0
  },
  {
    "open": 100.72,
    "high": 100.72,
    "low": 100.72,
    "close": 100.72,
    "volume": 0.0
  }
]
```

#### Notes / caveats

Iron ore price.

### oil_crude

- Kind: `method`
- Signature: `(start=None, end=None, length=None)`
- Purpose: Crude oil price.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-07` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |

#### Source details

##### Source `spl`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.spl.commodity.CommodityPrice`
- Provider method: `oil_crude`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:10.884304+00:00`
- Success: `True`
- Row count: `5`

```text
open, high, low, close, volume
```
- Dtypes: `{'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'}`

```json
[
  {
    "open": 69.95,
    "high": 70.6,
    "low": 67.89,
    "close": 68.37,
    "volume": 332751.0
  },
  {
    "open": 68.46,
    "high": 68.56,
    "low": 66.77,
    "close": 68.26,
    "volume": 386750.0
  },
  {
    "open": 68.08,
    "high": 68.1,
    "low": 65.22,
    "close": 66.31,
    "volume": 382493.0
  }
]
```

#### Notes / caveats

Crude oil price.

### pork_china

- Kind: `method`
- Signature: `(start=None, end=None, length=None)`
- Purpose: China live hog price.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-07` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |

#### Source details

##### Source `spl`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.spl.commodity.CommodityPrice`
- Provider method: `pork_china`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:11.060577+00:00`
- Success: `True`
- Row count: `1`

```text
close
```
- Dtypes: `{'close': 'float64'}`

```json
[
  {
    "close": 15.47
  }
]
```

#### Notes / caveats

China live hog price.

### pork_north_vn

- Kind: `method`
- Signature: `(start=None, end=None, length=None)`
- Purpose: Northern Vietnam live hog price.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-07` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |

#### Source details

##### Source `spl`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.spl.commodity.CommodityPrice`
- Provider method: `pork_north_vn`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:11.346204+00:00`
- Success: `True`
- Row count: `4`

```text
close
```
- Dtypes: `{'close': 'float64'}`

```json
[
  {
    "close": 73600.0
  },
  {
    "close": 73400.0
  },
  {
    "close": 74800.0
  }
]
```

#### Notes / caveats

Northern Vietnam live hog price.

### soybean

- Kind: `method`
- Signature: `(start=None, end=None, length=None)`
- Purpose: Soybean price.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-07` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |

#### Source details

##### Source `spl`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.spl.commodity.CommodityPrice`
- Provider method: `soybean`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:11.545434+00:00`
- Success: `True`
- Row count: `5`

```text
open, high, low, close, volume
```
- Dtypes: `{'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'}`

```json
[
  {
    "open": 1026.0,
    "high": 1031.25,
    "low": 1008.0,
    "close": 1011.5,
    "volume": 154480.0
  },
  {
    "open": 1010.75,
    "high": 1013.0,
    "low": 991.0,
    "close": 999.0,
    "volume": 211927.0
  },
  {
    "open": 1004.0,
    "high": 1015.0,
    "low": 995.75,
    "close": 1011.75,
    "volume": 159722.0
  }
]
```

#### Notes / caveats

Soybean price.

### steel_d10

- Kind: `method`
- Signature: `(start=None, end=None, length=None)`
- Purpose: Vietnam rebar D10 price.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-07` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |

#### Source details

##### Source `spl`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.spl.commodity.CommodityPrice`
- Provider method: `steel_d10`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:11.795884+00:00`
- Success: `True`
- Row count: `7`

```text
close
```
- Dtypes: `{'close': 'float64'}`

```json
[
  {
    "close": 13.58
  },
  {
    "close": 13.58
  },
  {
    "close": 13.58
  }
]
```

#### Notes / caveats

Vietnam rebar D10 price.

### steel_hrc

- Kind: `method`
- Signature: `(start=None, end=None, length=None)`
- Purpose: Hot‑rolled coil (HRC) steel price.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-07` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |

#### Source details

##### Source `spl`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.spl.commodity.CommodityPrice`
- Provider method: `steel_hrc`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:13.029226+00:00`
- Success: `True`
- Row count: `5`

```text
open, high, low, close, volume
```
- Dtypes: `{'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'}`

```json
[
  {
    "open": 915.0,
    "high": 916.0,
    "low": 911.0,
    "close": 914.0,
    "volume": 481.0
  },
  {
    "open": 915.0,
    "high": 918.0,
    "low": 913.0,
    "close": 914.0,
    "volume": 298.0
  },
  {
    "open": 918.0,
    "high": 928.0,
    "low": 918.0,
    "close": 926.0,
    "volume": 348.0
  }
]
```

#### Notes / caveats

Hot‑rolled coil (HRC) steel price.

### sugar

- Kind: `method`
- Signature: `(start=None, end=None, length=None)`
- Purpose: Sugar price.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-01` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `2025-03-07` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |

#### Source details

##### Source `spl`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.spl.commodity.CommodityPrice`
- Provider method: `sugar`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-17T05:29:13.203142+00:00`
- Success: `True`
- Row count: `5`

```text
open, high, low, close, volume
```
- Dtypes: `{'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'}`

```json
[
  {
    "open": 18.52,
    "high": 18.56,
    "low": 18.14,
    "close": 18.22,
    "volume": 67344.0
  },
  {
    "open": 18.2,
    "high": 18.28,
    "low": 17.94,
    "close": 18.1,
    "volume": 60157.0
  },
  {
    "open": 18.21,
    "high": 18.51,
    "low": 18.13,
    "close": 18.2,
    "volume": 50966.0
  }
]
```

#### Notes / caveats

Sugar price.
