# TopStock

- Qualified name: `app.lib.vnstock_data_alt.api.insight.TopStock`
- Signature: `(source='vnd', **D)`
- Supported sources: `vnd`

Adapter for VND TopStock “insight” APIs.  Only supports source="vnd".

## Purpose

Adapter for VND TopStock “insight” APIs.  Only supports source="vnd".

## Members

### deal

- Kind: `method`
- Signature: `(index='VNINDEX', limit=10)`
- Purpose: Top 10 by block trade volume in the given index.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `index` | `POSITIONAL_OR_KEYWORD` | `False` | `VNINDEX` | `` | `omitted; default 'VNINDEX'` |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `10` | `` | `5` |

#### Source details

##### Source `vnd`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vnd.insight.TopStock`
- Provider method: `deal`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:20:50.880900+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, index, last_price, last_updated, price_change_1d, price_change_pct_1d, accumulated_value, avg_volume_20d, volume_spike_20d_pct, total_volume_avg_20d, deal_volume_spike_20d_pct, deal_volume_spike_5d_20d_pct, deal_volume_sum_5d, deal_value_avg_5d, deal_volume_avg_5d
```
- Dtypes: `{'symbol': 'str', 'index': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'accumulated_value': 'float64', 'avg_volume_20d': 'float64', 'volume_spike_20d_pct': 'float64', 'total_volume_avg_20d': 'float64', 'deal_volume_spike_20d_pct': 'float64', 'deal_volume_spike_5d_20d_pct': 'float64', 'deal_volume_sum_5d': 'float64', 'deal_value_avg_5d': 'float64', 'deal_volume_avg_5d': 'float64'}`

```json
[
  {
    "symbol": "TVB",
    "index": "VNINDEX",
    "last_price": 7.34,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 0.0,
    "price_change_pct_1d": 0.0,
    "accumulated_value": 35586000.0,
    "avg_volume_20d": 31725.0,
    "volume_spike_20d_pct": 15.445232466509061,
    "total_volume_avg_20d": 52955.130000000005,
    "deal_volume_spike_20d_pct": 667.7400282087873,
    "deal_volume_spike_5d_20d_pct": 160.3631602830547,
    "deal_volume_sum_5d": 424602.6,
    "deal_value_avg_5d": 97696004.83021152,
    "deal_volume_avg_5d": 84920.51999999999
  },
  {
    "symbol": "SAM",
    "index": "VNINDEX",
    "last_price": 6.72,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 0.019999999999999574,
    "price_change_pct_1d": 0.29850746268655914,
    "accumulated_value": 1323511000.0,
    "avg_volume_20d": 260040.0,
    "volume_spike_20d_pct": 75.68066451315183,
    "total_volume_avg_20d": 877974.0,
    "deal_volume_spike_20d_pct": 54.78522143024737,
    "deal_volume_spike_5d_20d_pct": 92.28063701202997,
    "deal_volume_sum_5d": 4051000.0,
    "deal_value_avg_5d": 4284000006.253,
    "deal_volume_avg_5d": 810200.0
  },
  {
    "symbol": "SHI",
    "index": "VNINDEX",
    "last_price": 14.85,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 0.8499999999999996,
    "price_change_pct_1d": 6.071428571428572,
    "accumulated_value": 8993785000.0,
    "avg_volume_20d": 449005.0,
    "volume_spike_20d_pct": 142.0028730192314,
    "total_volume_avg_20d": 933155.0,
    "deal_volume_spike_20d_pct": 47.23759718374761,
    "deal_volume_spike_5d_20d_pct": 141.46631588535666,
    "deal_volume_sum_5d": 6600500.0,
    "deal_value_avg_5d": 16993654012.591198,
    "deal_volume_avg_5d": 1320100.0
  }
]
```

#### Notes / caveats

Top 10 by block trade volume in the given index.

### foreign_buy

- Kind: `method`
- Signature: `(date=None, limit=10)`
- Purpose: Top 10 net foreign buys on the given date.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `date` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `10` | `` | `5` |

#### Source details

##### Source `vnd`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vnd.insight.TopStock`
- Provider method: `foreign_buy`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:20:51.127283+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, date, net_value
```
- Dtypes: `{'symbol': 'str', 'date': 'str', 'net_value': 'float64'}`

```json
[
  {
    "symbol": "MCH",
    "date": "2026-03-16",
    "net_value": 114295100000.0
  },
  {
    "symbol": "PVS",
    "date": "2026-03-16",
    "net_value": 81343240000.0
  },
  {
    "symbol": "VCK",
    "date": "2026-03-16",
    "net_value": 55230680000.0
  }
]
```

#### Notes / caveats

Top 10 net foreign buys on the given date.

### foreign_sell

- Kind: `method`
- Signature: `(date=None, limit=10)`
- Purpose: Top 10 net foreign sells on the given date.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `date` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | `omitted in live probe` |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `10` | `` | `5` |

#### Source details

##### Source `vnd`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vnd.insight.TopStock`
- Provider method: `foreign_sell`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:20:51.286714+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, date, net_value
```
- Dtypes: `{'symbol': 'str', 'date': 'str', 'net_value': 'float64'}`

```json
[
  {
    "symbol": "BSR",
    "date": "2026-03-16",
    "net_value": -181331770000.0
  },
  {
    "symbol": "VIC",
    "date": "2026-03-16",
    "net_value": -159806460000.0
  },
  {
    "symbol": "PVD",
    "date": "2026-03-16",
    "net_value": -147277600000.0
  }
]
```

#### Notes / caveats

Top 10 net foreign sells on the given date.

### gainer

- Kind: `method`
- Signature: `(index='VNINDEX', limit=10)`
- Purpose: Top 10 gainers in the given index.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `index` | `POSITIONAL_OR_KEYWORD` | `False` | `VNINDEX` | `` | `omitted; default 'VNINDEX'` |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `10` | `` | `5` |

#### Source details

##### Source `vnd`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vnd.insight.TopStock`
- Provider method: `gainer`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:20:51.444076+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, index, last_price, last_updated, price_change_1d, price_change_pct_1d, accumulated_value, avg_volume_20d, volume_spike_20d_pct, total_volume_avg_20d, deal_volume_spike_20d_pct, deal_volume_spike_5d_20d_pct, deal_volume_sum_5d, deal_value_avg_5d, deal_volume_avg_5d
```
- Dtypes: `{'symbol': 'str', 'index': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'accumulated_value': 'float64', 'avg_volume_20d': 'float64', 'volume_spike_20d_pct': 'float64', 'total_volume_avg_20d': 'float64', 'deal_volume_spike_20d_pct': 'float64', 'deal_volume_spike_5d_20d_pct': 'float64', 'deal_volume_sum_5d': 'float64', 'deal_value_avg_5d': 'float64', 'deal_volume_avg_5d': 'float64'}`

```json
[
  {
    "symbol": "MCH",
    "index": "VNINDEX",
    "last_price": 149.8,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 9.800000000000011,
    "price_change_pct_1d": 7.000000000000006,
    "accumulated_value": 221697820000.0,
    "avg_volume_20d": 307070.0,
    "volume_spike_20d_pct": 482.88663822581174,
    "total_volume_avg_20d": 530856.6,
    "deal_volume_spike_20d_pct": 0.0,
    "deal_volume_spike_5d_20d_pct": 0.0,
    "deal_volume_sum_5d": 0.0,
    "deal_value_avg_5d": 0.0,
    "deal_volume_avg_5d": 0.0
  },
  {
    "symbol": "VCK",
    "index": "VNINDEX",
    "last_price": 33.4,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 2.1499999999999986,
    "price_change_pct_1d": 6.879999999999997,
    "accumulated_value": 357437005000.0,
    "avg_volume_20d": 4083135.0,
    "volume_spike_20d_pct": 265.87413837651707,
    "total_volume_avg_20d": 4805482.15,
    "deal_volume_spike_20d_pct": 3.9602685861604954,
    "deal_volume_spike_5d_20d_pct": 0.792053717232099,
    "deal_volume_sum_5d": 190310.0,
    "deal_value_avg_5d": 12.648178,
    "deal_volume_avg_5d": 38062.0
  },
  {
    "symbol": "NVL",
    "index": "VNINDEX",
    "last_price": 13.5,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 0.8499999999999996,
    "price_change_pct_1d": 6.719367588932812,
    "accumulated_value": 656318670000.0,
    "avg_volume_20d": 10664010.0,
    "volume_spike_20d_pct": 459.486628388383,
    "total_volume_avg_20d": 10664010.0,
    "deal_volume_spike_20d_pct": 0.0,
    "deal_volume_spike_5d_20d_pct": 0.0,
    "deal_volume_sum_5d": 0.0,
    "deal_value_avg_5d": 0.0,
    "deal_volume_avg_5d": 0.0
  }
]
```

#### Notes / caveats

Top 10 gainers in the given index.

### history

- Kind: `method`
- Signature: `(*A, **B)`

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `A` | `VAR_POSITIONAL` | `True` | `None` | `` |
| `B` | `VAR_KEYWORD` | `True` | `None` | `` |

#### Source details

##### Source `vnd`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `history` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

### loser

- Kind: `method`
- Signature: `(index='VNINDEX', limit=10)`
- Purpose: Top 10 losers in the given index.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `index` | `POSITIONAL_OR_KEYWORD` | `False` | `VNINDEX` | `` | `omitted; default 'VNINDEX'` |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `10` | `` | `5` |

#### Source details

##### Source `vnd`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vnd.insight.TopStock`
- Provider method: `loser`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:20:55.615867+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, index, last_price, last_updated, price_change_1d, price_change_pct_1d, accumulated_value, avg_volume_20d, volume_spike_20d_pct, total_volume_avg_20d, deal_volume_spike_20d_pct, deal_volume_spike_5d_20d_pct, deal_volume_sum_5d, deal_value_avg_5d, deal_volume_avg_5d
```
- Dtypes: `{'symbol': 'str', 'index': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'accumulated_value': 'float64', 'avg_volume_20d': 'float64', 'volume_spike_20d_pct': 'float64', 'total_volume_avg_20d': 'float64', 'deal_volume_spike_20d_pct': 'float64', 'deal_volume_spike_5d_20d_pct': 'float64', 'deal_volume_sum_5d': 'float64', 'deal_value_avg_5d': 'float64', 'deal_volume_avg_5d': 'float64'}`

```json
[
  {
    "symbol": "PVD",
    "index": "VNINDEX",
    "last_price": 37.2,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": -2.799999999999997,
    "price_change_pct_1d": -6.999999999999995,
    "accumulated_value": 359934675000.0,
    "avg_volume_20d": 10418030.0,
    "volume_spike_20d_pct": 92.30055970274611,
    "total_volume_avg_20d": 11127740.0,
    "deal_volume_spike_20d_pct": 0.0,
    "deal_volume_spike_5d_20d_pct": 4.493275364090103,
    "deal_volume_sum_5d": 2500000.0,
    "deal_value_avg_5d": 20000000000.0,
    "deal_volume_avg_5d": 500000.0
  },
  {
    "symbol": "DCM",
    "index": "VNINDEX",
    "last_price": 44.55,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": -3.3500000000000014,
    "price_change_pct_1d": -6.993736951983298,
    "accumulated_value": 368183510000.0,
    "avg_volume_20d": 7388050.0,
    "volume_spike_20d_pct": 111.05772159094754,
    "total_volume_avg_20d": 7510784.01,
    "deal_volume_spike_20d_pct": 5.591959500377112e-05,
    "deal_volume_spike_5d_20d_pct": 0.00038930689474053987,
    "deal_volume_sum_5d": 146.2,
    "deal_value_avg_5d": 1341200.00042168,
    "deal_volume_avg_5d": 29.24
  },
  {
    "symbol": "VVS",
    "index": "VNINDEX",
    "last_price": 149.3,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": -11.199999999999989,
    "price_change_pct_1d": -6.978193146417433,
    "accumulated_value": 22070820000.0,
    "avg_volume_20d": 212060.0,
    "volume_spike_20d_pct": 68.94275205130623,
    "total_volume_avg_20d": 212060.0,
    "deal_volume_spike_20d_pct": 0.0,
    "deal_volume_spike_5d_20d_pct": 0.0,
    "deal_volume_sum_5d": 0.0,
    "deal_value_avg_5d": 0.0,
    "deal_volume_avg_5d": 0.0
  }
]
```

#### Notes / caveats

Top 10 losers in the given index.

### value

- Kind: `method`
- Signature: `(index='VNINDEX', limit=10)`
- Purpose: Top 10 by trading value in the given index.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `index` | `POSITIONAL_OR_KEYWORD` | `False` | `VNINDEX` | `` | `omitted; default 'VNINDEX'` |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `10` | `` | `5` |

#### Source details

##### Source `vnd`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vnd.insight.TopStock`
- Provider method: `value`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:20:55.752247+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, index, last_price, last_updated, price_change_1d, price_change_pct_1d, accumulated_value, avg_volume_20d, volume_spike_20d_pct, total_volume_avg_20d, deal_volume_spike_20d_pct, deal_volume_spike_5d_20d_pct, deal_volume_sum_5d, deal_value_avg_5d, deal_volume_avg_5d
```
- Dtypes: `{'symbol': 'str', 'index': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'accumulated_value': 'float64', 'avg_volume_20d': 'float64', 'volume_spike_20d_pct': 'float64', 'total_volume_avg_20d': 'float64', 'deal_volume_spike_20d_pct': 'float64', 'deal_volume_spike_5d_20d_pct': 'float64', 'deal_volume_sum_5d': 'float64', 'deal_value_avg_5d': 'float64', 'deal_volume_avg_5d': 'float64'}`

```json
[
  {
    "symbol": "SHB",
    "index": "VNINDEX",
    "last_price": 15.2,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 0.25,
    "price_change_pct_1d": 1.6722408026755842,
    "accumulated_value": 1258550880000.0,
    "avg_volume_20d": 66239890.0,
    "volume_spike_20d_pct": 127.1067932027061,
    "total_volume_avg_20d": 70049684.55,
    "deal_volume_spike_20d_pct": 1.3671724664461748,
    "deal_volume_spike_5d_20d_pct": 8.006231628347853,
    "deal_volume_sum_5d": 28041700.0,
    "deal_value_avg_5d": 78361320028.731,
    "deal_volume_avg_5d": 5608340.0
  },
  {
    "symbol": "SSI",
    "index": "VNINDEX",
    "last_price": 28.4,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 0.0,
    "price_change_pct_1d": 0.0,
    "accumulated_value": 744127825000.0,
    "avg_volume_20d": 45793770.0,
    "volume_spike_20d_pct": 56.865158732290446,
    "total_volume_avg_20d": 45910450.5,
    "deal_volume_spike_20d_pct": 0.039184978156552835,
    "deal_volume_spike_5d_20d_pct": 0.2343666830278653,
    "deal_volume_sum_5d": 537994.0,
    "deal_value_avg_5d": 3041821941.036151,
    "deal_volume_avg_5d": 107598.8
  },
  {
    "symbol": "STB",
    "index": "VNINDEX",
    "last_price": 66.6,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 0.7999999999999972,
    "price_change_pct_1d": 1.2158054711246091,
    "accumulated_value": 734224430000.0,
    "avg_volume_20d": 12284555.0,
    "volume_spike_20d_pct": 88.57789313491617,
    "total_volume_avg_20d": 13912955.0,
    "deal_volume_spike_20d_pct": 0.0,
    "deal_volume_spike_5d_20d_pct": 15.464292093232531,
    "deal_volume_sum_5d": 10757700.0,
    "deal_value_avg_5d": 131444720000.0,
    "deal_volume_avg_5d": 2151540.0
  }
]
```

#### Notes / caveats

Top 10 by trading value in the given index.

### volume

- Kind: `method`
- Signature: `(index='VNINDEX', limit=10)`
- Purpose: Top 10 by abnormal volume in the given index.

#### Parameters

| Name | Kind | Required | Default | Annotation | Observed example |
| --- | --- | --- | --- | --- | --- |
| `index` | `POSITIONAL_OR_KEYWORD` | `False` | `VNINDEX` | `` | `omitted; default 'VNINDEX'` |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `10` | `` | `5` |

#### Source details

##### Source `vnd`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.vnd.insight.TopStock`
- Provider method: `volume`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:20:56.613288+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, index, last_price, last_updated, price_change_1d, price_change_pct_1d, accumulated_value, avg_volume_20d, volume_spike_20d_pct, total_volume_avg_20d, deal_volume_spike_20d_pct, deal_volume_spike_5d_20d_pct, deal_volume_sum_5d, deal_value_avg_5d, deal_volume_avg_5d
```
- Dtypes: `{'symbol': 'str', 'index': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'accumulated_value': 'float64', 'avg_volume_20d': 'float64', 'volume_spike_20d_pct': 'float64', 'total_volume_avg_20d': 'float64', 'deal_volume_spike_20d_pct': 'float64', 'deal_volume_spike_5d_20d_pct': 'float64', 'deal_volume_sum_5d': 'float64', 'deal_value_avg_5d': 'float64', 'deal_volume_avg_5d': 'float64'}`

```json
[
  {
    "symbol": "PTL",
    "index": "VNINDEX",
    "last_price": 2.89,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 0.18000000000000016,
    "price_change_pct_1d": 6.6420664206642055,
    "accumulated_value": 657445000.0,
    "avg_volume_20d": 35095.0,
    "volume_spike_20d_pct": 648.2404900983046,
    "total_volume_avg_20d": 35095.0,
    "deal_volume_spike_20d_pct": 0.0,
    "deal_volume_spike_5d_20d_pct": 0.0,
    "deal_volume_sum_5d": 0.0,
    "deal_value_avg_5d": 0.0,
    "deal_volume_avg_5d": 0.0
  },
  {
    "symbol": "TRA",
    "index": "VNINDEX",
    "last_price": 69.0,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": -0.9000000000000057,
    "price_change_pct_1d": -1.2875536480686733,
    "accumulated_value": 5934880000.0,
    "avg_volume_20d": 16370.0,
    "volume_spike_20d_pct": 552.8405620036652,
    "total_volume_avg_20d": 16370.0,
    "deal_volume_spike_20d_pct": 0.0,
    "deal_volume_spike_5d_20d_pct": 0.0,
    "deal_volume_sum_5d": 0.0,
    "deal_value_avg_5d": 0.0,
    "deal_volume_avg_5d": 0.0
  },
  {
    "symbol": "MCH",
    "index": "VNINDEX",
    "last_price": 149.8,
    "last_updated": "2026-03-16 15:59",
    "price_change_1d": 9.800000000000011,
    "price_change_pct_1d": 7.000000000000006,
    "accumulated_value": 221697820000.0,
    "avg_volume_20d": 307070.0,
    "volume_spike_20d_pct": 482.88663822581174,
    "total_volume_avg_20d": 530856.6,
    "deal_volume_spike_20d_pct": 0.0,
    "deal_volume_spike_5d_20d_pct": 0.0,
    "deal_volume_sum_5d": 0.0,
    "deal_value_avg_5d": 0.0,
    "deal_volume_avg_5d": 0.0
  }
]
```

#### Notes / caveats

Top 10 by abnormal volume in the given index.
