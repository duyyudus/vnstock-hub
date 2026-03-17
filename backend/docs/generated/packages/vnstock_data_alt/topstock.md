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

- Captured at: `2026-03-17T05:29:13.367123+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, index, last_price, last_updated, price_change_1d, price_change_pct_1d, accumulated_value, avg_volume_20d, volume_spike_20d_pct, total_volume_avg_20d, deal_volume_spike_20d_pct, deal_volume_spike_5d_20d_pct, deal_volume_sum_5d, deal_value_avg_5d, deal_volume_avg_5d
```
- Dtypes: `{'symbol': 'str', 'index': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'accumulated_value': 'float64', 'avg_volume_20d': 'float64', 'volume_spike_20d_pct': 'float64', 'total_volume_avg_20d': 'float64', 'deal_volume_spike_20d_pct': 'float64', 'deal_volume_spike_5d_20d_pct': 'float64', 'deal_volume_sum_5d': 'float64', 'deal_value_avg_5d': 'float64', 'deal_volume_avg_5d': 'float64'}`

```json
[
  {
    "symbol": "SMC",
    "index": "VNINDEX",
    "last_price": 11.3,
    "last_updated": "2026-03-17 12:29",
    "price_change_1d": 0.3000000000000007,
    "price_change_pct_1d": 2.7272727272727337,
    "accumulated_value": 1059515000.0,
    "avg_volume_20d": 236430.0,
    "volume_spike_20d_pct": 39.92725119485683,
    "total_volume_avg_20d": 476430.0,
    "deal_volume_spike_20d_pct": 83.95776924207124,
    "deal_volume_spike_5d_20d_pct": 75.56199231786411,
    "deal_volume_sum_5d": 1800000.0,
    "deal_value_avg_5d": 2842000008.2,
    "deal_volume_avg_5d": 360000.0
  },
  {
    "symbol": "SSB",
    "index": "VNINDEX",
    "last_price": 16.65,
    "last_updated": "2026-03-17 12:29",
    "price_change_1d": 0.09999999999999787,
    "price_change_pct_1d": 0.6042296072507503,
    "accumulated_value": 10733495000.0,
    "avg_volume_20d": 2056855.0,
    "volume_spike_20d_pct": 31.426619766585397,
    "total_volume_avg_20d": 6241205.0,
    "deal_volume_spike_20d_pct": 25.956526023420157,
    "deal_volume_spike_5d_20d_pct": 95.20597384639665,
    "deal_volume_sum_5d": 29710000.0,
    "deal_value_avg_5d": 93992800053.784,
    "deal_volume_avg_5d": 5942000.0
  },
  {
    "symbol": "ACC",
    "index": "VNINDEX",
    "last_price": 12.0,
    "last_updated": "2026-03-17 12:29",
    "price_change_1d": -0.5,
    "price_change_pct_1d": -4.0000000000000036,
    "accumulated_value": 333760000.0,
    "avg_volume_20d": 46585.0,
    "volume_spike_20d_pct": 58.17323172695074,
    "total_volume_avg_20d": 82585.0,
    "deal_volume_spike_20d_pct": 24.217472906702188,
    "deal_volume_spike_5d_20d_pct": 4.843494581340437,
    "deal_volume_sum_5d": 20000.0,
    "deal_value_avg_5d": 0.466,
    "deal_volume_avg_5d": 4000.0
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

- Captured at: `2026-03-17T05:29:13.541047+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, date, net_value
```
- Dtypes: `{'symbol': 'str', 'date': 'str', 'net_value': 'float64'}`

```json
[
  {
    "symbol": "VCK",
    "date": "2026-03-17",
    "net_value": 143322490000.0
  },
  {
    "symbol": "MCH",
    "date": "2026-03-17",
    "net_value": 126778020000.0
  },
  {
    "symbol": "MSN",
    "date": "2026-03-17",
    "net_value": 54155540000.0
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

- Captured at: `2026-03-17T05:29:13.681916+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, date, net_value
```
- Dtypes: `{'symbol': 'str', 'date': 'str', 'net_value': 'float64'}`

```json
[
  {
    "symbol": "BID",
    "date": "2026-03-17",
    "net_value": -75455010000.0
  },
  {
    "symbol": "VIC",
    "date": "2026-03-17",
    "net_value": -55447478300.0
  },
  {
    "symbol": "DGC",
    "date": "2026-03-17",
    "net_value": -31025790000.0
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

- Captured at: `2026-03-17T05:29:13.892075+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, index, last_price, last_updated, price_change_1d, price_change_pct_1d, accumulated_value, avg_volume_20d, volume_spike_20d_pct, total_volume_avg_20d, deal_volume_spike_20d_pct, deal_volume_spike_5d_20d_pct, deal_volume_sum_5d, deal_value_avg_5d, deal_volume_avg_5d
```
- Dtypes: `{'symbol': 'str', 'index': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'accumulated_value': 'float64', 'avg_volume_20d': 'float64', 'volume_spike_20d_pct': 'float64', 'total_volume_avg_20d': 'float64', 'deal_volume_spike_20d_pct': 'float64', 'deal_volume_spike_5d_20d_pct': 'float64', 'deal_volume_sum_5d': 'float64', 'deal_value_avg_5d': 'float64', 'deal_volume_avg_5d': 'float64'}`

```json
[
  {
    "symbol": "GEE",
    "index": "VNINDEX",
    "last_price": 155.1,
    "last_updated": "2026-03-17 12:29",
    "price_change_1d": 10.099999999999994,
    "price_change_pct_1d": 6.965517241379304,
    "accumulated_value": 178933220000.0,
    "avg_volume_20d": 801425.0,
    "volume_spike_20d_pct": 145.04164457060858,
    "total_volume_avg_20d": 818825.0,
    "deal_volume_spike_20d_pct": 0.0,
    "deal_volume_spike_5d_20d_pct": 0.0,
    "deal_volume_sum_5d": 0.0,
    "deal_value_avg_5d": 0.0,
    "deal_volume_avg_5d": 0.0
  },
  {
    "symbol": "NO1",
    "index": "VNINDEX",
    "last_price": 5.99,
    "last_updated": "2026-03-17 12:29",
    "price_change_1d": 0.39000000000000057,
    "price_change_pct_1d": 6.964285714285734,
    "accumulated_value": 538024000.0,
    "avg_volume_20d": 35220.0,
    "volume_spike_20d_pct": 257.8080636002271,
    "total_volume_avg_20d": 35220.0,
    "deal_volume_spike_20d_pct": 0.0,
    "deal_volume_spike_5d_20d_pct": 0.0,
    "deal_volume_sum_5d": 0.0,
    "deal_value_avg_5d": 0.0,
    "deal_volume_avg_5d": 0.0
  },
  {
    "symbol": "PTL",
    "index": "VNINDEX",
    "last_price": 3.09,
    "last_updated": "2026-03-17 12:29",
    "price_change_1d": 0.19999999999999973,
    "price_change_pct_1d": 6.92041522491349,
    "accumulated_value": 279027000.0,
    "avg_volume_20d": 38530.0,
    "volume_spike_20d_pct": 234.36283415520376,
    "total_volume_avg_20d": 38530.0,
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

- Captured at: `2026-03-17T05:29:14.033602+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, index, last_price, last_updated, price_change_1d, price_change_pct_1d, accumulated_value, avg_volume_20d, volume_spike_20d_pct, total_volume_avg_20d, deal_volume_spike_20d_pct, deal_volume_spike_5d_20d_pct, deal_volume_sum_5d, deal_value_avg_5d, deal_volume_avg_5d
```
- Dtypes: `{'symbol': 'str', 'index': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'accumulated_value': 'float64', 'avg_volume_20d': 'float64', 'volume_spike_20d_pct': 'float64', 'total_volume_avg_20d': 'float64', 'deal_volume_spike_20d_pct': 'float64', 'deal_volume_spike_5d_20d_pct': 'float64', 'deal_volume_sum_5d': 'float64', 'deal_value_avg_5d': 'float64', 'deal_volume_avg_5d': 'float64'}`

```json
[
  {
    "symbol": "BSR",
    "index": "VNINDEX",
    "last_price": 31.05,
    "last_updated": "2026-03-17 12:29",
    "price_change_1d": -1.6999999999999993,
    "price_change_pct_1d": -5.190839694656491,
    "accumulated_value": 315492080000.0,
    "avg_volume_20d": 22442380.0,
    "volume_spike_20d_pct": 44.77733645005566,
    "total_volume_avg_20d": 22445380.0,
    "deal_volume_spike_20d_pct": 0.0,
    "deal_volume_spike_5d_20d_pct": 0.0,
    "deal_volume_sum_5d": 0.0,
    "deal_value_avg_5d": 0.0,
    "deal_volume_avg_5d": 0.0
  },
  {
    "symbol": "BFC",
    "index": "VNINDEX",
    "last_price": 60.0,
    "last_updated": "2026-03-17 12:29",
    "price_change_1d": -2.5,
    "price_change_pct_1d": -4.0000000000000036,
    "accumulated_value": 10581230000.0,
    "avg_volume_20d": 410090.0,
    "volume_spike_20d_pct": 43.79526445414421,
    "total_volume_avg_20d": 410090.0,
    "deal_volume_spike_20d_pct": 0.0,
    "deal_volume_spike_5d_20d_pct": 0.0,
    "deal_volume_sum_5d": 0.0,
    "deal_value_avg_5d": 0.0,
    "deal_volume_avg_5d": 0.0
  },
  {
    "symbol": "ACC",
    "index": "VNINDEX",
    "last_price": 12.0,
    "last_updated": "2026-03-17 12:29",
    "price_change_1d": -0.5,
    "price_change_pct_1d": -4.0000000000000036,
    "accumulated_value": 333760000.0,
    "avg_volume_20d": 46585.0,
    "volume_spike_20d_pct": 58.17323172695074,
    "total_volume_avg_20d": 82585.0,
    "deal_volume_spike_20d_pct": 24.217472906702188,
    "deal_volume_spike_5d_20d_pct": 4.843494581340437,
    "deal_volume_sum_5d": 20000.0,
    "deal_value_avg_5d": 0.466,
    "deal_volume_avg_5d": 4000.0
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

- Captured at: `2026-03-17T05:29:14.193051+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, index, last_price, last_updated, price_change_1d, price_change_pct_1d, accumulated_value, avg_volume_20d, volume_spike_20d_pct, total_volume_avg_20d, deal_volume_spike_20d_pct, deal_volume_spike_5d_20d_pct, deal_volume_sum_5d, deal_value_avg_5d, deal_volume_avg_5d
```
- Dtypes: `{'symbol': 'str', 'index': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'accumulated_value': 'float64', 'avg_volume_20d': 'float64', 'volume_spike_20d_pct': 'float64', 'total_volume_avg_20d': 'float64', 'deal_volume_spike_20d_pct': 'float64', 'deal_volume_spike_5d_20d_pct': 'float64', 'deal_volume_sum_5d': 'float64', 'deal_value_avg_5d': 'float64', 'deal_volume_avg_5d': 'float64'}`

```json
[
  {
    "symbol": "VIX",
    "index": "VNINDEX",
    "last_price": 17.1,
    "last_updated": "2026-03-17 12:29",
    "price_change_1d": 0.9000000000000021,
    "price_change_pct_1d": 5.555555555555558,
    "accumulated_value": 531107880000.0,
    "avg_volume_20d": 38675390.0,
    "volume_spike_20d_pct": 80.05659412872113,
    "total_volume_avg_20d": 41615934.25,
    "deal_volume_spike_20d_pct": 0.016820480246697814,
    "deal_volume_spike_5d_20d_pct": 7.470600038253377,
    "deal_volume_sum_5d": 15544800.0,
    "deal_value_avg_5d": 53485958800.2352,
    "deal_volume_avg_5d": 3108960.0
  },
  {
    "symbol": "SSI",
    "index": "VNINDEX",
    "last_price": 29.15,
    "last_updated": "2026-03-17 12:29",
    "price_change_1d": 0.75,
    "price_change_pct_1d": 2.640845070422526,
    "accumulated_value": 501466090000.0,
    "avg_volume_20d": 45696890.0,
    "volume_spike_20d_pct": 37.617439611317096,
    "total_volume_avg_20d": 45819715.75,
    "deal_volume_spike_20d_pct": 0.0,
    "deal_volume_spike_5d_20d_pct": 0.30550254995852955,
    "deal_volume_sum_5d": 699902.0,
    "deal_value_avg_5d": 4077962280.0,
    "deal_volume_avg_5d": 139980.4
  },
  {
    "symbol": "FPT",
    "index": "VNINDEX",
    "last_price": 80.3,
    "last_updated": "2026-03-17 12:29",
    "price_change_1d": 2.0999999999999943,
    "price_change_pct_1d": 2.6854219948849067,
    "accumulated_value": 414958600000.0,
    "avg_volume_20d": 17269915.0,
    "volume_spike_20d_pct": 29.94803390752068,
    "total_volume_avg_20d": 18540556.375,
    "deal_volume_spike_20d_pct": 0.9930095746654745,
    "deal_volume_spike_5d_20d_pct": 8.981825929697862,
    "deal_volume_sum_5d": 8326402.5,
    "deal_value_avg_5d": 136837793830.46017,
    "deal_volume_avg_5d": 1665280.5
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

- Captured at: `2026-03-17T05:29:14.412970+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, index, last_price, last_updated, price_change_1d, price_change_pct_1d, accumulated_value, avg_volume_20d, volume_spike_20d_pct, total_volume_avg_20d, deal_volume_spike_20d_pct, deal_volume_spike_5d_20d_pct, deal_volume_sum_5d, deal_value_avg_5d, deal_volume_avg_5d
```
- Dtypes: `{'symbol': 'str', 'index': 'str', 'last_price': 'float64', 'last_updated': 'str', 'price_change_1d': 'float64', 'price_change_pct_1d': 'float64', 'accumulated_value': 'float64', 'avg_volume_20d': 'float64', 'volume_spike_20d_pct': 'float64', 'total_volume_avg_20d': 'float64', 'deal_volume_spike_20d_pct': 'float64', 'deal_volume_spike_5d_20d_pct': 'float64', 'deal_volume_sum_5d': 'float64', 'deal_value_avg_5d': 'float64', 'deal_volume_avg_5d': 'float64'}`

```json
[
  {
    "symbol": "APG",
    "index": "VNINDEX",
    "last_price": 6.55,
    "last_updated": "2026-03-17 12:29",
    "price_change_1d": -0.08000000000000007,
    "price_change_pct_1d": -1.2066365007541435,
    "accumulated_value": 74535602000.0,
    "avg_volume_20d": 769010.0,
    "volume_spike_20d_pct": 1558.4972887218632,
    "total_volume_avg_20d": 787210.0,
    "deal_volume_spike_20d_pct": 0.0,
    "deal_volume_spike_5d_20d_pct": 2.28655631915245,
    "deal_volume_sum_5d": 90000.0,
    "deal_value_avg_5d": 137700000.0,
    "deal_volume_avg_5d": 18000.0
  },
  {
    "symbol": "MCH",
    "index": "VNINDEX",
    "last_price": 157.2,
    "last_updated": "2026-03-17 12:29",
    "price_change_1d": 7.399999999999977,
    "price_change_pct_1d": 4.939919893190914,
    "accumulated_value": 196749260000.0,
    "avg_volume_20d": 352855.0,
    "volume_spike_20d_pct": 358.6459027079112,
    "total_volume_avg_20d": 576641.6,
    "deal_volume_spike_20d_pct": 0.0,
    "deal_volume_spike_5d_20d_pct": 0.0,
    "deal_volume_sum_5d": 0.0,
    "deal_value_avg_5d": 0.0,
    "deal_volume_avg_5d": 0.0
  },
  {
    "symbol": "NO1",
    "index": "VNINDEX",
    "last_price": 5.99,
    "last_updated": "2026-03-17 12:29",
    "price_change_1d": 0.39000000000000057,
    "price_change_pct_1d": 6.964285714285734,
    "accumulated_value": 538024000.0,
    "avg_volume_20d": 35220.0,
    "volume_spike_20d_pct": 257.8080636002271,
    "total_volume_avg_20d": 35220.0,
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
