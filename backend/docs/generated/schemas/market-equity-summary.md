# market.equity.summary

- Class: `EquityMarket`
- Method: `summary`
- Signature: `(show_log = False, to_df = True) -> pd.DataFrame`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `kbs, msn`
- Declared signature: `(**B)`
- Default route source: `kbs`
- Default provider: `quote.Quote.summary`

Stock Info / Snapshot summary metrics including pricing,

## Purpose

Stock Info / Snapshot summary metrics including pricing, 
52-week ranges, and fundamental ratios.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `False` | Hiển thị log debug. |
| `to_df` | `POSITIONAL_OR_KEYWORD` | `False` | `True` | `` | `True` | Trả về DataFrame. Mặc định True. False để trả về JSON (dict chuỗi). |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.quote.Quote`
- Provider method: `summary`

```text
MAP52, MIP52, DIV, BT, EPS, BVPS, MC, PER, PBR, ROE, CMCM, CMCY, YD, FTO
```

| Raw | Normalized |
| --- | --- |
| `MAP52` | `high_52w` |
| `MIP52` | `low_52w` |
| `DIV` | `dividend` |
| `BT` | `beta` |
| `EPS` | `eps` |
| `BVPS` | `bvps` |
| `MC` | `market_cap` |
| `PER` | `pe` |
| `PBR` | `pb` |
| `ROE` | `roe` |
| `CMCM` | `change_1m` |
| `CMCY` | `change_1y` |
| `YD` | `dividend_yield` |
| `FTO` | `foreign_ownership_pct` |

#### Normalized output schema

- Coverage: `declared`

```text
high_52w, low_52w, dividend, beta, eps, bvps, market_cap, pe, pb, roe, change_1m, change_1y, dividend_yield, foreign_ownership_pct
```

#### Live-observed sample

- Captured at: `2026-03-17T05:26:58.629402+00:00`
- Success: `True`
- Row count: `1`

```text
high_52w, low_52w, dividend, beta, eps, bvps, market_cap, pe, pb, roe, change_1m, change_1y, dividend_yield, foreign_ownership_pct
```
- Dtypes: `{'high_52w': 'int64', 'low_52w': 'int64', 'dividend': 'int64', 'beta': 'float64', 'eps': 'int64', 'bvps': 'float64', 'market_cap': 'int64', 'pe': 'str', 'pb': 'str', 'roe': 'float64', 'change_1m': 'str', 'change_1y': 'str', 'dividend_yield': 'float64', 'foreign_ownership_pct': 'float64'}`

```json
[
  {
    "high_52w": 76000,
    "low_52w": 52162,
    "dividend": 450,
    "beta": 0.77,
    "eps": 4542,
    "bvps": 27231.3,
    "market_cap": 501340505640000,
    "pe": "12.95",
    "pb": "2.16",
    "roe": 16.61,
    "change_1m": "-6.83",
    "change_1y": "-10.27",
    "dividend_yield": 0.0075,
    "foreign_ownership_pct": 20.537
  }
]
```

### Source `msn`

#### Raw output contract

- Coverage: `declared`

```text
MAP52, MIP52, DIV, BT, EPS, BVPS, MC, PER, PBR, ROE, CMCM, CMCY, YD, FTO
```

| Raw | Normalized |
| --- | --- |
| `MAP52` | `high_52w` |
| `MIP52` | `low_52w` |
| `DIV` | `dividend` |
| `BT` | `beta` |
| `EPS` | `eps` |
| `BVPS` | `bvps` |
| `MC` | `market_cap` |
| `PER` | `pe` |
| `PBR` | `pb` |
| `ROE` | `roe` |
| `CMCM` | `change_1m` |
| `CMCY` | `change_1y` |
| `YD` | `dividend_yield` |
| `FTO` | `foreign_ownership_pct` |

#### Normalized output schema

- Coverage: `declared`

```text
high_52w, low_52w, dividend, beta, eps, bvps, market_cap, pe, pb, roe, change_1m, change_1y, dividend_yield, foreign_ownership_pct
```

#### Live-observed sample

- Captured at: `2026-03-17T05:26:59.279688+00:00`
- Success: `True`
- Row count: `1`

```text
high_52w, low_52w, dividend, beta, eps, bvps, market_cap, pe, pb, roe, change_1m, change_1y, dividend_yield, foreign_ownership_pct
```
- Dtypes: `{'high_52w': 'int64', 'low_52w': 'int64', 'dividend': 'int64', 'beta': 'float64', 'eps': 'int64', 'bvps': 'float64', 'market_cap': 'int64', 'pe': 'str', 'pb': 'str', 'roe': 'float64', 'change_1m': 'str', 'change_1y': 'str', 'dividend_yield': 'float64', 'foreign_ownership_pct': 'float64'}`

```json
[
  {
    "high_52w": 76000,
    "low_52w": 52162,
    "dividend": 450,
    "beta": 0.77,
    "eps": 4542,
    "bvps": 27231.3,
    "market_cap": 501340505640000,
    "pe": "12.95",
    "pb": "2.16",
    "roe": 16.61,
    "change_1m": "-6.83",
    "change_1y": "-10.27",
    "dividend_yield": 0.0075,
    "foreign_ownership_pct": 20.537
  }
]
```
