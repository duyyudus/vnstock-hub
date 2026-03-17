# company.shareholders

- Class: `CompanyReference`
- Method: `shareholders`
- Signature: `()`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `kbs, vci`
- Default route source: `vci`
- Default provider: `company.Company.shareholders`

Get company shareholders.

## Purpose

Get company shareholders.

## Parameters

_None._

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `declared`

```text
name, shares_owned, ownership_percentage, update_date
```

| Raw | Normalized |
| --- | --- |
| `name` | `name` |
| `shares_owned` | `total_shares` |
| `ownership_percentage` | `rate` |
| `update_date` | `date` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, name, total_shares, rate, date
```

#### Live-observed sample

- Captured at: `2026-03-17T05:26:48.360101+00:00`
- Success: `True`
- Row count: `47`

```text
name, total_shares, rate, date
```
- Dtypes: `{'name': 'str', 'total_shares': 'int64', 'rate': 'float64', 'date': 'str'}`

```json
[
  {
    "name": "Ngân Hàng Nhà Nước Việt Nam",
    "total_shares": 6250338579,
    "rate": 0.748,
    "date": "2026-02-02"
  },
  {
    "name": "Mizuho Bank Limited",
    "total_shares": 1253366534,
    "rate": 0.15,
    "date": "2025-11-21"
  },
  {
    "name": "Quỹ Đầu tư Chính phủ Singapore (GIC)",
    "total_shares": 84503639,
    "rate": 0.0101,
    "date": "2025-10-05"
  }
]
```

### Source `vci`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.company.Company`
- Provider method: `shareholders`

```text
symbol, share_holder, share_own_percent, quantity, update_date
```

| Raw | Normalized |
| --- | --- |
| `symbol` | `symbol` |
| `share_holder` | `name` |
| `share_own_percent` | `rate` |
| `quantity` | `total_shares` |
| `update_date` | `date` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, name, total_shares, rate, date
```

#### Live-observed sample

- Captured at: `2026-03-17T05:26:48.458787+00:00`
- Success: `True`
- Row count: `47`

```text
name, total_shares, rate, date
```
- Dtypes: `{'name': 'str', 'total_shares': 'int64', 'rate': 'float64', 'date': 'str'}`

```json
[
  {
    "name": "Ngân Hàng Nhà Nước Việt Nam",
    "total_shares": 6250338579,
    "rate": 0.748,
    "date": "2026-02-02"
  },
  {
    "name": "Mizuho Bank Limited",
    "total_shares": 1253366534,
    "rate": 0.15,
    "date": "2025-11-21"
  },
  {
    "name": "Quỹ Đầu tư Chính phủ Singapore (GIC)",
    "total_shares": 84503639,
    "rate": 0.0101,
    "date": "2025-10-05"
  }
]
```
