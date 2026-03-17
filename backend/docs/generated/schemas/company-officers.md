# company.officers

- Class: `CompanyReference`
- Method: `officers`
- Signature: `(filter_by='working')`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `kbs, vci`
- Default route source: `vci`
- Default provider: `company.Company.officers`

Get company officers.

## Purpose

Get company officers.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `filter_by` | `POSITIONAL_OR_KEYWORD` | `False` | `working` | `str` | `all` | `working`, `resigned`, `all`, `working` | 'working', 'resigned', or 'all'. Default 'working'. |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `declared`

```text
name, position, from_date, owner_code
```

| Raw | Normalized |
| --- | --- |
| `name` | `name` |
| `position` | `position` |
| `from_date` | `from_date` |
| `owner_code` | `owner_code` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, name, position, from_date, total_shares, rate
```

#### Live-observed sample

- Captured at: `2026-03-17T05:26:47.876673+00:00`
- Success: `True`
- Row count: `53`

```text
name, position, rate
```
- Dtypes: `{'name': 'str', 'position': 'str', 'rate': 'float64'}`

```json
[
  {
    "name": "Phùng Nguyễn Hải Yến",
    "position": "Phụ trách Công bố thông tin/Phó Tổng Giám đốc",
    "rate": 6.3e-06
  },
  {
    "name": "Nguyễn Thanh Tùng",
    "position": "Phó Tổng Giám đốc",
    "rate": 2.7e-06
  },
  {
    "name": "Đào Minh Tuấn",
    "position": "Phó Tổng Giám đốc",
    "rate": 2e-06
  }
]
```

### Source `vci`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.company.Company`
- Provider method: `officers`

```text
officer_name, officer_position, officer_own_percent, officer_own_quantity
```

| Raw | Normalized |
| --- | --- |
| `officer_name` | `name` |
| `officer_position` | `position` |
| `officer_own_percent` | `rate` |
| `officer_own_quantity` | `total_shares` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, name, position, from_date, total_shares, rate
```

#### Live-observed sample

- Captured at: `2026-03-17T05:26:48.008865+00:00`
- Success: `True`
- Row count: `53`

```text
name, position, rate
```
- Dtypes: `{'name': 'str', 'position': 'str', 'rate': 'float64'}`

```json
[
  {
    "name": "Phùng Nguyễn Hải Yến",
    "position": "Phụ trách Công bố thông tin/Phó Tổng Giám đốc",
    "rate": 6.3e-06
  },
  {
    "name": "Nguyễn Thanh Tùng",
    "position": "Phó Tổng Giám đốc",
    "rate": 2.7e-06
  },
  {
    "name": "Đào Minh Tuấn",
    "position": "Phó Tổng Giám đốc",
    "rate": 2e-06
  }
]
```
