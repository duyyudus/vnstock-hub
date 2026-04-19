# company.officers

- Class: `CompanyReference`
- Method: `officers`
- Signature: `(filter_by='working')`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `kbs`
- Default route source: `kbs`
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

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `officers`

```text
from_date, position, name, position_en, owner_code
```
- Note: No source-specific raw mapping declared; using normalized columns as a best-effort approximation.

#### Normalized output schema

- Coverage: `declared`

```text
from_date, position, name, position_en, owner_code
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
