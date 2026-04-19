# company.shareholders

- Class: `CompanyReference`
- Method: `shareholders`
- Signature: `(show_log = False) -> DataFrame chứa thông tin cổ đông.`
- Return type: `DataFrame chứa thông tin cổ đông.`
- Normalization mode: `contractual`
- Supported sources: `kbs`
- Declared signature: `()`
- Default route source: `kbs`
- Default provider: `company.Company.shareholders`

Get company shareholders.

## Purpose

Get company shareholders.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` | `omitted; default False` | Hiển thị log debug. |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `shareholders`

```text
name, update_date, shares_owned, ownership_percentage
```
- Note: No source-specific raw mapping declared; using normalized columns as a best-effort approximation.

#### Normalized output schema

- Coverage: `declared`

```text
name, update_date, shares_owned, ownership_percentage
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
