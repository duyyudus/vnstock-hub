# company.subsidiaries

- Class: `CompanyReference`
- Method: `subsidiaries`
- Signature: `(filter_by='all')`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `kbs`
- Default route source: `kbs`
- Default provider: `company.Company.subsidiaries`

Get company subsidiaries.

## Purpose

Get company subsidiaries.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `filter_by` | `POSITIONAL_OR_KEYWORD` | `False` | `all` | `str` | `all` | `all`, `subsidiary`, `affiliate`, `all` | 'all', 'subsidiary', or 'affiliate'. Default 'all'. |

## Source details

### Source `kbs`

#### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_data_alt.explorer.kbs.company.Company`
- Provider method: `subsidiaries`

```text
update_date, name, charter_capital, ownership_percent, currency, type
```
- Note: No source-specific raw mapping declared; using normalized columns as a best-effort approximation.

#### Normalized output schema

- Coverage: `declared`

```text
update_date, name, charter_capital, ownership_percent, currency, type
```

Enum/value normalization:

- `type`: {'công ty con': 'Subsidiary', 'công ty liên kết': 'Affiliate'}

#### Live-observed sample

- Captured at: `2026-03-17T05:26:48.672911+00:00`
- Success: `True`
- Row count: `12`

```text
name, rate, sub_symbol, type
```
- Dtypes: `{'name': 'str', 'rate': 'float64', 'sub_symbol': 'str', 'type': 'str'}`

```json
[
  {
    "name": "Công ty Chuyển tiền Vietcombank",
    "rate": 0.875,
    "sub_symbol": "2646966",
    "type": "Subsidiary"
  },
  {
    "name": "Ngân hàng Thương mại TNHH MTV Ngoại thương Công nghệ Số",
    "rate": 1.0,
    "sub_symbol": "TB",
    "type": "Subsidiary"
  },
  {
    "name": "Công ty TNHH Cao Ốc Vietcombank 198",
    "rate": 0.7,
    "sub_symbol": "VCB198",
    "type": "Subsidiary"
  }
]
```
