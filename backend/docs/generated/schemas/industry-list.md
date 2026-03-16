# industry.list

- Class: `IndustryReference`
- Method: `list`
- Signature: `(lang='vi')`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `vci`
- Default route source: `vci`
- Default provider: `listing.Listing.industries_icb`

List ICB industry classifications for all symbols in the market.

## Purpose

List ICB industry classifications for all symbols in the market.
Uses VCI as the data source over KBS because VCI provides deeper ICB levels (up to 4 levels).

Note: The underlying data source may not be available in all environments 
(e.g., might be blocked on Google Colab).

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lang` | `POSITIONAL_OR_KEYWORD` | `False` | `vi` | `str` | `vi` | `vi`, `en`, `vi` | Language code 'vi' or 'en'. Default 'vi'. |

## Source details

### Source `vci`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.listing.Listing`
- Provider method: `industries_icb`

```text
icb_code, icb_name, level
```

| Raw | Normalized |
| --- | --- |
| `icb_code` | `icb_code` |
| `icb_name` | `icb_name` |
| `level` | `icb_level` |

#### Normalized output schema

- Coverage: `declared`

```text
icb_code, icb_name, icb_name_en, icb_level
```

#### Live-observed sample

- Captured at: `2026-03-16T11:15:04.398488+00:00`
- Success: `True`
- Row count: `155`

```text
icb_code, icb_name, icb_level
```
- Dtypes: `{'icb_code': 'str', 'icb_name': 'str', 'icb_level': 'int64'}`

```json
[
  {
    "icb_code": "0530",
    "icb_name": "Sản xuất Dầu khí",
    "icb_level": 3
  },
  {
    "icb_code": "0570",
    "icb_name": "Thiết bị, Dịch vụ và Phân phối Dầu khí",
    "icb_level": 3
  },
  {
    "icb_code": "1350",
    "icb_name": "Hóa chất",
    "icb_level": 3
  }
]
```
