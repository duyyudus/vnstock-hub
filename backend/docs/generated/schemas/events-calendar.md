# events.calendar

- Class: `EventsReference`
- Method: `calendar`
- Signature: `(start=None, end=None, event_type=None, page=0, limit=20000)`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `vci`
- Default route source: `vci`
- Default provider: `event.Event.calendar`

Retrieve events calendar (dividends, AGM, new listings, ...) from the default data source.

## Purpose

Retrieve events calendar (dividends, AGM, new listings, ...) from the default data source.

    event_type (str, optional): Type of event or event group:
        - 'dividend': Returns dividends, share issuance
        - 'insider': Returns insider trading, major shareholders
        - 'agm': Returns shareholder meetings
        - 'others': Returns other fluctuations
        - Or a specific eventCode (e.g., 'ISS,DIV')
    page (int): Page index. Defaults to 0.
    limit (int): Number of records per page. Defaults to 20000.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `str, optional` | `2025-03-01` |  | Start date (YYYY-MM-DD). Defaults to the current date. |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `str, optional` | `2025-03-07` |  | End date (YYYY-MM-DD). Defaults to the current date. |
| `event_type` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `str, optional` | `omitted in live probe` | `dividend`, `insider`, `agm`, `others`, `ISS,DIV` | Type of event or event group: - 'dividend': Returns dividends, share issuance - 'insider': Returns insider trading, major shareholders - 'agm': Returns shareholder meetings - 'others': Returns other fluctuations - Or a specific eventCode (e.g., 'ISS,DIV') |
| `page` | `POSITIONAL_OR_KEYWORD` | `False` | `0` | `int` | `1` |  | Page index. Defaults to 0. |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `20000` | `int` | `5` |  | Number of records per page. Defaults to 20000. |

## Source details

### Source `vci`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.event.Event`
- Provider method: `calendar`

```text
ticker, event_name_vi, organ_name_vi, event_title_vi, exright_date, record_date, payout_date, public_date, issue_date, value_per_share, exercise_ratio, event_code
```

| Raw | Normalized |
| --- | --- |
| `ticker` | `symbol` |
| `event_name_vi` | `event_name` |
| `organ_name_vi` | `organ_name` |
| `event_title_vi` | `event_title` |
| `exright_date` | `ex_right_date` |
| `record_date` | `record_date` |
| `payout_date` | `payout_date` |
| `public_date` | `public_date` |
| `issue_date` | `issue_date` |
| `value_per_share` | `value` |
| `exercise_ratio` | `ratio` |
| `event_code` | `event_type` |

#### Normalized output schema

- Coverage: `declared`

```text
symbol, event_name, event_title, ex_right_date, record_date, payout_date, value, ratio, organ_name, public_date, issue_date, event_type
```

#### Live-observed sample

- Captured at: `2026-03-16T11:15:04.256171+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, event_name, event_title, organ_name, public_date, event_type
```
- Dtypes: `{'symbol': 'str', 'event_name': 'str', 'event_title': 'str', 'organ_name': 'str', 'public_date': 'datetime64[us]', 'event_type': 'str'}`

```json
[
  {
    "symbol": "VST",
    "event_name": "Giao dịch nội bộ: Giao dịch tổ chức",
    "event_title": "Công Đoàn Công ty Cổ Phần Vận Tải Và Thuê Tàu Biển Việt Nam - Đăng kí Bán 30,000 VST",
    "organ_name": "Công ty Cổ phần Vận tải và Thuê Tàu biển Việt Nam",
    "public_date": "2025-04-04T00:00:00",
    "event_type": "DDINS"
  },
  {
    "symbol": "HCB",
    "event_name": "Giao dịch nội bộ: Giao dịch cá nhân",
    "event_title": "Phạm Phú Kiệt - Đăng kí Mua HCB",
    "organ_name": "Công ty Cổ phần Dệt may 29/3",
    "public_date": "2025-03-31T00:00:00",
    "event_type": "DDIND"
  },
  {
    "symbol": "HNI",
    "event_name": "Giao dịch nội bộ: Giao dịch cá nhân",
    "event_title": "Hà Hải Ninh - Đăng kí Mua 200,000 HNI",
    "organ_name": "Công ty Cổ phần May Hữu Nghị",
    "public_date": "2025-03-28T00:00:00",
    "event_type": "DDIND"
  }
]
```
