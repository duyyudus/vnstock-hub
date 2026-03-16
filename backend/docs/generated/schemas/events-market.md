# events.market

- Class: `EventsReference`
- Method: `market`
- Signature: `(start=None, end=None, event_type=None)`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `vnstock`

Retrieve special stock market events (holidays, system incidents, ...)

## Purpose

Retrieve special stock market events (holidays, system incidents, ...)

Uses a static internal database from the vnstock library.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `str, optional` | `2025-03-01` |  | Start date (YYYY-MM-DD). Defaults to all events. |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `str, optional` | `2025-03-07` |  | End date (YYYY-MM-DD). Defaults to all events. |
| `event_type` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `str, optional` | `omitted in live probe` | `Holiday`, `Suspension`, `Compensation` | Event type ('Holiday', 'Suspension', 'Compensation'). |

## Source details

### Source `vnstock`

#### Raw output contract

- Coverage: `declared`

```text
date, event, type, duration
```

| Raw | Normalized |
| --- | --- |
| `date` | `date` |
| `event` | `event_name` |
| `type` | `event_type` |
| `duration` | `duration` |

#### Normalized output schema

- Coverage: `declared`

```text
date, event_name, event_type, duration
```

#### Live-observed sample

- Captured at: `2026-03-16T11:15:04.265500+00:00`
- Success: `True`
- Row count: `0`

```text
date, event_name, event_type, duration
```
- Dtypes: `{'date': 'datetime64[us]', 'event_name': 'str', 'event_type': 'str', 'duration': 'str'}`
