# reference.fund.list

- Class: `FundReference`
- Method: `list`
- Signature: `(fund_type = '')`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `fmarket`
- Declared signature: `()`
- Default route source: `fmarket`
- Default provider: `fund.Fund.listing`

Extracts the list of all available mutual funds.

## Purpose

Extracts the list of all available mutual funds.

## Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `fund_type` | `POSITIONAL_OR_KEYWORD` | `False` | `` | `` |

## Source details

### Source `fmarket`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.fmarket.fund.Fund`
- Provider method: `listing`

```text
shortName, name, dataFundAssetType.name, owner.name, managementFee, firstIssueAt, nav, productNavChange.navToPrevious, productNavChange.navToLastYear, productNavChange.navToBeginning, productNavChange.navTo1Months, productNavChange.navTo3Months, productNavChange.navTo6Months, productNavChange.navTo12Months, productNavChange.navTo24Months, productNavChange.navTo36Months, productNavChange.annualizedReturn36Months, productNavChange.updateAt, id, code, vsdFeeId
```

| Raw | Normalized |
| --- | --- |
| `short_name` | `ticker` |
| `name` | `organ_name` |
| `fund_type` | `fund_type` |
| `fund_owner_name` | `organ_short_name` |
| `management_fee` | `management_fee` |
| `inception_date` | `inception_date` |
| `nav` | `nav` |
| `nav_update_at` | `nav_update_at` |
- Note: Derived from `app.lib.vnstock_data_alt.explorer.fmarket.fund._FUND_LIST_COLUMNS`.

#### Normalized output schema

- Coverage: `declared`

```text
ticker, organ_name, organ_short_name, fund_type, management_fee, inception_date, nav, nav_update_at
```

Enum/value normalization:

- `fund_type`: {'Quỹ cổ phiếu': 'equity_fund', 'Quỹ cân bằng': 'balanced_fund', 'Quỹ trái phiếu': 'bond_fund'}

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
