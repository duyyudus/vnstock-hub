# market.fund.top_holding

- Class: `FundMarket`
- Method: `top_holding`
- Signature: `(fundId = 23)`
- Return type: `None`
- Normalization mode: `contractual`
- Supported sources: `fmarket`
- Declared signature: `(**B)`
- Default route source: `fmarket`
- Default provider: `fund.FundDetails.top_holding`

Extracts the top equity/bond holdings of the fund.

## Purpose

Extracts the top equity/bond holdings of the fund.

## Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `fundId` | `POSITIONAL_OR_KEYWORD` | `False` | `23` | `` |

## Source details

### Source `fmarket`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.fmarket.fund.Fund`
- Provider method: `top_holding`

```text
shortName, name, dataFundAssetType.name, owner.name, managementFee, firstIssueAt, nav, productNavChange.navToPrevious, productNavChange.navToLastYear, productNavChange.navToBeginning, productNavChange.navTo1Months, productNavChange.navTo3Months, productNavChange.navTo6Months, productNavChange.navTo12Months, productNavChange.navTo24Months, productNavChange.navTo36Months, productNavChange.annualizedReturn36Months, productNavChange.updateAt, id, code, vsdFeeId
```

| Raw | Normalized |
| --- | --- |
| `stock_code` | `ticker` |
| `industry` | `industry` |
| `net_asset_percent` | `net_asset_percent` |
| `type_asset` | `asset_type` |
| `update_at` | `update_at` |
- Note: Derived from `app.lib.vnstock_data_alt.explorer.fmarket.fund._FUND_LIST_COLUMNS`.

#### Normalized output schema

- Coverage: `declared`

```text
update_at, ticker, industry, net_asset_percent, asset_type
```

Enum/value normalization:

- `asset_type`: {'STOCK': 'equity', 'BOND': 'bond', 'Cổ phiếu': 'equity', 'Trái phiếu': 'bond', 'Tiền và tương đương tiền': 'cash_and_equivalents', 'Khác': 'others'}

#### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.
