# Finance

- Qualified name: `app.lib.vnstock_alt.api.financial.Finance`
- Signature: `(source: str, symbol: str, period: str = 'quarter', get_all: bool = True, show_log: bool = False)`
- Supported sources: `kbs, vci`

Base adapter that uses ProviderRegistry to discover and instantiate

## Purpose

Base adapter that uses ProviderRegistry to discover and instantiate
providers from both explorer and connector packages.

## Members

### balance_sheet

- Kind: `method`
- Signature: `(period: str = 'year', display_mode: Union[str, app.lib.vnstock_alt.explorer.kbs.financial.FieldDisplayMode, NoneType] = "<FieldDisplayMode.STD: 'std'>", show_log: Optional[bool] = False) -> Any`
- Declared signature: `(*args: Any, **kwargs: Any) -> Any`
- Effective signature source: provider `kbs`
- Return type: `Any`
- Purpose: Retrieve balance sheet data.

#### Parameters

| Name | Kind | Required | Default | Annotation | Example | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `year` | `str` |  | `year` | `year`, `quarter`, `year` | Loại kỳ báo cáo ('year' hoặc 'quarter'). Mặc định 'year'. |
| `display_mode` | `POSITIONAL_OR_KEYWORD` | `False` | `<FieldDisplayMode.STD: 'std'>` | `Union[str, app.lib.vnstock_alt.explorer.kbs.financial.FieldDisplayMode, NoneType]` | `FieldDisplayMode.STD` | `omitted; default "<FieldDisplayMode.STD: 'std'>"` | `item`, `item_id`, `vi`, `en` | Chế độ hiển thị trường dữ liệu. Mặc định FieldDisplayMode.STD. - FieldDisplayMode.STD: Chỉ giữ cột 'item' và 'item_id' (đã chuẩn hóa) - FieldDisplayMode.ALL: Giữ tất cả cột item (item, item_en, item_id) - 'vi': Chỉ giữ tên tiếng Việt (tương thích ngược) - 'en': Chỉ giữ tên tiếng Anh (tương thích ngược) - None: Giữ tất cả cột (tương thích ngược) |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `Optional[bool]` |  | `False` |  | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.kbs.financial.Finance`
- Provider method: `balance_sheet`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:16:28.861153+00:00`
- Success: `True`
- Row count: `77`

```text
item, item_id, 2025, 2024, 2023, 2022
```
- Dtypes: `{'item': 'str', 'item_id': 'str', '2025': 'float64', '2024': 'float64', '2023': 'float64', '2022': 'float64'}`

```json
[
  {
    "item": "A. TÀI SẢN",
    "item_id": "assets",
    "2025": NaN,
    "2024": NaN,
    "2023": NaN,
    "2022": NaN
  },
  {
    "item": "I. Tiền mặt, vàng bạc, đá quý",
    "item_id": "i.cash_gold_and_silver_precious_stones",
    "2025": 15542768000.0,
    "2024": 14268064000.0,
    "2023": 14504849000.0,
    "2022": 18348534000.0
  },
  {
    "item": "II. Tiền gửi tại NHNN",
    "item_id": "ii.balances_with_the_state_bank_of_vietnam",
    "2025": 37445504000.0,
    "2024": 49340493000.0,
    "2023": 58104503000.0,
    "2022": 92557809000.0
  }
]
```

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.vci.financial.Finance`
- Provider method: `balance_sheet`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:16:30.461616+00:00`
- Success: `True`
- Row count: `13`

```text
ticker, yearReport, TOTAL ASSETS (Bn. VND), Cash and cash equivalents (Bn. VND), Balances with the SBV, Placements with and loans to other credit institutions, Trading Securities, net, Trading Securities, Provision for diminution in value of Trading Securities, Derivatives and other financial liabilities, Loans and advances to customers, net, Loans and advances to customers, Less: Provision for losses on loans and advances to customers, Investment Securities, Available-for Sales Securities, Held-to-Maturity Securities, Less: Provision for diminution in value of investment securities, Long-term investments (Bn. VND), Investments in associate companies, Other long-term assets (Bn. VND), Less: Provision for diminuation in value of long term investments, Fixed assets (Bn. VND), Tangible fixed assets, Intagible fixed assets, Other Assets, TOTAL RESOURCES (Bn. VND), LIABILITIES (Bn. VND), Due to Gov and borrowings from SBV, Deposits and borrowings from other credit institutions, Deposits from customers, _Derivatives and other financial liabilities, Funds received from Gov, international and other institutions, Convertible bonds/CDs and other valuable papers issued, Other liabilities, OWNER'S EQUITY(Bn.VND), Capital, Reserves, Foreign Currency Difference reserve, Difference upon Assets Revaluation, Undistributed earnings (Bn. VND), Paid-in capital (Bn. VND), Other Reserves, MINORITY INTERESTS
```
- Dtypes: `{'ticker': 'str', 'yearReport': 'int64', 'TOTAL ASSETS (Bn. VND)': 'int64', 'Cash and cash equivalents (Bn. VND)': 'int64', 'Balances with the SBV': 'int64', 'Placements with and loans to other credit institutions': 'int64', 'Trading Securities, net': 'int64', 'Trading Securities': 'int64', 'Provision for diminution in value of Trading Securities': 'int64', 'Derivatives and other financial liabilities': 'int64', 'Loans and advances to customers, net': 'int64', 'Loans and advances to customers': 'int64', 'Less: Provision for losses on loans and advances to customers': 'int64', 'Investment Securities': 'int64', 'Available-for Sales Securities': 'int64', 'Held-to-Maturity Securities': 'int64', 'Less: Provision for diminution in value of investment securities': 'int64', 'Long-term investments (Bn. VND)': 'int64', 'Investments in associate companies': 'int64', 'Other long-term assets (Bn. VND)': 'int64', 'Less: Provision for diminuation in value of long term investments': 'int64', 'Fixed assets (Bn. VND)': 'int64', 'Tangible fixed assets': 'int64', 'Intagible fixed assets': 'int64', 'Other Assets': 'int64', 'TOTAL RESOURCES (Bn. VND)': 'int64', 'LIABILITIES (Bn. VND)': 'int64', 'Due to Gov and borrowings from SBV': 'int64', 'Deposits and borrowings from other credit institutions': 'int64', 'Deposits from customers': 'int64', '_Derivatives and other financial liabilities': 'int64', 'Funds received from Gov, international and other institutions': 'int64', 'Convertible bonds/CDs and other valuable papers issued': 'int64', 'Other liabilities': 'int64', "OWNER'S EQUITY(Bn.VND)": 'int64', 'Capital': 'int64', 'Reserves': 'int64', 'Foreign Currency Difference reserve': 'int64', 'Difference upon Assets Revaluation': 'int64', 'Undistributed earnings (Bn. VND)': 'int64', 'Paid-in capital (Bn. VND)': 'int64', 'Other Reserves': 'int64', 'MINORITY INTERESTS': 'int64'}`

```json
[
  {
    "ticker": "VCB",
    "yearReport": 2025,
    "TOTAL ASSETS (Bn. VND)": 2441928945000000,
    "Cash and cash equivalents (Bn. VND)": 15542768000000,
    "Balances with the SBV": 37445504000000,
    "Placements with and loans to other credit institutions": 521938509000000,
    "Trading Securities, net": 11479097000000,
    "Trading Securities": 11546520000000,
    "Provision for diminution in value of Trading Securities": -67423000000,
    "Derivatives and other financial liabilities": 374918000000,
    "Loans and advances to customers, net": 1648557141000000,
    "Loans and advances to customers": 1673525675000000,
    "Less: Provision for losses on loans and advances to customers": -24968534000000,
    "Investment Securities": 162104164000000,
    "Available-for Sales Securities": 143080817000000,
    "Held-to-Maturity Securities": 22384962000000,
    "Less: Provision for diminution in value of investment securities": -3361615000000,
    "Long-term investments (Bn. VND)": 2260728000000,
    "Investments in associate companies": 746639000000,
    "Other long-term assets (Bn. VND)": 1589089000000,
    "Less: Provision for diminuation in value of long term investments": -75000000000,
    "Fixed assets (Bn. VND)": 8232904000000,
    "Tangible fixed assets": 5618792000000,
    "Intagible fixed assets": 2614112000000,
    "Other Assets": 33993212000000,
    "TOTAL RESOURCES (Bn. VND)": 2441928945000000,
    "LIABILITIES (Bn. VND)": 2214393069000000,
    "Due to Gov and borrowings from SBV": 160128325000000,
    "Deposits and borrowings from other credit institutions": 321158844000000,
    "Deposits from customers": 1672534103000000,
    "_Derivatives and other financial liabilities": 0,
    "Funds received from Gov, international and other institutions": 0,
    "Convertible bonds/CDs and other valuable papers issued": 27101221000000,
    "Other liabilities": 33470576000000,
    "OWNER'S EQUITY(Bn.VND)": 227535876000000,
    "Capital": 89361977000000,
    "Reserves": 36993479000000,
    "Foreign Currency Difference reserve": -918673000000,
    "Difference upon Assets Revaluation": 0,
    "Undistributed earnings (Bn. VND)": 102027572000000,
    "Paid-in capital (Bn. VND)": 83556751000000,
    "Other Reserves": 809837000000,
    "MINORITY INTERESTS": 71521000000
  },
  {
    "ticker": "VCB",
    "yearReport": 2024,
    "TOTAL ASSETS (Bn. VND)": 2085873522000000,
    "Cash and cash equivalents (Bn. VND)": 14268064000000,
    "Balances with the SBV": 49340493000000,
    "Placements with and loans to other credit institutions": 389951898000000,
    "Trading Securities, net": 4876237000000,
    "Trading Securities": 4908527000000,
    "Provision for diminution in value of Trading Securities": -32290000000,
    "Derivatives and other financial liabilities": 1314434000000,
    "Loans and advances to customers, net": 1418015724000000,
    "Loans and advances to customers": 1449198899000000,
    "Less: Provision for losses on loans and advances to customers": -31183175000000,
    "Investment Securities": 167383349000000,
    "Available-for Sales Securities": 86799901000000,
    "Held-to-Maturity Securities": 80829540000000,
    "Less: Provision for diminution in value of investment securities": -246092000000,
    "Long-term investments (Bn. VND)": 2228098000000,
    "Investments in associate companies": 774176000000,
    "Other long-term assets (Bn. VND)": 1528922000000,
    "Less: Provision for diminuation in value of long term investments": -75000000000,
    "Fixed assets (Bn. VND)": 8092877000000,
    "Tangible fixed assets": 5530579000000,
    "Intagible fixed assets": 2562298000000,
    "Other Assets": 30402348000000,
    "TOTAL RESOURCES (Bn. VND)": 2085873522000000,
    "LIABILITIES (Bn. VND)": 1889664354000000,
    "Due to Gov and borrowings from SBV": 78237337000000,
    "Deposits and borrowings from other credit institutions": 234533958000000,
    "Deposits from customers": 1514664850000000,
    "_Derivatives and other financial liabilities": 0,
    "Funds received from Gov, international and other institutions": 529000000,
    "Convertible bonds/CDs and other valuable papers issued": 24125059000000,
    "Other liabilities": 38102621000000,
    "OWNER'S EQUITY(Bn.VND)": 196209168000000,
    "Capital": 61696139000000,
    "Reserves": 37052974000000,
    "Foreign Currency Difference reserve": -968292000000,
    "Difference upon Assets Revaluation": 0,
    "Undistributed earnings (Bn. VND)": 98332086000000,
    "Paid-in capital (Bn. VND)": 55890913000000,
    "Other Reserves": 809837000000,
    "MINORITY INTERESTS": 96261000000
  },
  {
    "ticker": "VCB",
    "yearReport": 2023,
    "TOTAL ASSETS (Bn. VND)": 1839724560000000,
    "Cash and cash equivalents (Bn. VND)": 14504849000000,
    "Balances with the SBV": 58104503000000,
    "Placements with and loans to other credit institutions": 336501657000000,
    "Trading Securities, net": 2495408000000,
    "Trading Securities": 2511395000000,
    "Provision for diminution in value of Trading Securities": -15987000000,
    "Derivatives and other financial liabilities": 0,
    "Loans and advances to customers, net": 1241677211000000,
    "Loans and advances to customers": 1270359018000000,
    "Less: Provision for losses on loans and advances to customers": -28681807000000,
    "Investment Securities": 145780067000000,
    "Available-for Sales Securities": 67882480000000,
    "Held-to-Maturity Securities": 78009747000000,
    "Less: Provision for diminution in value of investment securities": -112160000000,
    "Long-term investments (Bn. VND)": 2224945000000,
    "Investments in associate companies": 838225000000,
    "Other long-term assets (Bn. VND)": 1529145000000,
    "Less: Provision for diminuation in value of long term investments": -142425000000,
    "Fixed assets (Bn. VND)": 7805080000000,
    "Tangible fixed assets": 5212804000000,
    "Intagible fixed assets": 2592276000000,
    "Other Assets": 30630840000000,
    "TOTAL RESOURCES (Bn. VND)": 1839724560000000,
    "LIABILITIES (Bn. VND)": 1674644070000000,
    "Due to Gov and borrowings from SBV": 1670837000000,
    "Deposits and borrowings from other credit institutions": 213838980000000,
    "Deposits from customers": 1395697611000000,
    "_Derivatives and other financial liabilities": 117752000000,
    "Funds received from Gov, international and other institutions": 365000000,
    "Convertible bonds/CDs and other valuable papers issued": 19912623000000,
    "Other liabilities": 43405902000000,
    "OWNER'S EQUITY(Bn.VND)": 165080490000000,
    "Capital": 61696139000000,
    "Reserves": 27447116000000,
    "Foreign Currency Difference reserve": -983237000000,
    "Difference upon Assets Revaluation": 0,
    "Undistributed earnings (Bn. VND)": 76826482000000,
    "Paid-in capital (Bn. VND)": 55890913000000,
    "Other Reserves": 809837000000,
    "MINORITY INTERESTS": 93990000000
  }
]
```

#### Notes / caveats

Retrieve balance sheet data.

### cash_flow

- Kind: `method`
- Signature: `(period: str = 'year', display_mode: Union[str, app.lib.vnstock_alt.explorer.kbs.financial.FieldDisplayMode, NoneType] = "<FieldDisplayMode.STD: 'std'>", show_log: Optional[bool] = False) -> Any`
- Declared signature: `(*args: Any, **kwargs: Any) -> Any`
- Effective signature source: provider `kbs`
- Return type: `Any`
- Purpose: Retrieve cash flow data.

#### Parameters

| Name | Kind | Required | Default | Annotation | Example | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `year` | `str` |  | `year` | `year`, `quarter`, `year` | Loại kỳ báo cáo ('year' hoặc 'quarter'). Mặc định 'year'. |
| `display_mode` | `POSITIONAL_OR_KEYWORD` | `False` | `<FieldDisplayMode.STD: 'std'>` | `Union[str, app.lib.vnstock_alt.explorer.kbs.financial.FieldDisplayMode, NoneType]` | `FieldDisplayMode.STD` | `omitted; default "<FieldDisplayMode.STD: 'std'>"` | `item`, `item_id`, `vi`, `en` | Chế độ hiển thị trường dữ liệu. Mặc định FieldDisplayMode.STD. - FieldDisplayMode.STD: Chỉ giữ cột 'item' và 'item_id' (đã chuẩn hóa) - FieldDisplayMode.ALL: Giữ tất cả cột item (item, item_en, item_id) - 'vi': Chỉ giữ tên tiếng Việt (tương thích ngược) - 'en': Chỉ giữ tên tiếng Anh (tương thích ngược) - None: Giữ tất cả cột (tương thích ngược) |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `Optional[bool]` |  | `False` |  | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.kbs.financial.Finance`
- Provider method: `cash_flow`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:16:31.426190+00:00`
- Success: `True`
- Row count: `50`

```text
item, item_id, 2025, 2024, 2023, 2022
```
- Dtypes: `{'item': 'str', 'item_id': 'str', '2025': 'float64', '2024': 'float64', '2023': 'float64', '2022': 'float64'}`

```json
[
  {
    "item": "Lưu chuyển tiền từ hoạt động kinh doanh",
    "item_id": "i.cash_flows_from_operating_activities",
    "2025": NaN,
    "2024": NaN,
    "2023": NaN,
    "2022": NaN
  },
  {
    "item": "1. Thu nhập lãi và các khoản thu nhập tương tự nhận được",
    "item_id": "n_1.receipts_from_interest_and_similar_income",
    "2025": 104279621000.0,
    "2024": 93772270000.0,
    "2023": 108115649000.0,
    "2022": 86084771000.0
  },
  {
    "item": "2. Chi phí lãi và các chi phí tương tự đã trả ",
    "item_id": "n_2.payments_for_interest_and_similar_expenses",
    "2025": -44982266000.0,
    "2024": -43790244000.0,
    "2023": -47454819000.0,
    "2022": -31709129000.0
  }
]
```

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.vci.financial.Finance`
- Provider method: `cash_flow`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:16:32.229923+00:00`
- Success: `True`
- Row count: `13`

```text
ticker, yearReport, Profits from other activities, Operating profit before changes in working capital, Net Cash Flows from Operating Activities before BIT, Payment from reserves, Purchase of fixed assets, Gain on Dividend, Net Cash Flows from Investing Activities, Increase in charter captial, Cash flows from financial activities, Net increase/decrease in cash and cash equivalents, Cash and cash equivalents, Cash and Cash Equivalents at the end of period, Net cash inflows/outflows from operating activities, Proceeds from disposal of fixed assets, Investment in other entities, Proceeds from divestment in other entities, Dividends paid
```
- Dtypes: `{'ticker': 'str', 'yearReport': 'int64', 'Profits from other activities': 'int64', 'Operating profit before changes in working capital': 'int64', 'Net Cash Flows from Operating Activities before BIT': 'int64', 'Payment from reserves': 'int64', 'Purchase of fixed assets': 'int64', 'Gain on Dividend': 'int64', 'Net Cash Flows from Investing Activities': 'int64', 'Increase in charter captial': 'int64', 'Cash flows from financial activities': 'int64', 'Net increase/decrease in cash and cash equivalents': 'int64', 'Cash and cash equivalents': 'int64', 'Cash and Cash Equivalents at the end of period': 'int64', 'Net cash inflows/outflows from operating activities': 'int64', 'Proceeds from disposal of fixed assets': 'int64', 'Investment in other entities': 'int64', 'Proceeds from divestment in other entities': 'int64', 'Dividends paid': 'int64'}`

```json
[
  {
    "ticker": "VCB",
    "yearReport": 2025,
    "Profits from other activities": -340406000000,
    "Operating profit before changes in working capital": 38322527000000,
    "Net Cash Flows from Operating Activities before BIT": 117966268000000,
    "Payment from reserves": -2625051000000,
    "Purchase of fixed assets": -1453488000000,
    "Gain on Dividend": 118576000000,
    "Net Cash Flows from Investing Activities": -1379136000000,
    "Increase in charter captial": 0,
    "Cash flows from financial activities": -3776798000000,
    "Net increase/decrease in cash and cash equivalents": 110185283000000,
    "Cash and cash equivalents": 430614185000000,
    "Cash and Cash Equivalents at the end of period": 540799468000000,
    "Net cash inflows/outflows from operating activities": 115341217000000,
    "Proceeds from disposal of fixed assets": 17231000000,
    "Investment in other entities": -60167000000,
    "Proceeds from divestment in other entities": 0,
    "Dividends paid": -3776798000000
  },
  {
    "ticker": "VCB",
    "yearReport": 2024,
    "Profits from other activities": -1390558000000,
    "Operating profit before changes in working capital": 30183878000000,
    "Net Cash Flows from Operating Activities before BIT": 61999557000000,
    "Payment from reserves": -2876726000000,
    "Purchase of fixed assets": -1480121000000,
    "Gain on Dividend": 160709000000,
    "Net Cash Flows from Investing Activities": -1307413000000,
    "Increase in charter captial": 0,
    "Cash flows from financial activities": -19963000000,
    "Net increase/decrease in cash and cash equivalents": 57795455000000,
    "Cash and cash equivalents": 372818730000000,
    "Cash and Cash Equivalents at the end of period": 430614185000000,
    "Net cash inflows/outflows from operating activities": 59122831000000,
    "Proceeds from disposal of fixed assets": 12504000000,
    "Investment in other entities": 0,
    "Proceeds from divestment in other entities": 747000000,
    "Dividends paid": -19963000000
  },
  {
    "ticker": "VCB",
    "yearReport": 2023,
    "Profits from other activities": 179388000000,
    "Operating profit before changes in working capital": 43370999000000,
    "Net Cash Flows from Operating Activities before BIT": -35738696000000,
    "Payment from reserves": -2802834000000,
    "Purchase of fixed assets": -1008160000000,
    "Gain on Dividend": 146088000000,
    "Net Cash Flows from Investing Activities": -859407000000,
    "Increase in charter captial": 0,
    "Cash flows from financial activities": -15627000000,
    "Net increase/decrease in cash and cash equivalents": -39416564000000,
    "Cash and cash equivalents": 412235294000000,
    "Cash and Cash Equivalents at the end of period": 372818730000000,
    "Net cash inflows/outflows from operating activities": -38541530000000,
    "Proceeds from disposal of fixed assets": 9435000000,
    "Investment in other entities": 0,
    "Proceeds from divestment in other entities": 0,
    "Dividends paid": -15627000000
  }
]
```

#### Notes / caveats

Retrieve cash flow data.

### history

- Kind: `method`
- Signature: `(*args, **kwargs)`

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `args` | `VAR_POSITIONAL` | `True` | `None` | `` |
| `kwargs` | `VAR_KEYWORD` | `True` | `None` | `` |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `history` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

##### Source `vci`

###### Raw output contract

- Coverage: `not-available`

_No raw columns derived for this source._
- Note: Provider does not expose `history` for this source.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

_No live sample is attached to this exact endpoint yet._

Live samples come from both explicit probes in `backend/docs/live_probe_manifest.json` and auto-generated per-source probes. If a source still has no sample here, that source either failed during capture or is not currently probeable with the default inputs.

### income_statement

- Kind: `method`
- Signature: `(period: str = 'year', display_mode: Union[str, app.lib.vnstock_alt.explorer.kbs.financial.FieldDisplayMode, NoneType] = "<FieldDisplayMode.STD: 'std'>", show_log: Optional[bool] = False) -> Any`
- Declared signature: `(*args: Any, **kwargs: Any) -> Any`
- Effective signature source: provider `kbs`
- Return type: `Any`
- Purpose: Retrieve income statement data.

#### Parameters

| Name | Kind | Required | Default | Annotation | Example | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `year` | `str` |  | `year` | `year`, `quarter`, `year` | Loại kỳ báo cáo ('year' hoặc 'quarter'). Mặc định 'year'. |
| `display_mode` | `POSITIONAL_OR_KEYWORD` | `False` | `<FieldDisplayMode.STD: 'std'>` | `Union[str, app.lib.vnstock_alt.explorer.kbs.financial.FieldDisplayMode, NoneType]` | `FieldDisplayMode.STD` | `omitted; default "<FieldDisplayMode.STD: 'std'>"` | `item`, `item_id`, `vi`, `en` | Chế độ hiển thị trường dữ liệu. Mặc định FieldDisplayMode.STD. - FieldDisplayMode.STD: Chỉ giữ cột 'item' và 'item_id' (đã chuẩn hóa) - FieldDisplayMode.ALL: Giữ tất cả cột item (item, item_en, item_id) - 'vi': Chỉ giữ tên tiếng Việt (tương thích ngược) - 'en': Chỉ giữ tên tiếng Anh (tương thích ngược) - None: Giữ tất cả cột (tương thích ngược) |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `Optional[bool]` |  | `False` |  | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.kbs.financial.Finance`
- Provider method: `income_statement`

```text
item, item_id, unit, periods
```
- Note: Derived from provider docstring column hints.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:16:41.110397+00:00`
- Success: `True`
- Row count: `24`

```text
item, item_id, 2025, 2024, 2023, 2022
```
- Dtypes: `{'item': 'str', 'item_id': 'str', '2025': 'float64', '2024': 'float64', '2023': 'float64', '2022': 'float64'}`

```json
[
  {
    "item": "1. Thu nhập lãi và các khoản thu nhập tương tự",
    "item_id": "n_1.interest_income_and_similar_income",
    "2025": 105119449000.0,
    "2024": 93654841000.0,
    "2023": 108122278000.0,
    "2022": 88112700000.0
  },
  {
    "item": "2. Chi phí lãi và các chi phí tương tự",
    "item_id": "n_2.interest_expense_and_similar_expenses",
    "2025": 46445074000.0,
    "2024": 38249106000.0,
    "2023": 54501409000.0,
    "2022": 34866222000.0
  },
  {
    "item": "I. Thu nhập lãi thuần",
    "item_id": "i.net_interest_income",
    "2025": 58674375000.0,
    "2024": 55405735000.0,
    "2023": 53620869000.0,
    "2022": 53246478000.0
  }
]
```

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.vci.financial.Finance`
- Provider method: `income_statement`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:16:41.718989+00:00`
- Success: `True`
- Row count: `13`

```text
ticker, yearReport, Revenue (Bn. VND), Revenue YoY (%), Attribute to parent company (Bn. VND), Attribute to parent company YoY (%), Interest and Similar Income, Interest and Similar Expenses, Net Interest Income, Fees and Comission Income, Fees and Comission Expenses, Net Fee and Commission Income, Net gain (loss) from foreign currency and gold dealings, Net gain (loss) from trading of trading securities, Net gain (loss) from disposal of investment securities, Net Other income/(expenses), Other expenses, Net Other income/expenses, Dividends received, Total operating revenue, General & Admin Expenses, Operating Profit before Provision, Provision for credit losses, Profit before tax, Tax For the Year, Business income tax - current, Business income tax - deferred, Minority Interest, Net Profit For the Year, Attributable to parent company, EPS_basis
```
- Dtypes: `{'ticker': 'str', 'yearReport': 'int64', 'Revenue (Bn. VND)': 'int64', 'Revenue YoY (%)': 'float64', 'Attribute to parent company (Bn. VND)': 'int64', 'Attribute to parent company YoY (%)': 'float64', 'Interest and Similar Income': 'int64', 'Interest and Similar Expenses': 'int64', 'Net Interest Income': 'int64', 'Fees and Comission Income': 'int64', 'Fees and Comission Expenses': 'int64', 'Net Fee and Commission Income': 'int64', 'Net gain (loss) from foreign currency and gold dealings': 'int64', 'Net gain (loss) from trading of trading securities': 'int64', 'Net gain (loss) from disposal of investment securities': 'int64', 'Net Other income/(expenses)': 'int64', 'Other expenses': 'int64', 'Net Other income/expenses': 'int64', 'Dividends received': 'int64', 'Total operating revenue': 'int64', 'General & Admin Expenses': 'int64', 'Operating Profit before Provision': 'int64', 'Provision for credit losses': 'int64', 'Profit before tax': 'int64', 'Tax For the Year': 'int64', 'Business income tax - current': 'int64', 'Business income tax - deferred': 'int64', 'Minority Interest': 'int64', 'Net Profit For the Year': 'int64', 'Attributable to parent company': 'int64', 'EPS_basis': 'int64'}`

```json
[
  {
    "ticker": "VCB",
    "yearReport": 2025,
    "Revenue (Bn. VND)": 105119449000000,
    "Revenue YoY (%)": 0.12241340519706824,
    "Attribute to parent company (Bn. VND)": 35178155000000,
    "Attribute to parent company YoY (%)": 0.039808271526327654,
    "Interest and Similar Income": 105119449000000,
    "Interest and Similar Expenses": -46445074000000,
    "Net Interest Income": 58674375000000,
    "Fees and Comission Income": 11854531000000,
    "Fees and Comission Expenses": -8384664000000,
    "Net Fee and Commission Income": 3469867000000,
    "Net gain (loss) from foreign currency and gold dealings": 6165112000000,
    "Net gain (loss) from trading of trading securities": 171160000000,
    "Net gain (loss) from disposal of investment securities": 3616000000,
    "Net Other income/(expenses)": 5269106000000,
    "Other expenses": -1677513000000,
    "Net Other income/expenses": 3591593000000,
    "Dividends received": 281863000000,
    "Total operating revenue": 72357586000000,
    "General & Admin Expenses": -25152290000000,
    "Operating Profit before Provision": 47205296000000,
    "Provision for credit losses": -3185040000000,
    "Profit before tax": 44020256000000,
    "Tax For the Year": -8821823000000,
    "Business income tax - current": -7843123000000,
    "Business income tax - deferred": -978700000000,
    "Minority Interest": -20278000000,
    "Net Profit For the Year": 35198433000000,
    "Attributable to parent company": 35178155000000,
    "EPS_basis": 4210
  },
  {
    "ticker": "VCB",
    "yearReport": 2024,
    "Revenue (Bn. VND)": 93654841000000,
    "Revenue YoY (%)": -0.13375467461567148,
    "Attribute to parent company (Bn. VND)": 33831386000000,
    "Attribute to parent company YoY (%)": 0.022064541985358944,
    "Interest and Similar Income": 93654841000000,
    "Interest and Similar Expenses": -38249106000000,
    "Net Interest Income": 55405735000000,
    "Fees and Comission Income": 13143005000000,
    "Fees and Comission Expenses": -8006444000000,
    "Net Fee and Commission Income": 5136561000000,
    "Net gain (loss) from foreign currency and gold dealings": 5291751000000,
    "Net gain (loss) from trading of trading securities": 62123000000,
    "Net gain (loss) from disposal of investment securities": 3444000000,
    "Net Other income/(expenses)": 4468806000000,
    "Other expenses": -2097103000000,
    "Net Other income/expenses": 2371703000000,
    "Dividends received": 307179000000,
    "Total operating revenue": 68578496000000,
    "General & Admin Expenses": -23027363000000,
    "Operating Profit before Provision": 45551133000000,
    "Provision for credit losses": -3314998000000,
    "Profit before tax": 42236135000000,
    "Tax For the Year": -8383018000000,
    "Business income tax - current": -8526496000000,
    "Business income tax - deferred": 143478000000,
    "Minority Interest": -21731000000,
    "Net Profit For the Year": 33853117000000,
    "Attributable to parent company": 33831386000000,
    "EPS_basis": 5571
  },
  {
    "ticker": "VCB",
    "yearReport": 2023,
    "Revenue (Bn. VND)": 108115840000000,
    "Revenue YoY (%)": 0.22701767168637438,
    "Attribute to parent company (Bn. VND)": 33101027000000,
    "Attribute to parent company YoY (%)": 0.10709426738955338,
    "Interest and Similar Income": 108115840000000,
    "Interest and Similar Expenses": -54501409000000,
    "Net Interest Income": 53614431000000,
    "Fees and Comission Income": 12698083000000,
    "Fees and Comission Expenses": -6872673000000,
    "Net Fee and Commission Income": 5825410000000,
    "Net gain (loss) from foreign currency and gold dealings": 5660028000000,
    "Net gain (loss) from trading of trading securities": 124539000000,
    "Net gain (loss) from disposal of investment securities": 0,
    "Net Other income/(expenses)": 4051437000000,
    "Other expenses": -1777975000000,
    "Net Other income/expenses": 2273462000000,
    "Dividends received": 266456000000,
    "Total operating revenue": 67764326000000,
    "General & Admin Expenses": -21905912000000,
    "Operating Profit before Provision": 45858414000000,
    "Provision for credit losses": -4529947000000,
    "Profit before tax": 41328467000000,
    "Tax For the Year": -8206195000000,
    "Business income tax - current": -8096357000000,
    "Business income tax - deferred": -109838000000,
    "Minority Interest": -21245000000,
    "Net Profit For the Year": 33122272000000,
    "Attributable to parent company": 33101027000000,
    "EPS_basis": 5462
  }
]
```

#### Notes / caveats

Retrieve income statement data.

### ratio

- Kind: `method`
- Signature: `(period: str = 'year', display_mode: Union[str, app.lib.vnstock_alt.explorer.kbs.financial.FieldDisplayMode, NoneType] = "<FieldDisplayMode.STD: 'std'>", show_log: Optional[bool] = False) -> Any`
- Declared signature: `(*args: Any, **kwargs: Any) -> Any`
- Effective signature source: provider `kbs`
- Return type: `Any`
- Purpose: Retrieve financial ratio data.

#### Parameters

| Name | Kind | Required | Default | Annotation | Example | Observed example | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `year` | `str` |  | `year` | `year`, `quarter`, `year` | Loại kỳ báo cáo ('year' hoặc 'quarter'). Mặc định 'year'. |
| `display_mode` | `POSITIONAL_OR_KEYWORD` | `False` | `<FieldDisplayMode.STD: 'std'>` | `Union[str, app.lib.vnstock_alt.explorer.kbs.financial.FieldDisplayMode, NoneType]` | `FieldDisplayMode.STD` | `omitted; default "<FieldDisplayMode.STD: 'std'>"` | `item`, `item_id`, `vi`, `en` | Chế độ hiển thị trường dữ liệu. Mặc định FieldDisplayMode.STD. - FieldDisplayMode.STD: Chỉ giữ cột 'item' và 'item_id' (đã chuẩn hóa) - FieldDisplayMode.ALL: Giữ tất cả cột item (item, item_en, item_id) - 'vi': Chỉ giữ tên tiếng Việt (tương thích ngược) - 'en': Chỉ giữ tên tiếng Anh (tương thích ngược) - None: Giữ tất cả cột (tương thích ngược) |
| `show_log` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `Optional[bool]` |  | `False` |  | Hiển thị log debug. |

#### Source details

##### Source `kbs`

###### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_alt.explorer.kbs.financial.Finance`
- Provider method: `ratio`

```text
item, item_en, item_id, unit, levels, row_number
```
- Note: Derived from static analysis of provider DataFrame shaping logic.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:16:42.271104+00:00`
- Success: `True`
- Row count: `32`

```text
item, item_id, 2025, 2024, 2023, 2022
```
- Dtypes: `{'item': 'str', 'item_id': 'str', '2025': 'float64', '2024': 'float64', '2023': 'float64', '2022': 'float64'}`

```json
[
  {
    "item": "Thu nhập trên mỗi cổ phần của 4 quý gần nhất (EPS)",
    "item_id": "trailing_eps",
    "2025": 4542.29,
    "2024": 6053.11,
    "2023": 6507.05,
    "2022": 6334.29
  },
  {
    "item": "Giá trị sổ sách của cổ phiếu (BVPS)",
    "item_id": "book_value_per_share_bvps",
    "2025": 27231.3,
    "2024": 35105.74,
    "2023": 29524.06,
    "2022": 28662.57
  },
  {
    "item": "Chỉ số giá thị trường trên thu nhập (P/E)",
    "item_id": "p_e",
    "2025": 12.66,
    "2024": 15.07,
    "2023": 12.34,
    "2022": 12.63
  }
]
```

##### Source `vci`

###### Raw output contract

- Coverage: `partially-derived`
- Provider: `app.lib.vnstock_alt.explorer.vci.financial.Finance`
- Provider method: `ratio`

_No raw columns derived for this source._
- Note: No explicit column constant or recoverable DataFrame-shaping pattern found in provider method.

###### Normalized output schema

_No normalized schema declared for this API surface._

###### Live-observed sample

- Captured at: `2026-03-16T11:16:43.014913+00:00`
- Success: `True`
- Row count: `13`

```text
['Meta', 'ticker'], ['Meta', 'yearReport'], ['Meta', 'lengthReport'], ['Chỉ tiêu cơ cấu nguồn vốn', 'Fixed Asset-To-Equity'], ['Chỉ tiêu cơ cấu nguồn vốn', "Owners' Equity/Charter Capital"], ['Chỉ tiêu khả năng sinh lợi', 'Net Profit Margin (%)'], ['Chỉ tiêu khả năng sinh lợi', 'ROE (%)'], ['Chỉ tiêu khả năng sinh lợi', 'ROA (%)'], ['Chỉ tiêu khả năng sinh lợi', 'Dividend yield (%)'], ['Chỉ tiêu thanh khoản', 'Financial Leverage'], ['Chỉ tiêu định giá', 'Market Capital (Bn. VND)'], ['Chỉ tiêu định giá', 'Outstanding Share (Mil. Shares)'], ['Chỉ tiêu định giá', 'P/E'], ['Chỉ tiêu định giá', 'P/B'], ['Chỉ tiêu định giá', 'P/S'], ['Chỉ tiêu định giá', 'P/Cash Flow'], ['Chỉ tiêu định giá', 'EPS (VND)'], ['Chỉ tiêu định giá', 'BVPS (VND)']
```
- Dtypes: `{"('Meta', 'ticker')": 'str', "('Meta', 'yearReport')": 'int64', "('Meta', 'lengthReport')": 'int64', "('Chỉ tiêu cơ cấu nguồn vốn', 'Fixed Asset-To-Equity')": 'float64', '(\'Chỉ tiêu cơ cấu nguồn vốn\', "Owners\' Equity/Charter Capital")': 'float64', "('Chỉ tiêu khả năng sinh lợi', 'Net Profit Margin (%)')": 'float64', "('Chỉ tiêu khả năng sinh lợi', 'ROE (%)')": 'float64', "('Chỉ tiêu khả năng sinh lợi', 'ROA (%)')": 'float64', "('Chỉ tiêu khả năng sinh lợi', 'Dividend yield (%)')": 'float64', "('Chỉ tiêu thanh khoản', 'Financial Leverage')": 'float64', "('Chỉ tiêu định giá', 'Market Capital (Bn. VND)')": 'int64', "('Chỉ tiêu định giá', 'Outstanding Share (Mil. Shares)')": 'int64', "('Chỉ tiêu định giá', 'P/E')": 'float64', "('Chỉ tiêu định giá', 'P/B')": 'float64', "('Chỉ tiêu định giá', 'P/S')": 'float64', "('Chỉ tiêu định giá', 'P/Cash Flow')": 'float64', "('Chỉ tiêu định giá', 'EPS (VND)')": 'float64', "('Chỉ tiêu định giá', 'BVPS (VND)')": 'float64'}`

```json
[
  {
    "('Meta', 'ticker')": "VCB",
    "('Meta', 'yearReport')": 2025,
    "('Meta', 'lengthReport')": 5,
    "('Chỉ tiêu cơ cấu nguồn vốn', 'Fixed Asset-To-Equity')": 0.03618288308961001,
    "('Chỉ tiêu cơ cấu nguồn vốn', \"Owners' Equity/Charter Capital\")": 2.723129770368738,
    "('Chỉ tiêu khả năng sinh lợi', 'Net Profit Margin (%)')": 0.5995488660935886,
    "('Chỉ tiêu khả năng sinh lợi', 'ROE (%)')": 0.1661302403,
    "('Chỉ tiêu khả năng sinh lợi', 'ROA (%)')": 0.0155476893,
    "('Chỉ tiêu khả năng sinh lợi', 'Dividend yield (%)')": 0.0,
    "('Chỉ tiêu thanh khoản', 'Financial Leverage')": 10.732061193725775,
    "('Chỉ tiêu định giá', 'Market Capital (Bn. VND)')": 492984830900000,
    "('Chỉ tiêu định giá', 'Outstanding Share (Mil. Shares)')": 8355675100,
    "('Chỉ tiêu định giá', 'P/E')": 14.0139478861,
    "('Chỉ tiêu định giá', 'P/B')": 2.167305866,
    "('Chỉ tiêu định giá', 'P/S')": 6.8131741004,
    "('Chỉ tiêu định giá', 'P/Cash Flow')": 4.2741427889,
    "('Chỉ tiêu định giá', 'EPS (VND)')": 4210.091294717766,
    "('Chỉ tiêu định giá', 'BVPS (VND)')": 27222.7381124477
  },
  {
    "('Meta', 'ticker')": "VCB",
    "('Meta', 'yearReport')": 2024,
    "('Meta', 'lengthReport')": 5,
    "('Chỉ tiêu cơ cấu nguồn vốn', 'Fixed Asset-To-Equity')": 0.041246171534655304,
    "('Chỉ tiêu cơ cấu nguồn vốn', \"Owners' Equity/Charter Capital\")": 2.3482144266343346,
    "('Chỉ tiêu khả năng sinh lợi', 'Net Profit Margin (%)')": 0.6106116271176621,
    "('Chỉ tiêu khả năng sinh lợi', 'ROE (%)')": 0.1874015281,
    "('Chỉ tiêu khả năng sinh lợi', 'ROA (%)')": 0.0172473678,
    "('Chỉ tiêu khả năng sinh lợi', 'Dividend yield (%)')": 0.007826087,
    "('Chỉ tiêu thanh khoản', 'Financial Leverage')": 10.630866759498211,
    "('Chỉ tiêu định giá', 'Market Capital (Bn. VND)')": 321372749750000,
    "('Chỉ tiêu định giá', 'Outstanding Share (Mil. Shares)')": 5589091300,
    "('Chỉ tiêu định giá', 'P/E')": 10.3303411404,
    "('Chỉ tiêu định giá', 'P/B')": 1.6387128959,
    "('Chỉ tiêu định giá', 'P/S')": 4.6862029425,
    "('Chỉ tiêu định giá', 'P/Cash Flow')": 5.435679319,
    "('Chỉ tiêu định giá', 'EPS (VND)')": 6053.110279304258,
    "('Chỉ tiêu định giá', 'BVPS (VND)')": 35088.5137625145
  },
  {
    "('Meta', 'ticker')": "VCB",
    "('Meta', 'yearReport')": 2023,
    "('Meta', 'lengthReport')": 5,
    "('Chỉ tiêu cơ cấu nguồn vốn', 'Fixed Asset-To-Equity')": 0.04728045088792746,
    "('Chỉ tiêu cơ cấu nguồn vốn', \"Owners' Equity/Charter Capital\")": 1.9756690888871462,
    "('Chỉ tiêu khả năng sinh lợi', 'Net Profit Margin (%)')": 0.6173902507703569,
    "('Chỉ tiêu khả năng sinh lợi', 'ROE (%)')": 0.2198801657,
    "('Chỉ tiêu khả năng sinh lợi', 'ROA (%)')": 0.01809503,
    "('Chỉ tiêu khả năng sinh lợi', 'Dividend yield (%)')": 0.0,
    "('Chỉ tiêu thanh khoản', 'Financial Leverage')": 11.144409372664208,
    "('Chỉ tiêu định giá', 'Market Capital (Bn. VND)')": 509725126560000,
    "('Chỉ tiêu định giá', 'Outstanding Share (Mil. Shares)')": 5589091300,
    "('Chỉ tiêu định giá', 'P/E')": 16.7480098322,
    "('Chỉ tiêu định giá', 'P/B')": 3.0907665458,
    "('Chỉ tiêu định giá', 'P/S')": 7.5265662963,
    "('Chỉ tiêu định giá', 'P/Cash Flow')": -13.2253474774,
    "('Chỉ tiêu định giá', 'EPS (VND)')": 5922.434475171303,
    "('Chỉ tiêu định giá', 'BVPS (VND)')": 29507.2431541779
  }
]
```

#### Notes / caveats

Retrieve financial ratio data.
