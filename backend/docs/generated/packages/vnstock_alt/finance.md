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

- Captured at: `2026-03-17T05:27:39.230658+00:00`
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

- Captured at: `2026-03-17T05:26:39.782902+00:00`
- Success: `True`
- Row count: `52`

```text
ticker, yearReport, lengthReport, TOTAL ASSETS (Bn. VND), Cash and cash equivalents (Bn. VND), Balances with the SBV, Placements with and loans to other credit institutions, Trading Securities, net, Trading Securities, Provision for diminution in value of Trading Securities, Derivatives and other financial liabilities, Loans and advances to customers, net, Loans and advances to customers, Less: Provision for losses on loans and advances to customers, Investment Securities, Available-for Sales Securities, Held-to-Maturity Securities, Less: Provision for diminution in value of investment securities, Long-term investments (Bn. VND), Investment in joint ventures, Investments in associate companies, Other long-term assets (Bn. VND), Less: Provision for diminuation in value of long term investments, Fixed assets (Bn. VND), Tangible fixed assets, Intagible fixed assets, Other Assets, TOTAL RESOURCES (Bn. VND), LIABILITIES (Bn. VND), Due to Gov and borrowings from SBV, Deposits and borrowings from other credit institutions, Deposits from customers, _Derivatives and other financial liabilities, Funds received from Gov, international and other institutions, Convertible bonds/CDs and other valuable papers issued, Other liabilities, OWNER'S EQUITY(Bn.VND), Capital, Reserves, Foreign Currency Difference reserve, Difference upon Assets Revaluation, Undistributed earnings (Bn. VND), Minority Interest, Paid-in capital (Bn. VND), Other Reserves, MINORITY INTERESTS
```
- Dtypes: `{'ticker': 'str', 'yearReport': 'int64', 'lengthReport': 'int64', 'TOTAL ASSETS (Bn. VND)': 'int64', 'Cash and cash equivalents (Bn. VND)': 'int64', 'Balances with the SBV': 'int64', 'Placements with and loans to other credit institutions': 'int64', 'Trading Securities, net': 'int64', 'Trading Securities': 'int64', 'Provision for diminution in value of Trading Securities': 'int64', 'Derivatives and other financial liabilities': 'int64', 'Loans and advances to customers, net': 'int64', 'Loans and advances to customers': 'int64', 'Less: Provision for losses on loans and advances to customers': 'int64', 'Investment Securities': 'int64', 'Available-for Sales Securities': 'int64', 'Held-to-Maturity Securities': 'int64', 'Less: Provision for diminution in value of investment securities': 'int64', 'Long-term investments (Bn. VND)': 'int64', 'Investment in joint ventures': 'int64', 'Investments in associate companies': 'int64', 'Other long-term assets (Bn. VND)': 'int64', 'Less: Provision for diminuation in value of long term investments': 'int64', 'Fixed assets (Bn. VND)': 'int64', 'Tangible fixed assets': 'int64', 'Intagible fixed assets': 'int64', 'Other Assets': 'int64', 'TOTAL RESOURCES (Bn. VND)': 'int64', 'LIABILITIES (Bn. VND)': 'int64', 'Due to Gov and borrowings from SBV': 'int64', 'Deposits and borrowings from other credit institutions': 'int64', 'Deposits from customers': 'int64', '_Derivatives and other financial liabilities': 'int64', 'Funds received from Gov, international and other institutions': 'int64', 'Convertible bonds/CDs and other valuable papers issued': 'int64', 'Other liabilities': 'int64', "OWNER'S EQUITY(Bn.VND)": 'int64', 'Capital': 'int64', 'Reserves': 'int64', 'Foreign Currency Difference reserve': 'int64', 'Difference upon Assets Revaluation': 'int64', 'Undistributed earnings (Bn. VND)': 'int64', 'Minority Interest': 'int64', 'Paid-in capital (Bn. VND)': 'int64', 'Other Reserves': 'int64', 'MINORITY INTERESTS': 'int64'}`

```json
[
  {
    "ticker": "VCB",
    "yearReport": 2025,
    "lengthReport": 4,
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
    "Investment in joint ventures": 0,
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
    "Minority Interest": 0,
    "Paid-in capital (Bn. VND)": 83556751000000,
    "Other Reserves": 809837000000,
    "MINORITY INTERESTS": 71521000000
  },
  {
    "ticker": "VCB",
    "yearReport": 2025,
    "lengthReport": 3,
    "TOTAL ASSETS (Bn. VND)": 2378185628000000,
    "Cash and cash equivalents (Bn. VND)": 12909550000000,
    "Balances with the SBV": 56577155000000,
    "Placements with and loans to other credit institutions": 497858712000000,
    "Trading Securities, net": 5251130000000,
    "Trading Securities": 5307342000000,
    "Provision for diminution in value of Trading Securities": -56212000000,
    "Derivatives and other financial liabilities": 0,
    "Loans and advances to customers, net": 1595933982000000,
    "Loans and advances to customers": 1629942508000000,
    "Less: Provision for losses on loans and advances to customers": -34008526000000,
    "Investment Securities": 165634799000000,
    "Available-for Sales Securities": 135556078000000,
    "Held-to-Maturity Securities": 30320838000000,
    "Less: Provision for diminution in value of investment securities": -242117000000,
    "Long-term investments (Bn. VND)": 2418879000000,
    "Investment in joint ventures": 0,
    "Investments in associate companies": 904790000000,
    "Other long-term assets (Bn. VND)": 1589089000000,
    "Less: Provision for diminuation in value of long term investments": -75000000000,
    "Fixed assets (Bn. VND)": 8141017000000,
    "Tangible fixed assets": 5671432000000,
    "Intagible fixed assets": 2469585000000,
    "Other Assets": 33460404000000,
    "TOTAL RESOURCES (Bn. VND)": 2378185628000000,
    "LIABILITIES (Bn. VND)": 2155394083000000,
    "Due to Gov and borrowings from SBV": 162585347000000,
    "Deposits and borrowings from other credit institutions": 317172542000000,
    "Deposits from customers": 1611966698000000,
    "_Derivatives and other financial liabilities": 28651000000,
    "Funds received from Gov, international and other institutions": 2000000,
    "Convertible bonds/CDs and other valuable papers issued": 23085140000000,
    "Other liabilities": 40555703000000,
    "OWNER'S EQUITY(Bn.VND)": 222791545000000,
    "Capital": 89361977000000,
    "Reserves": 37060181000000,
    "Foreign Currency Difference reserve": -912953000000,
    "Difference upon Assets Revaluation": 0,
    "Undistributed earnings (Bn. VND)": 97169628000000,
    "Minority Interest": 0,
    "Paid-in capital (Bn. VND)": 83556751000000,
    "Other Reserves": 809837000000,
    "MINORITY INTERESTS": 112712000000
  },
  {
    "ticker": "VCB",
    "yearReport": 2025,
    "lengthReport": 2,
    "TOTAL ASSETS (Bn. VND)": 2217636600000000,
    "Cash and cash equivalents (Bn. VND)": 14790170000000,
    "Balances with the SBV": 37686004000000,
    "Placements with and loans to other credit institutions": 432541459000000,
    "Trading Securities, net": 7992828000000,
    "Trading Securities": 8031786000000,
    "Provision for diminution in value of Trading Securities": -38958000000,
    "Derivatives and other financial liabilities": 0,
    "Loans and advances to customers, net": 1522462472000000,
    "Loans and advances to customers": 1555769476000000,
    "Less: Provision for losses on loans and advances to customers": -33307004000000,
    "Investment Securities": 162434366000000,
    "Available-for Sales Securities": 128742046000000,
    "Held-to-Maturity Securities": 33936124000000,
    "Less: Provision for diminution in value of investment securities": -243804000000,
    "Long-term investments (Bn. VND)": 2314396000000,
    "Investment in joint ventures": 0,
    "Investments in associate companies": 860474000000,
    "Other long-term assets (Bn. VND)": 1528922000000,
    "Less: Provision for diminuation in value of long term investments": -75000000000,
    "Fixed assets (Bn. VND)": 7967792000000,
    "Tangible fixed assets": 5487426000000,
    "Intagible fixed assets": 2480366000000,
    "Other Assets": 29447113000000,
    "TOTAL RESOURCES (Bn. VND)": 2217636600000000,
    "LIABILITIES (Bn. VND)": 2003887178000000,
    "Due to Gov and borrowings from SBV": 99737152000000,
    "Deposits and borrowings from other credit institutions": 258784288000000,
    "Deposits from customers": 1586682675000000,
    "_Derivatives and other financial liabilities": 47261000000,
    "Funds received from Gov, international and other institutions": 2000000,
    "Convertible bonds/CDs and other valuable papers issued": 24165140000000,
    "Other liabilities": 34470660000000,
    "OWNER'S EQUITY(Bn.VND)": 213749422000000,
    "Capital": 89361977000000,
    "Reserves": 37060591000000,
    "Foreign Currency Difference reserve": -926111000000,
    "Difference upon Assets Revaluation": 0,
    "Undistributed earnings (Bn. VND)": 88145614000000,
    "Minority Interest": 0,
    "Paid-in capital (Bn. VND)": 83556751000000,
    "Other Reserves": 809837000000,
    "MINORITY INTERESTS": 107351000000
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

- Captured at: `2026-03-17T05:27:39.636649+00:00`
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

- Captured at: `2026-03-17T05:26:40.296552+00:00`
- Success: `True`
- Row count: `52`

```text
ticker, yearReport, lengthReport, Profits from other activities, Operating profit before changes in working capital, Net Cash Flows from Operating Activities before BIT, Payment from reserves, Purchase of fixed assets, Gain on Dividend, Net Cash Flows from Investing Activities, Increase in charter captial, Cash flows from financial activities, Net increase/decrease in cash and cash equivalents, Cash and cash equivalents, Foreign exchange differences Adjustment, Cash and Cash Equivalents at the end of period, Net cash inflows/outflows from operating activities, Proceeds from disposal of fixed assets, Investment in other entities, Proceeds from divestment in other entities, Dividends paid
```
- Dtypes: `{'ticker': 'str', 'yearReport': 'int64', 'lengthReport': 'int64', 'Profits from other activities': 'int64', 'Operating profit before changes in working capital': 'int64', 'Net Cash Flows from Operating Activities before BIT': 'int64', 'Payment from reserves': 'int64', 'Purchase of fixed assets': 'int64', 'Gain on Dividend': 'int64', 'Net Cash Flows from Investing Activities': 'int64', 'Increase in charter captial': 'int64', 'Cash flows from financial activities': 'int64', 'Net increase/decrease in cash and cash equivalents': 'int64', 'Cash and cash equivalents': 'int64', 'Foreign exchange differences Adjustment': 'int64', 'Cash and Cash Equivalents at the end of period': 'int64', 'Net cash inflows/outflows from operating activities': 'int64', 'Proceeds from disposal of fixed assets': 'int64', 'Investment in other entities': 'int64', 'Proceeds from divestment in other entities': 'int64', 'Dividends paid': 'int64'}`

```json
[
  {
    "ticker": "VCB",
    "yearReport": 2025,
    "lengthReport": 4,
    "Profits from other activities": -222398000000,
    "Operating profit before changes in working capital": 12013519000000,
    "Net Cash Flows from Operating Activities before BIT": 9537064000000,
    "Payment from reserves": -479136000000,
    "Purchase of fixed assets": -607910000000,
    "Gain on Dividend": 2249000000,
    "Net Cash Flows from Investing Activities": -660767000000,
    "Increase in charter captial": 0,
    "Cash flows from financial activities": -3776798000000,
    "Net increase/decrease in cash and cash equivalents": 4620363000000,
    "Cash and cash equivalents": 536179105000000,
    "Foreign exchange differences Adjustment": 0,
    "Cash and Cash Equivalents at the end of period": 540799468000000,
    "Net cash inflows/outflows from operating activities": 9057928000000,
    "Proceeds from disposal of fixed assets": 5692000000,
    "Investment in other entities": -60167000000,
    "Proceeds from divestment in other entities": 0,
    "Dividends paid": -3776798000000
  },
  {
    "ticker": "VCB",
    "yearReport": 2025,
    "lengthReport": 3,
    "Profits from other activities": 37554000000,
    "Operating profit before changes in working capital": 13260522000000,
    "Net Cash Flows from Operating Activities before BIT": 81932953000000,
    "Payment from reserves": -386371000000,
    "Purchase of fixed assets": -464524000000,
    "Gain on Dividend": 94521000000,
    "Net Cash Flows from Investing Activities": -364267000000,
    "Increase in charter captial": 0,
    "Cash flows from financial activities": 0,
    "Net increase/decrease in cash and cash equivalents": 81182315000000,
    "Cash and cash equivalents": 454996790000000,
    "Foreign exchange differences Adjustment": 0,
    "Cash and Cash Equivalents at the end of period": 536179105000000,
    "Net cash inflows/outflows from operating activities": 81546582000000,
    "Proceeds from disposal of fixed assets": 6107000000,
    "Investment in other entities": 0,
    "Proceeds from divestment in other entities": 0,
    "Dividends paid": 0
  },
  {
    "ticker": "VCB",
    "yearReport": 2025,
    "lengthReport": 2,
    "Profits from other activities": -57261000000,
    "Operating profit before changes in working capital": 9933187000000,
    "Net Cash Flows from Operating Activities before BIT": 38500239000000,
    "Payment from reserves": -582937000000,
    "Purchase of fixed assets": -194907000000,
    "Gain on Dividend": 12103000000,
    "Net Cash Flows from Investing Activities": -179204000000,
    "Increase in charter captial": 0,
    "Cash flows from financial activities": 0,
    "Net increase/decrease in cash and cash equivalents": 37738098000000,
    "Cash and cash equivalents": 417258692000000,
    "Foreign exchange differences Adjustment": 0,
    "Cash and Cash Equivalents at the end of period": 454996790000000,
    "Net cash inflows/outflows from operating activities": 37917302000000,
    "Proceeds from disposal of fixed assets": 3754000000,
    "Investment in other entities": 0,
    "Proceeds from divestment in other entities": 0,
    "Dividends paid": 0
  }
]
```

#### Notes / caveats

Retrieve cash flow data.

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

- Captured at: `2026-03-17T05:27:40.114741+00:00`
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

- Captured at: `2026-03-17T05:26:39.259072+00:00`
- Success: `True`
- Row count: `52`

```text
ticker, yearReport, lengthReport, Revenue (Bn. VND), Revenue YoY (%), Attribute to parent company (Bn. VND), Attribute to parent company YoY (%), Interest and Similar Income, Interest and Similar Expenses, Net Interest Income, Fees and Comission Income, Fees and Comission Expenses, Net Fee and Commission Income, Net gain (loss) from foreign currency and gold dealings, Net gain (loss) from trading of trading securities, Net gain (loss) from disposal of investment securities, Net Other income/(expenses), Other expenses, Net Other income/expenses, Dividends received, Total operating revenue, General & Admin Expenses, Operating Profit before Provision, Provision for credit losses, Profit before tax, Tax For the Year, Business income tax - current, Business income tax - deferred, Minority Interest, Net Profit For the Year, Attributable to parent company, EPS_basis
```
- Dtypes: `{'ticker': 'str', 'yearReport': 'int64', 'lengthReport': 'int64', 'Revenue (Bn. VND)': 'int64', 'Revenue YoY (%)': 'float64', 'Attribute to parent company (Bn. VND)': 'int64', 'Attribute to parent company YoY (%)': 'float64', 'Interest and Similar Income': 'int64', 'Interest and Similar Expenses': 'int64', 'Net Interest Income': 'int64', 'Fees and Comission Income': 'int64', 'Fees and Comission Expenses': 'int64', 'Net Fee and Commission Income': 'int64', 'Net gain (loss) from foreign currency and gold dealings': 'int64', 'Net gain (loss) from trading of trading securities': 'int64', 'Net gain (loss) from disposal of investment securities': 'int64', 'Net Other income/(expenses)': 'int64', 'Other expenses': 'int64', 'Net Other income/expenses': 'int64', 'Dividends received': 'int64', 'Total operating revenue': 'int64', 'General & Admin Expenses': 'int64', 'Operating Profit before Provision': 'int64', 'Provision for credit losses': 'int64', 'Profit before tax': 'int64', 'Tax For the Year': 'int64', 'Business income tax - current': 'int64', 'Business income tax - deferred': 'int64', 'Minority Interest': 'int64', 'Net Profit For the Year': 'int64', 'Attributable to parent company': 'int64', 'EPS_basis': 'int64'}`

```json
[
  {
    "ticker": "VCB",
    "yearReport": 2025,
    "lengthReport": 4,
    "Revenue (Bn. VND)": 28613751000000,
    "Revenue YoY (%)": 0.2134105036448007,
    "Attribute to parent company (Bn. VND)": 8629542000000,
    "Attribute to parent company YoY (%)": 0.007491087958990251,
    "Interest and Similar Income": 28613751000000,
    "Interest and Similar Expenses": -12443961000000,
    "Net Interest Income": 16169790000000,
    "Fees and Comission Income": 3288248000000,
    "Fees and Comission Expenses": -2423624000000,
    "Net Fee and Commission Income": 864624000000,
    "Net gain (loss) from foreign currency and gold dealings": 1226361000000,
    "Net gain (loss) from trading of trading securities": 30874000000,
    "Net gain (loss) from disposal of investment securities": 0,
    "Net Other income/(expenses)": 1677141000000,
    "Other expenses": -833124000000,
    "Net Other income/expenses": 844017000000,
    "Dividends received": 36829000000,
    "Total operating revenue": 19172495000000,
    "General & Admin Expenses": -7437685000000,
    "Operating Profit before Provision": 11734810000000,
    "Provision for credit losses": -847507000000,
    "Profit before tax": 10887303000000,
    "Tax For the Year": -2253520000000,
    "Business income tax - current": -1269144000000,
    "Business income tax - deferred": -984376000000,
    "Minority Interest": -4241000000,
    "Net Profit For the Year": 8633783000000,
    "Attributable to parent company": 8629542000000,
    "EPS_basis": 1033
  },
  {
    "ticker": "VCB",
    "yearReport": 2025,
    "lengthReport": 3,
    "Revenue (Bn. VND)": 26713295000000,
    "Revenue YoY (%)": 0.15187145027176674,
    "Attribute to parent company (Bn. VND)": 9020499000000,
    "Attribute to parent company YoY (%)": 0.052984977233669656,
    "Interest and Similar Income": 26713295000000,
    "Interest and Similar Expenses": -12056055000000,
    "Net Interest Income": 14657240000000,
    "Fees and Comission Income": 3028260000000,
    "Fees and Comission Expenses": -2089925000000,
    "Net Fee and Commission Income": 938335000000,
    "Net gain (loss) from foreign currency and gold dealings": 1279521000000,
    "Net gain (loss) from trading of trading securities": 106423000000,
    "Net gain (loss) from disposal of investment securities": 0,
    "Net Other income/(expenses)": 1159760000000,
    "Other expenses": -227990000000,
    "Net Other income/expenses": 931770000000,
    "Dividends received": 138839000000,
    "Total operating revenue": 18052128000000,
    "General & Admin Expenses": -6037436000000,
    "Operating Profit before Provision": 12014692000000,
    "Provision for credit losses": -775587000000,
    "Profit before tax": 11239105000000,
    "Tax For the Year": -2213552000000,
    "Business income tax - current": -2216968000000,
    "Business income tax - deferred": 3416000000,
    "Minority Interest": -5054000000,
    "Net Profit For the Year": 9025553000000,
    "Attributable to parent company": 9020499000000,
    "EPS_basis": 1080
  },
  {
    "ticker": "VCB",
    "yearReport": 2025,
    "lengthReport": 2,
    "Revenue (Bn. VND)": 25217499000000,
    "Revenue YoY (%)": 0.10299707729272828,
    "Attribute to parent company (Bn. VND)": 8831885000000,
    "Attribute to parent company YoY (%)": 0.08774906661983242,
    "Interest and Similar Income": 25217499000000,
    "Interest and Similar Expenses": -11057307000000,
    "Net Interest Income": 14160192000000,
    "Fees and Comission Income": 2810503000000,
    "Fees and Comission Expenses": -1949559000000,
    "Net Fee and Commission Income": 860944000000,
    "Net gain (loss) from foreign currency and gold dealings": 1635290000000,
    "Net gain (loss) from trading of trading securities": -2625000000,
    "Net gain (loss) from disposal of investment securities": 3616000000,
    "Net Other income/(expenses)": 1415156000000,
    "Other expenses": -262228000000,
    "Net Other income/expenses": 1152928000000,
    "Dividends received": 57892000000,
    "Total operating revenue": 17868237000000,
    "General & Admin Expenses": -6024732000000,
    "Operating Profit before Provision": 11843505000000,
    "Provision for credit losses": -809590000000,
    "Profit before tax": 11033915000000,
    "Tax For the Year": -2196544000000,
    "Business income tax - current": -2198412000000,
    "Business income tax - deferred": 1868000000,
    "Minority Interest": -5486000000,
    "Net Profit For the Year": 8837371000000,
    "Attributable to parent company": 8831885000000,
    "EPS_basis": 1057
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

- Captured at: `2026-03-17T05:27:40.567813+00:00`
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

- Captured at: `2026-03-17T05:26:43.609307+00:00`
- Success: `True`
- Row count: `52`

```text
['Meta', 'ticker'], ['Meta', 'yearReport'], ['Meta', 'lengthReport'], ['Chỉ tiêu cơ cấu nguồn vốn', 'Debt/Equity'], ['Chỉ tiêu cơ cấu nguồn vốn', 'Fixed Asset-To-Equity'], ['Chỉ tiêu cơ cấu nguồn vốn', "Owners' Equity/Charter Capital"], ['Chỉ tiêu khả năng sinh lợi', 'Net Profit Margin (%)'], ['Chỉ tiêu khả năng sinh lợi', 'ROE (%)'], ['Chỉ tiêu khả năng sinh lợi', 'ROIC (%)'], ['Chỉ tiêu khả năng sinh lợi', 'ROA (%)'], ['Chỉ tiêu thanh khoản', 'Financial Leverage'], ['Chỉ tiêu định giá', 'Market Capital (Bn. VND)'], ['Chỉ tiêu định giá', 'Outstanding Share (Mil. Shares)'], ['Chỉ tiêu định giá', 'P/E'], ['Chỉ tiêu định giá', 'P/B'], ['Chỉ tiêu định giá', 'P/S'], ['Chỉ tiêu định giá', 'P/Cash Flow'], ['Chỉ tiêu định giá', 'EPS (VND)'], ['Chỉ tiêu định giá', 'BVPS (VND)']
```
- Dtypes: `{"('Meta', 'ticker')": 'str', "('Meta', 'yearReport')": 'int64', "('Meta', 'lengthReport')": 'int64', "('Chỉ tiêu cơ cấu nguồn vốn', 'Debt/Equity')": 'float64', "('Chỉ tiêu cơ cấu nguồn vốn', 'Fixed Asset-To-Equity')": 'float64', '(\'Chỉ tiêu cơ cấu nguồn vốn\', "Owners\' Equity/Charter Capital")': 'float64', "('Chỉ tiêu khả năng sinh lợi', 'Net Profit Margin (%)')": 'float64', "('Chỉ tiêu khả năng sinh lợi', 'ROE (%)')": 'float64', "('Chỉ tiêu khả năng sinh lợi', 'ROIC (%)')": 'float64', "('Chỉ tiêu khả năng sinh lợi', 'ROA (%)')": 'float64', "('Chỉ tiêu thanh khoản', 'Financial Leverage')": 'float64', "('Chỉ tiêu định giá', 'Market Capital (Bn. VND)')": 'int64', "('Chỉ tiêu định giá', 'Outstanding Share (Mil. Shares)')": 'int64', "('Chỉ tiêu định giá', 'P/E')": 'float64', "('Chỉ tiêu định giá', 'P/B')": 'float64', "('Chỉ tiêu định giá', 'P/S')": 'float64', "('Chỉ tiêu định giá', 'P/Cash Flow')": 'float64', "('Chỉ tiêu định giá', 'EPS (VND)')": 'float64', "('Chỉ tiêu định giá', 'BVPS (VND)')": 'float64'}`

```json
[
  {
    "('Meta', 'ticker')": "VCB",
    "('Meta', 'yearReport')": 2025,
    "('Meta', 'lengthReport')": 4,
    "('Chỉ tiêu cơ cấu nguồn vốn', 'Debt/Equity')": 9.7320611937,
    "('Chỉ tiêu cơ cấu nguồn vốn', 'Fixed Asset-To-Equity')": 0.03618288308961001,
    "('Chỉ tiêu cơ cấu nguồn vốn', \"Owners' Equity/Charter Capital\")": 2.723129770368738,
    "('Chỉ tiêu khả năng sinh lợi', 'Net Profit Margin (%)')": 0.5336829977383751,
    "('Chỉ tiêu khả năng sinh lợi', 'ROE (%)')": 0.1660345318,
    "('Chỉ tiêu khả năng sinh lợi', 'ROIC (%)')": 0.0,
    "('Chỉ tiêu khả năng sinh lợi', 'ROA (%)')": 0.0155387322,
    "('Chỉ tiêu thanh khoản', 'Financial Leverage')": 10.732061193725775,
    "('Chỉ tiêu định giá', 'Market Capital (Bn. VND)')": 491313695527200,
    "('Chỉ tiêu định giá', 'Outstanding Share (Mil. Shares)')": 8355675094,
    "('Chỉ tiêu định giá', 'P/E')": 13.9664430315,
    "('Chỉ tiêu định giá', 'P/B')": 2.1599590649,
    "('Chỉ tiêu định giá', 'P/S')": 6.7900785901,
    "('Chỉ tiêu định giá', 'P/Cash Flow')": 4.2596541856,
    "('Chỉ tiêu định giá', 'EPS (VND)')": 1032.77615547745,
    "('Chỉ tiêu định giá', 'BVPS (VND)')": 27222.7381319956
  },
  {
    "('Meta', 'ticker')": "VCB",
    "('Meta', 'yearReport')": 2025,
    "('Meta', 'lengthReport')": 3,
    "('Chỉ tiêu cơ cấu nguồn vốn', 'Debt/Equity')": 9.6744877953,
    "('Chỉ tiêu cơ cấu nguồn vốn', 'Fixed Asset-To-Equity')": 0.03654096029541875,
    "('Chỉ tiêu cơ cấu nguồn vốn', \"Owners' Equity/Charter Capital\")": 2.666350025505192,
    "('Chỉ tiêu khả năng sinh lợi', 'Net Profit Margin (%)')": 0.6154295761002754,
    "('Chỉ tiêu khả năng sinh lợi', 'ROE (%)')": 0.1676701809,
    "('Chỉ tiêu khả năng sinh lợi', 'ROIC (%)')": 0.0,
    "('Chỉ tiêu khả năng sinh lợi', 'ROA (%)')": 0.0159773247,
    "('Chỉ tiêu thanh khoản', 'Financial Leverage')": 10.674487795306595,
    "('Chỉ tiêu định giá', 'Market Capital (Bn. VND)')": 635031307144000,
    "('Chỉ tiêu định giá', 'Outstanding Share (Mil. Shares)')": 8355675094,
    "('Chỉ tiêu định giá', 'P/E')": 18.084851421,
    "('Chỉ tiêu định giá', 'P/B')": 2.851781189,
    "('Chỉ tiêu định giá', 'P/S')": 8.9572704836,
    "('Chỉ tiêu định giá', 'P/Cash Flow')": 2.842624935,
    "('Chỉ tiêu định giá', 'EPS (VND)')": 1079.5655525760442,
    "('Chỉ tiêu định giá', 'BVPS (VND)')": 26650.0109799506
  },
  {
    "('Meta', 'ticker')": "VCB",
    "('Meta', 'yearReport')": 2025,
    "('Meta', 'lengthReport')": 2,
    "('Chỉ tiêu cơ cấu nguồn vốn', 'Debt/Equity')": 9.374936125,
    "('Chỉ tiêu cơ cấu nguồn vốn', 'Fixed Asset-To-Equity')": 0.0372763206816999,
    "('Chỉ tiêu cơ cấu nguồn vốn', \"Owners' Equity/Charter Capital\")": 2.5581346760776764,
    "('Chỉ tiêu khả năng sinh lợi', 'Net Profit Margin (%)')": 0.623712234975345,
    "('Chỉ tiêu khả năng sinh lợi', 'ROE (%)')": 0.1721818224,
    "('Chỉ tiêu khả năng sinh lợi', 'ROIC (%)')": 0.0,
    "('Chỉ tiêu khả năng sinh lợi', 'ROA (%)')": 0.0166133201,
    "('Chỉ tiêu thanh khoản', 'Financial Leverage')": 10.374936124973475,
    "('Chỉ tiêu định giá', 'Market Capital (Bn. VND)')": 527243098431400,
    "('Chỉ tiêu định giá', 'Outstanding Share (Mil. Shares)')": 8355675094,
    "('Chỉ tiêu định giá', 'P/E')": 15.2118214589,
    "('Chỉ tiêu định giá', 'P/B')": 2.467880488,
    "('Chỉ tiêu định giá', 'P/S')": 7.5666941013,
    "('Chỉ tiêu định giá', 'P/Cash Flow')": 4.1635556135,
    "('Chỉ tiêu định giá', 'EPS (VND)')": 1056.9923914755798,
    "('Chỉ tiêu định giá', 'BVPS (VND)')": 25568.499085539
  }
]
```

#### Notes / caveats

Retrieve financial ratio data.
