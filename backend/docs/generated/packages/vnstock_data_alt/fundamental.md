# Fundamental

- Qualified name: `app.lib.vnstock_data_alt.ui.fundamental.Fundamental`
- Signature: `()`

Central API Gateway for Layer 3 - Fundamental Data (Unified UI).

## Purpose

Central API Gateway for Layer 3 - Fundamental Data (Unified UI).
Provides financial statements, ratios, and diagnostic analysis for equities.

✅ METHODS AVAILABLE (5 total):

equity(symbol) → EquityFundamental object with 5 methods:
    - ratio()               → Financial ratios (P/E, ROE, Debt/Equity, etc.)
    - income_statement()    → Revenue, expenses, profit (quarterly/annual)
    - balance_sheet()       → Assets, liabilities, equity position
    - cash_flow()           → Operating, investing, financing cash flows
    - note()                → Footnotes and disclosures (Thuyết minh)

Example:
    fund = Fundamental()
    vic = fund.equity('VIC')
    
    ratios = vic.ratio()                    # 12 periods
    income = vic.income_statement()         # 12 periods
    balance = vic.balance_sheet()           # 12 periods
    cash = vic.cash_flow()                  # 12 periods
    notes = vic.note()                      # All footnotes

## Members

### equity

- Kind: `method`
- Signature: `(symbol)`
- Return type: `EquityFundamental`
- Purpose: Access financial data for a specific corporate equity (Fundamental Layer).

#### Parameters

| Name | Kind | Required | Default | Annotation | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `str` | `VIC`, `VNM`, `FPT` | The stock ticker symbol (e.g. 'VIC', 'VNM', 'FPT') |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access financial data for a specific corporate equity (Fundamental Layer).

    EquityFundamental: Object with 5 methods for financial analysis:
        - ratio()          - Key financial ratios
        - income_statement()- Income statement (12+ periods)
        - balance_sheet()  - Balance sheet (12+ periods)
        - cash_flow()      - Cash flow statement (12+ periods)
        - note()           - Financial disclosures/footnotes

Example:
    fund = Fundamental()
    vic_data = fund.equity('VIC')
    ratios = vic_data.ratio()
