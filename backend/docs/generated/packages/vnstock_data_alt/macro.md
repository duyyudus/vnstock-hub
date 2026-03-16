# Macro

- Qualified name: `app.lib.vnstock_data_alt.ui.macro.Macro`
- Signature: `()`

Central API Gateway for Layer 6 - Macroeconomic Data (Unified UI).

## Purpose

Central API Gateway for Layer 6 - Macroeconomic Data (Unified UI).

✅ NEW STRUCTURE (Recommended):
    m = Macro()
    
    # 📊 Economy Indicators (8 methods)
    m.economy().gdp()
    m.economy().cpi()
    m.economy().industry_prod()
    m.economy().import_export()
    m.economy().retail()
    m.economy().fdi()
    m.economy().money_supply()
    m.economy().population_labor()
    
    # 💰 Commodity Prices (11 methods)
    m.commodity().gold(market='VN')        # Vietnam SJC gold or GLOBAL futures
    m.commodity().gas(market='VN')         # Vietnam RON/DO or GLOBAL natural gas
    m.commodity().oil_crude()              # Crude oil futures
    m.commodity().coke()                   # Coal/Coke prices
    m.commodity().steel(market='GLOBAL')   # HRC1! global or D10 Vietnam
    m.commodity().iron_ore()               # Iron ore prices
    m.commodity().fertilizer_ure()         # URE fertilizer
    m.commodity().soybean()                # Soybean prices
    m.commodity().corn()                   # Corn prices
    m.commodity().sugar()                  # Sugar prices
    m.commodity().pork(market='VN')        # Vietnam North Pig or China market
    
    # 💱 Currency & Interest Rates (2 methods)
    m.currency().exchange_rate()           # Foreign exchange rates (USD, JPY, EUR, etc.)
    m.currency().interest_rate()           # VND deposit/lending rates

❌ LEGACY STRUCTURE (Deprecated, will be removed 31/8/2026):
    m = Macro()
    m.gdp()  # Shows deprecation warning
    m.exchange_rate()  # Shows deprecation warning

## Members

### economy

- Kind: `method`
- Signature: `()`
- Return type: `EconomyReference`
- Purpose: Access standard macroeconomic indicators (Macro Layer - Economy Domain).

#### Parameters

_None._

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access standard macroeconomic indicators (Macro Layer - Economy Domain).

Methods available (8 total):
    - gdp()                  → GDP data (quarterly, annual)
    - cpi()                  → Consumer Price Index (monthly, annual)
    - industry_prod()        → Industrial Production Index
    - import_export()        → Import/Export statistics
    - retail()               → Retail sales volume
    - fdi()                  → Foreign Direct Investment flows
    - money_supply()         → Credit/Money supply (M0, M1, M2)
    - population_labor()     → Population and labor force statistics

Example:
    m = Macro()
    gdp_df = m.economy().gdp(length='5Y')  # Get 5 years of GDP data

### commodity

- Kind: `method`
- Signature: `()`
- Return type: `CommodityReference`
- Purpose: Access global and local commodity prices (Macro Layer - Commodity Domain).

#### Parameters

_None._

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access global and local commodity prices (Macro Layer - Commodity Domain).

Methods available (11 total):
    - gold(market='VN'|'GLOBAL')        → Gold prices (Vietnam SJC or international futures)
    - gas(market='VN'|'GLOBAL')         → Gas prices (Vietnam RON/DO or natural gas futures)
    - oil_crude()                       → Crude oil futures (WTI, Brent)
    - coke()                            → Coal/Coke prices
    - steel(market='GLOBAL'|'VN')       → Steel (HRC1! global or D10 Vietnam)
    - iron_ore()                        → Iron ore prices
    - fertilizer_ure()                  → Urea fertilizer prices
    - soybean()                         → Soybean futures
    - corn()                            → Corn futures
    - sugar()                           → Sugar futures
    - pork(market='VN'|'CHINA')         → Pork prices (Vietnam or China market)

Example:
    m = Macro()
    gold_vn = m.commodity().gold(market='VN')
    gold_global = m.commodity().gold(market='GLOBAL')
    steel = m.commodity().steel(market='GLOBAL')

### currency

- Kind: `method`
- Signature: `()`
- Return type: `CurrencyReference`
- Purpose: Access foreign exchange rates and interest rate data (Macro Layer - Currency Domain).

#### Parameters

_None._

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access foreign exchange rates and interest rate data (Macro Layer - Currency Domain).

Methods available (2 total):
    - exchange_rate()       → Foreign exchange rates (USD, JPY, EUR, GBP, etc.) vs VND
    - interest_rate()       → VND deposit/lending rates (overnight, 1M, 3M, 6M, 12M)

Example:
    m = Macro()
    fx_rates = m.currency().exchange_rate(length='1Y')
    int_rates = m.currency().interest_rate(length='1Y')

### gdp

- Kind: `method`
- Signature: `(start = None, end = None, period = 'quarter', keep_label = False, length = None)`
- Declared signature: `(start=None, end=None, period='quarter', keep_label=False, length=None, **B)`
- Purpose: [DEPRECATED] GDP data. | Dữ liệu GDP — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().economy().gdp() thay thế.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `quarter` | `` |
| `keep_label` | `POSITIONAL_OR_KEYWORD` | `False` | `False` | `` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

[DEPRECATED] GDP data. | Dữ liệu GDP — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().economy().gdp() thay thế.

### cpi

- Kind: `method`
- Signature: `(start = None, end = None, period = 'month', length = None)`
- Declared signature: `(start=None, end=None, period='month', length=None, **B)`
- Purpose: [DEPRECATED] Consumer Price Index. | Chỉ số giá tiêu dùng CPI — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().economy().cpi() thay thế.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `month` | `` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

[DEPRECATED] Consumer Price Index. | Chỉ số giá tiêu dùng CPI — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().economy().cpi() thay thế.

### industry_prod

- Kind: `method`
- Signature: `(start = None, end = None, period = 'month', length = None)`
- Declared signature: `(start=None, end=None, period='month', length=None, **B)`
- Purpose: [DEPRECATED] Industrial Production. | Sản xuất công nghiệp — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().economy().industry_prod() thay thế.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `month` | `` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

[DEPRECATED] Industrial Production. | Sản xuất công nghiệp — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().economy().industry_prod() thay thế.

### import_export

- Kind: `method`
- Signature: `(start = None, end = None, period = 'month', length = None)`
- Declared signature: `(start=None, end=None, period='month', length=None, **B)`
- Purpose: [DEPRECATED] Import-Export data. | Dữ liệu Xuất - Nhập khẩu — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().economy().import_export() thay thế.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `month` | `` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

[DEPRECATED] Import-Export data. | Dữ liệu Xuất - Nhập khẩu — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().economy().import_export() thay thế.

### retail

- Kind: `method`
- Signature: `(start = None, end = None, period = 'month', length = None)`
- Declared signature: `(start=None, end=None, period='month', length=None, **B)`
- Purpose: [DEPRECATED] Retail sales. | Bán lẻ — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().economy().retail() thay thế.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `month` | `` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

[DEPRECATED] Retail sales. | Bán lẻ — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().economy().retail() thay thế.

### fdi

- Kind: `method`
- Signature: `(start = None, end = None, period = 'month', length = None)`
- Declared signature: `(start=None, end=None, period='month', length=None, **B)`
- Purpose: [DEPRECATED] Foreign Direct Investment. | Đầu tư trực tiếp nước ngoài FDI — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().economy().fdi() thay thế.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `month` | `` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

[DEPRECATED] Foreign Direct Investment. | Đầu tư trực tiếp nước ngoài FDI — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().economy().fdi() thay thế.

### money_supply

- Kind: `method`
- Signature: `(start = None, end = None, period = 'month', length = None)`
- Declared signature: `(start=None, end=None, period='month', length=None, **B)`
- Purpose: [DEPRECATED] Money Supply. | Tín dụng — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().economy().money_supply() thay thế.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `month` | `` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

[DEPRECATED] Money Supply. | Tín dụng — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().economy().money_supply() thay thế.

### population_labor

- Kind: `method`
- Signature: `(start = None, end = None, period = 'year', length = None)`
- Declared signature: `(start=None, end=None, period='year', length=None, **B)`
- Purpose: [DEPRECATED] Population and Labor. | Dân số và lao động — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().economy().population_labor() thay thế.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `year` | `` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

[DEPRECATED] Population and Labor. | Dân số và lao động — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().economy().population_labor() thay thế.

### exchange_rate

- Kind: `method`
- Signature: `(start = None, end = None, period = 'day', length = None)`
- Declared signature: `(start=None, end=None, period='day', length=None, **B)`
- Purpose: [DEPRECATED] Exchange Rates. | Tỷ giá hối đoái — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().currency().exchange_rate() thay thế.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `day` | `` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

[DEPRECATED] Exchange Rates. | Tỷ giá hối đoái — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().currency().exchange_rate() thay thế.

### interest_rate

- Kind: `method`
- Signature: `(start = None, end = None, period = 'day', format = 'pivot', length = None)`
- Declared signature: `(start=None, end=None, period='day', format='pivot', length=None, **B)`
- Purpose: [DEPRECATED] Interest Rates. | Lãi suất — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().currency().interest_rate() thay thế.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `start` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `end` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |
| `period` | `POSITIONAL_OR_KEYWORD` | `False` | `day` | `` |
| `format` | `POSITIONAL_OR_KEYWORD` | `False` | `pivot` | `` |
| `length` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

[DEPRECATED] Interest Rates. | Lãi suất — cấu trúc method sẽ thay đổi sau 31/8/2026, dùng Macro().currency().interest_rate() thay thế.
