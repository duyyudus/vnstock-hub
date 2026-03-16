# Analytics

- Qualified name: `app.lib.vnstock_data_alt.ui.analytics.Analytics`
- Signature: `()`

Central API Gateway for Layer 4 - Analytics (Unified UI).

## Purpose

Central API Gateway for Layer 4 - Analytics (Unified UI).
Provides valuation models, risk metrics, and quality scores.

✅ METHODS AVAILABLE:

val = Analytics().valuation('VNINDEX')
pe_data = val.pe(duration='5Y')

## Members

### valuation

- Kind: `method`
- Signature: `(index='VNINDEX')`
- Return type: `MarketValuation`
- Purpose: Access historical valuation multiples for market indices (Analytics Layer).

#### Parameters

| Name | Kind | Required | Default | Annotation | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `index` | `POSITIONAL_OR_KEYWORD` | `False` | `VNINDEX` | `str` | `VNINDEX`, `HNX` | Market index code (e.g. 'VNINDEX', 'HNX') |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access historical valuation multiples for market indices (Analytics Layer).

Methods available (3 total):
    - pe(duration)         - P/E ratio (Price-to-Earnings) historical
    - pb(duration)         - P/B ratio (Price-to-Book) historical
    - evaluation(duration) - Market evaluation (P/E + P/B overview)
