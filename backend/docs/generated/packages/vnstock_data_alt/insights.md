# Insights

- Qualified name: `app.lib.vnstock_data_alt.ui.insights.Insights`
- Signature: `()`

Central API Gateway for Layer 4 - Insights & Analytics (Unified UI).

## Purpose

Central API Gateway for Layer 4 - Insights & Analytics (Unified UI).
Provides market ranking, valuation multiples, and analytical insights.

✅ METHODS AVAILABLE (7 total across 1 domain):

Ranking Domain (7 methods) - Market top movers:
    - gainer()        - Top stocks with highest price increase
    - loser()         - Top stocks with highest price decrease
    - value()         - Top stocks by trading value
    - volume()        - Top stocks by volume spikes
    - deal()          - Top stocks by put-through/deal volume
    - foreign_buy()   - Top stocks by foreign net buy value
    - foreign_sell()  - Top stocks by foreign net sell value

Example:
    ins = Insights()
    
    # Ranking methods
    gainers = ins.ranking().gainer(index='VNINDEX', limit=10)
    volume_movers = ins.ranking().volume(index='VNINDEX', limit=10)

## Members

### ranking

- Kind: `method`
- Signature: `()`
- Return type: `RankingReference`
- Purpose: Access market ranking metrics - Top movers by various criteria (Insights Layer).

#### Parameters

_None._

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access market ranking metrics - Top movers by various criteria (Insights Layer).

Methods available (7 total):
    - gainer(index, limit)      - Top gainers
    - loser(index, limit)       - Top losers
    - value(index, limit)       - Top by trading value
    - volume(index, limit)      - Top by volume spikes
    - deal(index, limit)        - Top by put-through volume
    - foreign_buy(date, limit)  - Top by foreign net buy
    - foreign_sell(date, limit) - Top by foreign net sell

### screener

- Kind: `method`
- Signature: `()`
- Return type: `ScreenerReference`
- Purpose: Access stock screener functionality (Insights Layer).

#### Parameters

_None._

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access stock screener functionality (Insights Layer).
