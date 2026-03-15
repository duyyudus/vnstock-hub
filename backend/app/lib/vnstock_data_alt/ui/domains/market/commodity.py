'\nCommodity Market Data Domain.\n'
_A=None
from app.lib.vnstock_data_alt.ui.domains.market.base import BaseMarketData
from app.lib.vnstock_data_alt.ui._registry import MARKET_SOURCES
class CommodityMarket(BaseMarketData):
	'\n    Commodity Market Data (Layer 2).\n    Provides access to historical pricing data for commodities via MSN source.\n    ';trades=_A;intraday=_A;order_book=_A;price_depth=_A;session_stats=_A;trading_stats=_A
	def __init__(A,symbol):super().__init__(symbol=symbol,domain_name='market.commodity',layer_sources=MARKET_SOURCES)