'\nIndex Market Data Domain.\n'
_A=None
from app.lib.vnstock_data_alt.ui.domains.market.base import BaseMarketData
from app.lib.vnstock_data_alt.ui._registry import MARKET_SOURCES
class IndexMarket(BaseMarketData):
	'\n    Index Market Data (Layer 2).\n    ';trades=_A;intraday=_A;order_book=_A;price_depth=_A;session_stats=_A;trading_stats=_A
	def __init__(A,symbol):super().__init__(symbol=symbol,domain_name='market.index',layer_sources=MARKET_SOURCES)