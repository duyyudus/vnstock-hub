'\nDerivatives Market Data Domain.\n'
from app.lib.vnstock_data_alt.ui.domains.market.base import BaseMarketData
from app.lib.vnstock_data_alt.ui._registry import MARKET_SOURCES
class FuturesMarket(BaseMarketData):
	'\n    Futures Market Data (Layer 2).\n    ';session_stats=None
	def __init__(A,symbol):super().__init__(symbol=symbol,domain_name='market.futures',layer_sources=MARKET_SOURCES)
class WarrantMarket(BaseMarketData):
	'\n    Warrant Market Data (Layer 2).\n    ';session_stats=None
	def __init__(A,symbol):super().__init__(symbol=symbol,domain_name='market.warrant',layer_sources=MARKET_SOURCES)
class DerivativesMarket:
	'\n    Derivatives market wrapper containing Futures and Warrants.\n    Provides nested access to derivatives market data.\n    '
	def futures(A,symbol):"\n        Access futures market data.\n        \n        Args:\n            symbol (str): Futures symbol (e.g., 'VN30F2503', 'VN100F2503').\n        ";return FuturesMarket(symbol)
	def warrant(A,symbol):"\n        Access warrant market data.\n        \n        Args:\n            symbol (str): Warrant symbol (e.g., 'CACB2511', 'VICW-VIC26A').\n        ";return WarrantMarket(symbol=symbol)