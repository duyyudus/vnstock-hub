from app.lib.vnstock_data_alt.ui.domains.market.base import BaseMarketData
from app.lib.vnstock_data_alt.ui._registry import MARKET_SOURCES
class ETFMarket(BaseMarketData):
	'\n    Electronic Traded Funds (ETF) market data domain.\n    Provides access to quotes, trades, and history for ETF symbols.\n    '
	def __init__(A,symbol):super().__init__(symbol,'market.etf',MARKET_SOURCES)