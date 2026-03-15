'\nEquity Market Data Domain.\n'
import pandas as pd
from typing import Optional
from app.lib.vnstock_data_alt.ui.domains.market.base import BaseMarketData
from app.lib.vnstock_data_alt.ui._registry import MARKET_SOURCES
class EquityMarket(BaseMarketData):
	'\n    Equity Market Data (Layer 2).\n    Provides standard methods (history, intraday, price_depth, price_board)\n    plus specialized flow and statistics models for Equities.\n    '
	def __init__(A,symbol):super().__init__(symbol=symbol,domain_name='market.equity',layer_sources=MARKET_SOURCES)
	def foreign_flow(A,**B):'\n        Historical or daily foreign buy/sell volume and value.\n        ';return A._dispatch('foreign_flow',**B)
	def proprietary_flow(A,**B):'\n        Trade data for proprietary desks (Tự doanh).\n        ';return A._dispatch('proprietary_flow',**B)
	def box_trades(A):0
	def block_trades(C,limit=1000,**A):
		'\n        Real-time or historical data for negotiated/block trades (giao dịch thoả thuận).\n        \n        Args:\n            limit (int): Number of records to fetch (default: 1000).\n        ';B='page_size'
		if B not in A:A[B]=limit
		return C._dispatch('block_trades',**A)
	def put_through(A,**B):'Alias for block_trades (Negotiated Trades).';return A.block_trades(**B)
	def odd_lot(B,**A):
		'\n        Real-time pricing or trades for odd-lot execution (Lô lẻ).\n        ';C='symbols_list'
		if C not in A:A[C]=[B.symbol]
		return B._dispatch('odd_lot',**A)
	def volume_profile(A,**B):'\n        Aggregated volume distributed across executed price levels (Volume Profile).\n        ';return A._dispatch('volume_profile',**B)
	def matched_by_price(A,**B):'Alias for volume_profile.';return A.volume_profile(**B)