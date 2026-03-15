'\nMarket Data Layer Entry Point.\n'
from __future__ import annotations
_A=False
from typing import List,TYPE_CHECKING
import pandas as pd
if TYPE_CHECKING:from app.lib.vnstock_data_alt.ui.domains.market.equity import EquityMarket;from app.lib.vnstock_data_alt.ui.domains.market.derivatives import DerivativesMarket,FuturesMarket,WarrantMarket;from app.lib.vnstock_data_alt.ui.domains.market.fund import FundMarket;from app.lib.vnstock_data_alt.ui.domains.market.etf import ETFMarket;from app.lib.vnstock_data_alt.ui.domains.market.crypto import CryptoMarket;from app.lib.vnstock_data_alt.ui.domains.market.forex import ForexMarket;from app.lib.vnstock_data_alt.ui.domains.market.commodity import CommodityMarket
from app.lib.vnstock_data_alt.ui._base import BaseDomain
from app.lib.vnstock_data_alt.ui._registry import MARKET_SOURCES
class Market:
	'\n    Market Data Layer (Layer 2).\n    Provides access to real-time and historical pricing data across all asset classes.\n    '
	def __init__(A,index='VNINDEX',random_agent=_A,show_log=_A,**B):A._index=index;A._deprecated_warned=_A
	def _warn_deprecate(A):
		if not getattr(A,'_deprecated_warned',_A):import warnings as B;B.warn('Market methods pe(), pb(), evaluation() will be removed on 2026-08-31. Please use the new structure: `Analytics().valuation(index).pe()` instead.',PendingDeprecationWarning,stacklevel=3);A._deprecated_warned=True
	def pe(A,duration='5Y'):A._warn_deprecate();from app.lib.vnstock_data_alt.ui.analytics import Analytics as B;return B().valuation(index=A._index).pe(duration=duration)
	def pb(A,duration='5Y'):A._warn_deprecate();from app.lib.vnstock_data_alt.ui.analytics import Analytics as B;return B().valuation(index=A._index).pb(duration=duration)
	def evaluation(A,duration='5Y'):A._warn_deprecate();from app.lib.vnstock_data_alt.ui.analytics import Analytics as B;return B().valuation(index=A._index).evaluation(duration=duration)
	def equity(B,symbol):'Access equity market data.';from app.lib.vnstock_data_alt.ui.domains.market.equity import EquityMarket as A;return A(symbol)
	def index(B,symbol):'Access index market data.';from app.lib.vnstock_data_alt.ui.domains.market.index import IndexMarket as A;return A(symbol)
	def futures(B,symbol):'Access futures market data.';from app.lib.vnstock_data_alt.ui.domains.market.derivatives import FuturesMarket as A;return A(symbol)
	def warrant(B,symbol):'Access warrant market data.';from app.lib.vnstock_data_alt.ui.domains.market.derivatives import WarrantMarket as A;return A(symbol)
	def etf(B,symbol):'Access ETF market data.';from app.lib.vnstock_data_alt.ui.domains.market.etf import ETFMarket as A;return A(symbol)
	def fund(B,symbol=None):'\n        Access historical NAVs and portfolio compositions for a specific Mutual Fund.\n        ';from app.lib.vnstock_data_alt.ui.domains.market.fund import FundMarket as A;return A(symbol=symbol)
	def crypto(B,symbol):"Access crypto market data (e.g., 'BTC').";from app.lib.vnstock_data_alt.ui.domains.market.crypto import CryptoMarket as A;return A(symbol)
	def forex(B,symbol):"Access forex market data (e.g., 'USDVND').";from app.lib.vnstock_data_alt.ui.domains.market.forex import ForexMarket as A;return A(symbol)
	def commodity(B,symbol):"Access commodity market data (e.g., 'GC=F').";from app.lib.vnstock_data_alt.ui.domains.market.commodity import CommodityMarket as A;return A(symbol)
	def quote(C,symbols_list,**A):'\n        Real-time multi-symbol pricing snapshot.\n        ';B=BaseDomain('market.market_wide',MARKET_SOURCES);A['symbols_list']=symbols_list;A['get_all']=True;return B._dispatch('quote',**A)
	def price_board(A,symbols_list,**B):'\n        [Backward Compatible Alias] Relays to `.quote()`.\n        Provides real-time multi-symbol pricing snapshot.\n        ';return A.quote(symbols_list,**B)