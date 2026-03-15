'\nFund Market Domain (Layer 2).\nWraps the `fmarket.fund` module for NAV history and portfolio holdings.\n'
_A='market.fund'
import pandas as pd
from app.lib.vnstock_data_alt.ui._base import BaseDetail
from app.lib.vnstock_data_alt.ui._registry import MARKET_SOURCES
from app.lib.vnstock_data_alt.ui.schemas.core import standardize_columns
class FundMarket(BaseDetail):
	"\n    Access point for fetching a specific fund's historical data and portfolio details.\n    "
	def __init__(A,symbol):super().__init__(symbol=symbol,domain_name=_A,layer_sources=MARKET_SOURCES)
	def _dispatch_and_format(A,method_name,**E):
		B=method_name;C=A._dispatch(B,**E)
		if C.empty:return C
		from app.lib.vnstock_data_alt.ui.config import get_route as F;G,D,D,D=F(A._domain_name,B,A._sources_config);H=_A;return standardize_columns(C,f"{H}.{B}",G,strict=False)
	def history(A,**B):'\n        Extracts the historical Net Asset Value (NAV) of the fund over time.\n        ';return A._dispatch_and_format('history',**B)
	def top_holding(A,**B):'\n        Extracts the top equity/bond holdings of the fund.\n        ';return A._dispatch_and_format('top_holding',**B)
	def industry_holding(A,**B):"\n        Extracts the industry weighting inside the fund's portfolio.\n        ";return A._dispatch_and_format('industry_holding',**B)
	def asset_holding(A,**B):'\n        Extracts the asset allocation (Equities vs Cash/Bonds) of the fund.\n        ';return A._dispatch_and_format('asset_holding',**B)