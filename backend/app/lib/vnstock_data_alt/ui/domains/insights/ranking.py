'\nRanking Domain (Insights Layer).\n'
_A='VNINDEX'
import pandas as pd
from app.lib.vnstock_data_alt.ui._base import BaseDomain
from app.lib.vnstock_data_alt.ui.schemas.core import standardize_columns
from app.lib.vnstock_data_alt.ui._registry import INSIGHTS_SOURCES
class RankingReference(BaseDomain):
	'\n    Market Ranking Insights.\n    Wraps provider methods (like vnd.insight.TopStock) to identify\n    market movers based on volume, value, foreign flow, etc.\n    '
	def __init__(A):super().__init__(domain_name='insights.ranking',layer_sources=INSIGHTS_SOURCES)
	def _dispatch_and_format(A,method_name,**D):B=method_name;E=A._dispatch(B,**D);from app.lib.vnstock_data_alt.ui.config import get_route as F;G,C,C,C=F(A._domain_name,B,A._sources_config);return standardize_columns(E,f"{A._domain_name}.{B}",G,strict=True)
	def gainer(A,index=_A,limit=10,**B):'Top 10 stocks with highest price increase.';return A._dispatch_and_format('gainer',index=index,limit=limit,**B)
	def loser(A,index=_A,limit=10,**B):'Top 10 stocks with highest price decrease.';return A._dispatch_and_format('loser',index=index,limit=limit,**B)
	def value(A,index=_A,limit=10,**B):'Top 10 stocks with highest trading value.';return A._dispatch_and_format('value',index=index,limit=limit,**B)
	def volume(A,index=_A,limit=10,**B):'Top 10 stocks with highest volume spikes.';return A._dispatch_and_format('volume',index=index,limit=limit,**B)
	def deal(A,index=_A,limit=10,**B):'Top 10 stocks with highest put-through/deal volume spikes.';return A._dispatch_and_format('deal',index=index,limit=limit,**B)
	def foreign_buy(A,date=None,limit=10,**B):'Top 10 stocks with highest foreign net buy value.';return A._dispatch_and_format('foreign_buy',date=date,limit=limit,**B)
	def foreign_sell(A,date=None,limit=10,**B):'Top 10 stocks with highest foreign net sell value.';return A._dispatch_and_format('foreign_sell',date=date,limit=limit,**B)