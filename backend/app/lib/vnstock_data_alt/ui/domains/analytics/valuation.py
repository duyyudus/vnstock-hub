'\nValuation Domain (Insights Layer).\n'
import pandas as pd
from app.lib.vnstock_data_alt.ui._base import BaseDomain
from app.lib.vnstock_data_alt.ui.schemas.core import standardize_columns
from app.lib.vnstock_data_alt.ui._registry import INSIGHTS_SOURCES
class MarketValuation(BaseDomain):
	'\n    Market Valuation Insights.\n    Wraps provider methods (like vnd.market.Market) to provide\n    historical P/E and P/B ratios for an index.\n    '
	def __init__(A,index='VNINDEX'):super().__init__(domain_name='insights.valuation',layer_sources=INSIGHTS_SOURCES);A.index=index
	def _dispatch_custom(B,method_name,**G):
		'\n        Custom dispatch to inject the index parameter during initialization\n        for providers that require it (like vnd.market.Market).\n        ';D=method_name;from app.lib.vnstock_data_alt.ui.config import get_route as H;from app.lib.vnstock_data_alt.core.registry import ProviderRegistry as E;import importlib as I;A,C,N,J=H(B._domain_name,D,B._sources_config)
		if not E.is_registered(C,A):
			try:I.import_module(f"vnstock_data.explorer.{A}.{C}")
			except ImportError:pass
		F=E.get(C,A)
		if not F:raise ValueError(f"Provider not found for {A}.{C}")
		K=F(index=B.index);L=getattr(K,J);M=L(**G);return standardize_columns(M,f"{B._domain_name}.{D}",A,strict=True)
	def pe(A,duration='5Y',**B):'Retrieves P/E (Price-to-Earnings) ratio data.';return A._dispatch_custom('pe',duration=duration,**B)
	def pb(A,duration='5Y',**B):'Retrieves P/B (Price-to-Book) ratio data.';return A._dispatch_custom('pb',duration=duration,**B)
	def evaluation(A,duration='5Y',**B):'Retrieves an overview of the market with both P/E and P/B ratios.';return A._dispatch_custom('evaluation',duration=duration,**B)