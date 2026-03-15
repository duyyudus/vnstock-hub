'\nETF Reference Domain (Layer 1).\n'
import pandas as pd
from app.lib.vnstock_data_alt.ui._base import BaseDomain
from app.lib.vnstock_data_alt.ui._registry import REFERENCE_SOURCES
class ETFReference(BaseDomain):
	'\n    ETF Reference Data (Layer 1).\n    '
	def __init__(A):super().__init__(domain_name='etf',layer_sources=REFERENCE_SOURCES)
	def list(B):
		'\n        List all Exchange-Traded Funds (ETFs) available in the market.\n        ';A=B._dispatch('list')
		if A is None or A.empty:return pd.DataFrame()
		return A