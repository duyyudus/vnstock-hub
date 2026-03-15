'\nEquity Reference Domain.\n'
import pandas as pd
from app.lib.vnstock_data_alt.ui._base import BaseDomain
from app.lib.vnstock_data_alt.ui._registry import REFERENCE_SOURCES
class EquityReference(BaseDomain):
	'\n    Equity Reference Data (Layer 1).\n    '
	def __init__(A):super().__init__(domain_name='equity',layer_sources=REFERENCE_SOURCES)
	def list(A):'List all equity symbols.';return A._dispatch('list')
	def list_by_group(A,group):'\n        List equities by group (e.g., VN30, HOSE).\n        \n        Args:\n            group (str): Group code (VN30, HOSE, HNX, etc.)\n        '
	def list_by_industry(C,icb_code=None,lang='vi'):
		"\n        List equities by industry (ICB classification).\n        \n        Args:\n            icb_code (str, optional): The ICB code to filter by (e.g., '8773'). If None, returns all.\n            lang (str): Language ('vi' or 'en'). Default 'vi'.\n            \n        Returns:\n            DataFrame: List of symbols related to the industry with ICB info.\n        ";B=icb_code;A=C._dispatch('by_industry',lang=lang)
		if A is not None and not A.empty and B is not None:A=A[A['icb_code']==B].reset_index(drop=True)
		return A
	def by_group(A,group):'[DEPRECATED] Use list_by_group() instead.';return A.list_by_group(group)
	def list_by_exchange(A):'List all equities organized by exchange.';return A._dispatch('by_exchange')
	def by_exchange(A):'[DEPRECATED] Use list_by_exchange() instead.';return A.list_by_exchange()