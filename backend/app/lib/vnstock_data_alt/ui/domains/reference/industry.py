'\nIndustry Reference Domain.\n'
import pandas as pd
from app.lib.vnstock_data_alt.ui._base import BaseDomain
from app.lib.vnstock_data_alt.ui._registry import REFERENCE_SOURCES
class IndustryReference(BaseDomain):
	'\n    Industry Reference Data (Layer 1).\n    '
	def __init__(A):super().__init__(domain_name='industry',layer_sources=REFERENCE_SOURCES)
	def list(D,lang='vi'):
		"\n        List ICB industry classifications for all symbols in the market.\n        Uses VCI as the data source over KBS because VCI provides deeper ICB levels (up to 4 levels).\n        \n        Note: The underlying data source may not be available in all environments \n        (e.g., might be blocked on Google Colab).\n        \n        Args:\n            lang (str): Language code 'vi' or 'en'. Default 'vi'.\n        ";C='icb_name';B='en_icb_name';A=D._dispatch('list')
		if lang=='vi':
			if B in A.columns:A=A.drop(columns=[B])
		elif lang=='en':
			if C in A.columns:A=A.drop(columns=[C])
		return A
	def sectors(C,icb_code=None,lang='vi'):
		"\n        List all symbols by their industry sectors.\n        \n        Args:\n            icb_code (str, optional): Filter by a specific ICB code.\n            lang (str): Language code 'vi' or 'en'. Default 'vi'.\n        ";B=icb_code;A=C._dispatch('sectors',lang=lang)
		if A is not None and not A.empty and B is not None:A=A[A['icb_code']==B].reset_index(drop=True)
		return A