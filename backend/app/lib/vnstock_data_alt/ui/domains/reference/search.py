'\nSearch Reference Domain.\n'
from typing import Optional
import pandas as pd
from app.lib.vnstock_data_alt.ui._base import BaseDomain
from app.lib.vnstock_data_alt.ui._registry import REFERENCE_SOURCES
class SearchReference(BaseDomain):
	'\n    Search Reference Data.\n    Provides access to cross-domain search for symbols (stocks, crypto, forex, indices) globally.\n    '
	def __init__(A):super().__init__(domain_name='search',layer_sources=REFERENCE_SOURCES)
	def symbol(A,query,locale=None,limit=10):"\n        Retrieves a list of symbols from the market matching the query.\n        Backed by MSN Autosuggest to find global and local financial instruments.\n        \n        Args:\n            query (str): Search keyword to find the symbol.\n            locale (str, optional): Target language/locale to filter results (e.g., 'vi-vn', 'en-us').\n            limit (int, optional): Max number of results. Defaults to 10.\n            \n        Returns:\n            pd.DataFrame: List of matched symbols and their descriptions.\n        ";return A._dispatch('symbol',query=query,locale=locale,limit=limit)