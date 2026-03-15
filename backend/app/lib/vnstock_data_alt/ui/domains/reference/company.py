'\nCompany Reference Domain.\n'
from typing import Optional
import pandas as pd
from app.lib.vnstock_data_alt.ui._base import BaseDetail
from app.lib.vnstock_data_alt.ui._registry import REFERENCE_SOURCES
class CompanyReference(BaseDetail):
	'\n    Company Reference Data (Layer 1).\n    Wraps functionality for retrieving company-specific static data.\n    '
	def __init__(A,symbol):super().__init__(symbol=symbol,domain_name='company',layer_sources=REFERENCE_SOURCES)
	def info(A):'Get company info/overview.';return A._dispatch('info')
	def shareholders(A):'Get company shareholders.';return A._dispatch('shareholders')
	def officers(A,filter_by='working'):"\n        Get company officers.\n        \n        Args:\n            filter_by (str): 'working', 'resigned', or 'all'. Default 'working'.\n        ";return A._dispatch('officers',filter_by=filter_by)
	def subsidiaries(A,filter_by='all'):"\n        Get company subsidiaries.\n        \n        Args:\n             filter_by (str): 'all', 'subsidiary', 'affiliate'. Default 'all'.\n        ";return A._dispatch('subsidiaries',filter_by=filter_by)
	def news(A):'Get company news.';return A._dispatch('news')
	def events(A):'Get company events.';return A._dispatch('events')
	def margin_ratio(A):'Get margin lending ratio for the company across brokers.';return A._dispatch('margin_ratio')