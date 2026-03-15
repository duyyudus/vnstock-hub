_C="()' requires a data source. "
_B="Method '"
_A=None
from typing import Any,Union
from functools import wraps
from tenacity import retry,stop_after_attempt,wait_exponential
from app.lib._vnstock_shared.common import indices as shared_indices
from app.lib.vnstock_data_alt.config import Config
from app.lib.vnstock_data_alt.base import BaseAdapter,dynamic_method
from app.lib.vnstock_data_alt.enums import IndexGroup
def source_required(func):
	'Decorator to ensure source is specified for source-specific methods.\n    \n    This decorator is applied BEFORE @retry so that ValueError from missing source\n    is raised immediately without retry logic.\n    ';A=func;B=A.__name__
	@wraps(A)
	def C(self,*C,**D):
		if self._provider is _A:raise ValueError(_B+B+_C+"Initialize with Listing(source='vci'|'kbs'|'vnd') "+'or use source-independent methods like all_indices(), indices_by_group().')
		return A(self,*C,**D)
	return C
def _normalize_index_group(group):
	"\n    Normalize short index group names to their full names.\n    \n    Accepts both enum values and string values:\n        'HOSE' or IndexGroup.HOSE -> 'HOSE Indices'\n        'SECTOR' or IndexGroup.SECTOR -> 'Sector Indices'\n        'INVESTMENT' or IndexGroup.INVESTMENT -> 'Investment Indices'\n        'VNX' or IndexGroup.VNX -> 'VNX Indices'\n    \n    Args:\n        group: Index group name or IndexGroup enum value\n        \n    Returns:\n        Full name of the index group, or original value if not mapped\n    ";A=group
	if A is _A:return
	if isinstance(A,IndexGroup):return A.full_name
	B=str(A).upper().strip();C={'HOSE':'HOSE Indices','SECTOR':'Sector Indices','INVESTMENT':'Investment Indices','VNX':'VNX Indices'};return C.get(B,A)
class Listing(BaseAdapter):
	_module_name='listing';'\n    Adapter for listing data:\n      - all_symbols\n      - symbols_by_industries\n      - symbols_by_exchange\n      - industries_icb\n      - symbols_by_group\n      - all_future_indices\n      - all_government_bonds\n      - all_covered_warrant\n      - all_bonds\n\n    Usage:\n        lst = Listing(source="vci", random_agent=False, show_log=True)\n        df = lst.all_symbols(to_df=True)\n        df2 = lst.symbols_by_exchange(lang="en")\n        idx = lst.industries_icb()\n        group = lst.symbols_by_group(group="VN30")\n        fu = lst.all_future_indices()\n    '
	def __init__(B,source=_A,random_agent=False,show_log=False):
		C=show_log;A=source
		if A is not _A and A.lower()not in['vci','kbs','vnd']:raise ValueError("Lớp Listing chỉ nhận giá trị tham số source là 'VCI', 'KBS' hoặc 'VND'.")
		B.source=A;B.show_log=C;B._provider=_A
		if A is not _A:super().__init__(source=A,random_agent=random_agent,show_log=C)
	def _ensure_provider(A,method_name):
		'Ensure provider is initialized for source-specific methods.'
		if A._provider is _A:raise ValueError(_B+method_name+_C+"Initialize with Listing(source='vci'|'kbs'|'vnd') or use "+'source-independent methods like all_indices(), '+'indices_by_group().')
	@source_required
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def all_symbols(self,*A,**B):'Retrieve all symbols (filtered to STOCK).'
	@source_required
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def symbols_by_industries(self,*A,**B):'Retrieve symbols grouped by ICB industries.'
	@source_required
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def symbols_by_exchange(self,*A,**B):'Retrieve symbols by exchange/board.'
	@source_required
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def industries_icb(self,*A,**B):'Retrieve ICB code hierarchy and mapping.'
	@source_required
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def symbols_by_group(self,*A,**B):'Retrieve symbols by predefined group (VN30, HNX30, CW, etc.).'
	@source_required
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	def all_future_indices(self,**A):"Retrieve all futures indices (group='FU_INDEX').";return self.symbols_by_group(group='FU_INDEX',**A)
	@source_required
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@source_required
	@dynamic_method
	def all_government_bonds(self,*A,**B):'Retrieve all government bonds.'
	@source_required
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	def all_covered_warrant(self,**A):"Retrieve all covered warrants (group='CW').";return self.symbols_by_group(group='CW',**A)
	@source_required
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	def all_bonds(self,**A):"Retrieve all bonds (group='BOND').";return self.symbols_by_group(group='BOND',**A)
	@source_required
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def all_etf(self,*A,**B):'Retrieve all ETF (exchange-traded funds).'
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	def all_indices(self,*B,**C):'\n        Retrieve all standardized market indices with metadata.\n\n        Returns:\n            DataFrame with columns:\n            - symbol: Index symbol (VN30, VNIT, etc.)\n            - name: Index name\n            - description: Vietnamese description\n            - full_name: Full English name\n            - group: Index group (HOSE Indices, Sector Indices, etc.)\n            - index_id: Unique index ID\n            - sector_id: ICB sector ID (for sector indices only)\n        ';return shared_indices.get_all_indices()
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	def indices_by_group(self,*B,**C):
		"\n        Retrieve standardized market indices by group/category.\n\n        Args:\n            group (str or IndexGroup): Index group name. Accepts both short names, \n                enum values, and full names:\n                - 'HOSE', IndexGroup.HOSE, or 'HOSE Indices' - HOSE market indices\n                - 'SECTOR', IndexGroup.SECTOR, or 'Sector Indices' - Sector indices\n                - 'INVESTMENT', IndexGroup.INVESTMENT, or 'Investment Indices' - Investment indices\n                - 'VNX', IndexGroup.VNX, or 'VNX Indices' - VNX market indices\n\n        Returns:\n            DataFrame with index information filtered by group,\n            or None if group not found.\n\n        Examples:\n            >>> from app.lib.vnstock_data_alt.api.listing import Listing\n            >>> from app.lib.vnstock_data_alt.enums import IndexGroup\n            >>> lst = Listing()\n            >>> # Using string short names\n            >>> lst.indices_by_group('HOSE')\n            >>> # Using enum values\n            >>> lst.indices_by_group(IndexGroup.SECTOR)\n            >>> # Using full names\n            >>> lst.indices_by_group('HOSE Indices')\n        ";A=C.get('group')
		if A is _A and len(B)>0:A=B[0]
		A=_normalize_index_group(A);return shared_indices.get_indices_by_group(A)
