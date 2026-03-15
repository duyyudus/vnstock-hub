'\nWarrant domain reference structure.\n'
_C='derivatives.futures'
_B='derivatives.warrant'
_A=None
import pandas as pd
from app.lib.vnstock_data_alt.ui._base import BaseDetail,BaseDomain
from app.lib.vnstock_data_alt.ui._registry import REFERENCE_SOURCES
class WarrantReference(BaseDetail):
	'\n    Reference Data Layer for Covered Warrants.\n    '
	def __init__(A,symbol=_A):
		B=symbol
		if B is _A:super().__init__('WARRANT',_B,REFERENCE_SOURCES);A._is_listing=True
		else:super().__init__(B,_B,REFERENCE_SOURCES);A._is_listing=False
	def list(B):'\n        List all available covered warrants.\n        ';A=BaseDomain(_B,REFERENCE_SOURCES);return A._dispatch('list')
	def info(A):'\n        Get info and realtime information for the specific covered warrant.\n        \n        Returns:\n            pd.DataFrame: A standardized dataframe with warrant details.\n        ';return A._dispatch('info')
class FuturesReference(BaseDetail):
	'\n    Reference Data Layer for Index Futures.\n    '
	def __init__(A,symbol=_A):
		B=symbol
		if B is _A:super().__init__('FUTURES',_C,REFERENCE_SOURCES);A._is_listing=True
		else:super().__init__(B,_C,REFERENCE_SOURCES);A._is_listing=False
	def list(B):'\n        List all available futures indices with metadata.\n        \n        Returns:\n            DataFrame with futures information\n        ';A=BaseDomain(_C,REFERENCE_SOURCES);return A._dispatch('list')
	def info(A):'\n        Get info and realtime information for the specific index future.\n        ';return A._dispatch('info')
class DerivativesReference:
	'\n    Derivatives domain wrapper containing Futures and Warrants.\n    '
	def warrant(A,symbol=_A):"\n        Access covered warrant profile and reference data.\n        \n        Args:\n            symbol (str, optional): Warrant symbol (e.g., 'CACB2511').\n        ";return WarrantReference(symbol)
	def futures(A,symbol=_A):"\n        Access index futures profile and reference data.\n        \n        Args:\n            symbol (str, optional): Future symbol (e.g., 'VN30F2503').\n        ";return FuturesReference(symbol)