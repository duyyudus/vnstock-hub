'\nCurrency Reference Domain (Layer 6).\nWraps the `mbk.macro.Macro.exchange_rate` module.\n'
_A='macro.currency'
import pandas as pd
from app.lib.vnstock_data_alt.ui._base import BaseDetail
from app.lib.vnstock_data_alt.ui.schemas.core import standardize_columns
from app.lib.vnstock_data_alt.ui._registry import MACRO_SOURCES
class CurrencyReference(BaseDetail):
	'\n    Access point for fetching currency exchange rates.\n    '
	def __init__(A):super().__init__(symbol='MACRO',domain_name=_A,layer_sources=MACRO_SOURCES)
	def _dispatch_and_format(A,method_name,**E):
		'\n        Dispatches method to MBK Macro and standardizes columns without strict trimming.\n        ';B=method_name;C=A._dispatch(B,**E)
		if C.empty:return C
		from app.lib.vnstock_data_alt.ui.config import get_route as F;G,D,D,D=F(A._domain_name,B,A._sources_config);H=_A;return standardize_columns(C,f"{H}.{B}",G,strict=False)
	def exchange_rate(A,**B):'Foreign exchange rates.';return A._dispatch_and_format('exchange_rate',**B)
	def interest_rate(A,**B):'Interest rates data.';return A._dispatch_and_format('interest_rate',**B)