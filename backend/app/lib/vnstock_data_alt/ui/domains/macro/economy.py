'\nEconomy Reference Domain (Layer 6).\nWraps the `mbk.macro.Macro` module for generic macroeconomic indicators.\n'
_A='macro.economy'
import pandas as pd
from app.lib.vnstock_data_alt.ui._base import BaseDetail
from app.lib.vnstock_data_alt.ui.schemas.core import standardize_columns
from app.lib.vnstock_data_alt.ui._registry import MACRO_SOURCES
class EconomyReference(BaseDetail):
	'\n    Access point for fetching macroeconomic data such as GDP, CPI, FDI, etc.\n    '
	def __init__(A):super().__init__(symbol='MACRO',domain_name=_A,layer_sources=MACRO_SOURCES)
	def _dispatch_and_format(A,method_name,**E):
		'\n        Dispatches method to MBK Macro and standardizes columns without strict trimming.\n        ';B=method_name;C=A._dispatch(B,**E)
		if C.empty:return C
		from app.lib.vnstock_data_alt.ui.config import get_route as F;G,D,D,D=F(A._domain_name,B,A._sources_config);H=_A;return standardize_columns(C,f"{H}.{B}",G,strict=False)
	def gdp(A,**B):'GDP data.';return A._dispatch_and_format('gdp',**B)
	def cpi(A,**B):'Consumer Price Index data.';return A._dispatch_and_format('cpi',**B)
	def industry_prod(A,**B):'Industrial Production data.';return A._dispatch_and_format('industry_prod',**B)
	def import_export(A,**B):'Import/Export macro data.';return A._dispatch_and_format('import_export',**B)
	def retail(A,**B):'Retail macro data.';return A._dispatch_and_format('retail',**B)
	def fdi(A,**B):'Foreign Direct Investment data.';return A._dispatch_and_format('fdi',**B)
	def money_supply(A,**B):'Money supply data.';return A._dispatch_and_format('money_supply',**B)
	def population_labor(A,**B):'Population and Labor data.';return A._dispatch_and_format('population_labor',**B)