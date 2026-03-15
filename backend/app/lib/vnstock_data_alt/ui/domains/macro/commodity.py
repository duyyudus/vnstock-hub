'\nCommodity Market Domain (Layer 6).\nWraps the `spl.commodity.CommodityPrice` module for global and local commodity prices.\n'
_C='macro.commodity'
_B='GLOBAL'
_A='VN'
import pandas as pd
from app.lib.vnstock_data_alt.ui._base import BaseDetail
from app.lib.vnstock_data_alt.ui.schemas.core import standardize_columns
from app.lib.vnstock_data_alt.ui._registry import MACRO_SOURCES
class CommodityReference(BaseDetail):
	'\n    Access point for fetching commodity prices (Gold, Gas, Oil, Steel, etc.).\n    '
	def __init__(A):super().__init__(symbol='MACRO',domain_name=_C,layer_sources=MACRO_SOURCES)
	def _dispatch_and_format(C,method_name,**G):
		'\n        Dispatches method to SPL CommodityPrice and standardizes columns.\n        ';E='time';D=method_name;B='report_time';A=C._dispatch(D,**G)
		if A.empty:return A
		from app.lib.vnstock_data_alt.ui.config import get_route as H;I,F,F,F=H(C._domain_name,D,C._sources_config);J=_C;A=standardize_columns(A,f"{J}.{D}",I,strict=False)
		if A.index.name==E or A.index.name==B:
			A=A.reset_index()
			if E in A.columns:A=A.rename(columns={E:B})
			A[B]=pd.to_datetime(A[B]);A=A.set_index(B)
		return A
	def gold(A,market=_A,**B):
		"\n        Gold prices.\n        \n        Args:\n            market (str): 'VN' for Vietnam SJC/local gold, 'GLOBAL' for GC=F international futures.\n        "
		if str(market).upper()==_B:return A._dispatch_and_format('gold_global',**B)
		return A._dispatch_and_format('gold_vn',**B)
	def gas(A,market=_A,**B):
		"\n        Gas prices. Note: 'GLOBAL' returns natural gas futures. 'VN' returns aggregated RON/DO prices.\n        "
		if str(market).upper()==_B:return A._dispatch_and_format('gas_natural',**B)
		return A._dispatch_and_format('gas_vn',**B)
	def oil_crude(A,**B):'Crude Oil prices.';return A._dispatch_and_format('oil_crude',**B)
	def coke(A,**B):'Coke (Coal) prices.';return A._dispatch_and_format('coke',**B)
	def steel(A,market=_B,**B):
		"Steel prices. 'GLOBAL' for HRC1!, 'VN' for D10."
		if str(market).upper()==_A:return A._dispatch_and_format('steel_d10',**B)
		return A._dispatch_and_format('steel_hrc',**B)
	def iron_ore(A,**B):'Iron ore prices.';return A._dispatch_and_format('iron_ore',**B)
	def fertilizer_ure(A,**B):'Fertilizer URE prices.';return A._dispatch_and_format('fertilizer_ure',**B)
	def soybean(A,**B):'Soybean prices.';return A._dispatch_and_format('soybean',**B)
	def corn(A,**B):'Corn prices.';return A._dispatch_and_format('corn',**B)
	def sugar(A,**B):'Sugar prices.';return A._dispatch_and_format('sugar',**B)
	def pork(A,market=_A,**B):
		"Pork prices. 'VN' for local North Pig, 'CHINA' for China market."
		if str(market).upper()=='CHINA':return A._dispatch_and_format('pork_china',**B)
		return A._dispatch_and_format('pork_vn',**B)