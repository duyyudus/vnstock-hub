'\nBond Reference Domain (Layer 1).\n'
import pandas as pd
from app.lib.vnstock_data_alt.ui._base import BaseDomain
from app.lib.vnstock_data_alt.ui._registry import REFERENCE_SOURCES
class BondReference(BaseDomain):
	'\n    Bond Reference Data (Layer 1).\n    '
	def __init__(A):super().__init__(domain_name='bond',layer_sources=REFERENCE_SOURCES)
	def list(C,bond_type='all'):
		"\n        List bonds available in the market.\n        \n        Args:\n            bond_type (str): Type of bond to filter ('all', 'corporate', 'government'). Default is 'all'.\n                             - If 'corporate' or 'government', returns a pandas Series of symbols.\n                             - If 'all', returns a pandas DataFrame with 'symbol' and 'type' columns.\n                             \n        Note: The government bond data source might be restricted in some environments (e.g., Google Colab).\n        ";L=False;K='empty';H=True;F='type';E='symbol';D=bond_type;B='government';A='corporate';import logging as M;I=['all',A,B]
		if D not in I:raise ValueError(f"Invalid bond_type: {D}. Must be one of {I}.")
		if D==A:return C._dispatch(A)
		elif D==B:return C._dispatch(B)
		J=C._dispatch(A)
		try:G=C._dispatch(B)
		except Exception as N:M.warning(f"Could not fetch government bonds (source may be blocked): {N}");G=pd.Series(dtype='object')
		O=pd.DataFrame({E:J,F:A})if getattr(J,K,H)is L else pd.DataFrame(columns=[E,F]);P=pd.DataFrame({E:G,F:B})if getattr(G,K,H)is L else pd.DataFrame(columns=[E,F]);return pd.concat([O,P],ignore_index=H)