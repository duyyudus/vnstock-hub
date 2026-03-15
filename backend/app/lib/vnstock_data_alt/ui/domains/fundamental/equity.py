'\nEquity Fundamental Domain (Layer 3).\nWraps the `kbs.financial.Finance` module.\n'
_A='display_mode'
import pandas as pd
from app.lib.vnstock_data_alt.ui._base import BaseDetail
from app.lib.vnstock_data_alt.ui.schemas.core import standardize_columns
from app.lib.vnstock_data_alt.ui._registry import FUNDAMENTAL_SOURCES
class EquityFundamental(BaseDetail):
	'\n    Access point for fetching company financial statements and valuation ratios.\n    '
	def __init__(A,symbol):super().__init__(symbol=symbol,domain_name='equity.fundamental',layer_sources=FUNDAMENTAL_SOURCES)
	def _dispatch_and_format(C,method_name,**E):
		'\n        Dispatches method to KBS Finance and standardizes columns without strict trimming.\n        Financial statements vary widely depending on sector, so we use `strict=False`\n        to preserve all snake_cased accounting fields.\n        ';I='period';H='item_id';D=method_name
		if _A not in E:from app.lib.vnstock_data_alt.explorer.kbs.financial import FieldDisplayMode as J;E[_A]=J.STD
		B=C._dispatch(D,**E)
		if B.empty:return B
		from app.lib.vnstock_data_alt.ui.config import get_route as K;L,F,F,F=K(C._domain_name,D,C._sources_config);M='fundamental.equity';B=standardize_columns(B,f"{M}.{D}",L,strict=False)
		if H in B.columns:
			A=B.set_index(H);G=[A for A in A.columns if str(A).replace('-','Q').replace('Q','1').isdigit()]
			if G:A=A[G].transpose();A=A.reset_index();A=A.rename(columns={'index':I});A.insert(1,'ticker',C.symbol);A=A.sort_values(I).reset_index(drop=True);return A
		return B
	def ratio(A,**B):'\n        Extracts key financial ratios (P/E, ROE, Debt/Equity, etc.).\n        \n        Returns:\n            pd.DataFrame: Ratios pivoted by period.\n        ';return A._dispatch_and_format('ratio',**B)
	def income_statement(A,**B):'\n        Extracts Income Statement.\n        ';return A._dispatch_and_format('income_statement',**B)
	def balance_sheet(A,**B):'\n        Extracts Balance Sheet.\n        ';return A._dispatch_and_format('balance_sheet',**B)
	def cash_flow(A,**B):'\n        Extracts Cash Flow statement.\n        ';return A._dispatch_and_format('cash_flow',**B)
	def note(A,**C):
		"\n        Extracts Footnotes (Thuyết minh Báo cáo tài chính).\n        Note: This method doesn't accept display_mode parameter.\n        ";E='note';C.pop(_A,None);B=A._dispatch(E,**C)
		if B.empty:return B
		from app.lib.vnstock_data_alt.ui.config import get_route as F;from app.lib.vnstock_data_alt.ui.schemas.core import standardize_columns as G;H,D,D,D=F(A._domain_name,E,A._sources_config);return G(B,f"{A._domain_name}.note",H,strict=False)