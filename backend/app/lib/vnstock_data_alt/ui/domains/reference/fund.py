'\nFund Reference Domain (Layer 1).\nWraps the `fmarket.fund` module for global fund operations like listing.\n'
import pandas as pd
from app.lib.vnstock_data_alt.ui._base import BaseDomain
from app.lib.vnstock_data_alt.ui._registry import REFERENCE_SOURCES
from app.lib.vnstock_data_alt.ui.schemas.core import standardize_columns
class FundReference(BaseDomain):
	'\n    Access point for listing mutual funds.\n    '
	def __init__(A):super().__init__(domain_name='reference.fund',layer_sources=REFERENCE_SOURCES)
	def list(A):
		'\n        Extracts the list of all available mutual funds.\n        ';D='list';B=A._dispatch(D)
		if B is None or B.empty:return pd.DataFrame()
		from app.lib.vnstock_data_alt.ui.config import get_route as E;F,C,C,C=E(A._domain_name,D,A._sources_config);return standardize_columns(B,f"{A._domain_name}.list",F,strict=True)