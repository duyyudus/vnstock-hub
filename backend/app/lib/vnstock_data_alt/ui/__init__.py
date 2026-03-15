'\nUI Module - Unified Interface for vnstock-data\n'
_G='show_doc'
_F='show_api'
_E='Analytics'
_D='Fundamental'
_C='Insights'
_B='Market'
_A='Reference'
from typing import TYPE_CHECKING,Any
if TYPE_CHECKING:from app.lib.vnstock_data_alt.ui.reference import Reference;from app.lib.vnstock_data_alt.ui.market import Market;from app.lib.vnstock_data_alt.ui.insights import Insights;from app.lib.vnstock_data_alt.ui.fundamental import Fundamental;from app.lib.vnstock_data_alt.ui.macro import Macro;from app.lib.vnstock_data_alt.ui.analytics import Analytics
__all__=[_A,_B,_C,_D,'Macro',_E,_F,_G]
def __getattr__(name):
	'\n    Lazy load UI modules using PEP 562. \n    Allows IDE autocomplete and type hints to work correctly.\n    ';A=name
	if A==_A:from app.lib.vnstock_data_alt.ui.reference import Reference as B;return B
	elif A==_B:from app.lib.vnstock_data_alt.ui.market import Market as C;return C
	elif A==_C:from app.lib.vnstock_data_alt.ui.insights import Insights as D;return D
	elif A==_D:from app.lib.vnstock_data_alt.ui.fundamental import Fundamental as E;return E
	elif A=='Macro':from app.lib.vnstock_data_alt.ui.macro import Macro as F;return F
	elif A==_E:from app.lib.vnstock_data_alt.ui.analytics import Analytics as G;return G
	elif A==_F:from app.lib.vnstock_data_alt.ui.helper import show_api as H;return H
	elif A==_G:from app.lib.vnstock_data_alt.ui.helper import show_doc as I;return I
	raise AttributeError(f"module {__name__!r} has no attribute {A!r}")