_G='show_doc'
_F='show_api'
_E='Analytics'
_D='Fundamental'
_C='Insights'
_B='Market'
_A='Reference'
__version__='3.0.0'
from.api.quote import Quote
from.api.company import Company
from.api.financial import Finance
from.api.listing import Listing
from.api.trading import Trading
from.api.commodity import CommodityPrice
from.api.insight import TopStock
from.explorer.fmarket import Fund
from.enums import IndexGroup
from typing import TYPE_CHECKING,Any
if TYPE_CHECKING:from.ui.reference import Reference;from.ui.market import Market;from.ui.insights import Insights;from.ui.fundamental import Fundamental;from.ui.macro import Macro;from.ui.analytics import Analytics;from.ui.helper import show_api,show_doc
__all__=['Quote','Company','Finance','Listing','Trading','CommodityPrice','TopStock','Fund',_A,_B,_C,_D,'Macro',_E,_F,_G]
def __getattr__(name):
	'\n    Lazy load UI modules at root level using PEP 562. \n    Allows IDE autocomplete and type hints to work correctly.\n    ';A=name
	if A==_A:from.ui.reference import Reference as B;return B
	elif A==_B:from.ui.market import Market as C;return C
	elif A==_C:from.ui.insights import Insights as D;return D
	elif A==_D:from.ui.fundamental import Fundamental as E;return E
	elif A=='Macro':from.ui.macro import Macro as F;return F
	elif A==_E:from.ui.analytics import Analytics as G;return G
	elif A==_F:from.ui.helper import show_api as H;return H
	elif A==_G:from.ui.helper import show_doc as I;return I
	raise AttributeError(f"module {__name__!r} has no attribute {A!r}")
try:from.core.utils.startup import display_startup_messages;display_startup_messages()
except Exception:pass