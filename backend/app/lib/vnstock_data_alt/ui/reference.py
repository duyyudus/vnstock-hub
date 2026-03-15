'\nReference Layer Entry Point.\n'
from app.lib.vnstock_data_alt.ui.domains.reference.company import CompanyReference
from app.lib.vnstock_data_alt.ui.domains.reference.industry import IndustryReference
from app.lib.vnstock_data_alt.ui.domains.reference.equity import EquityReference
from app.lib.vnstock_data_alt.ui.domains.reference.index import IndexReference
from app.lib.vnstock_data_alt.ui.domains.reference.derivatives import DerivativesReference,WarrantReference,FuturesReference
from app.lib.vnstock_data_alt.ui.domains.reference.fund import FundReference
from app.lib.vnstock_data_alt.ui.domains.reference.etf import ETFReference
from app.lib.vnstock_data_alt.ui.domains.reference.bond import BondReference
from app.lib.vnstock_data_alt.ui.domains.reference.events import EventsReference
from app.lib.vnstock_data_alt.ui.domains.reference.search import SearchReference
class Reference:
	'\n    Reference Data Layer (Layer 1).\n    Provides access to static/master data for various domains.\n    '
	def company(A,symbol):"\n        Access company-specific reference data.\n        \n        Args:\n            symbol (str): Ticker symbol (e.g., 'VNM', 'TCB').\n        ";return CompanyReference(symbol)
	def futures(A,symbol=None):"\n        Access index futures reference data (listing or symbol-specific info).\n        \n        Args:\n            symbol (str, optional): Futures symbol (e.g., 'VN30F2503', 'VN30F1M').\n                                    If None, returns listing interface.\n            \n        Example:\n            r = Reference()\n            # List all futures indices\n            futures_list = r.futures().list()\n            \n            # Get specific futures info\n            futures_info = r.futures('VN30F2503').info()\n        ";return FuturesReference(symbol)
	def warrant(A,symbol=None):"\n        Access covered warrant reference data (info, specifications, pricing).\n        \n        Args:\n            symbol (str): Warrant symbol (e.g., 'CACB2511', 'CACB25C100').\n            \n        Example:\n            r = Reference()\n            warrant_info = r.warrant('CACB2511').info()\n        ";return WarrantReference(symbol)
	@property
	def industry(self):'Access industry reference data.';return IndustryReference()
	@property
	def fund(self):'\n        Master data for Mutual Funds (Chứng Chỉ Quỹ).\n        ';return FundReference()
	@property
	def etf(self):'Access ETF reference data.';return ETFReference()
	@property
	def equity(self):'Access equity reference data.';return EquityReference()
	@property
	def index(self):'Access index reference data.';return IndexReference()
	@property
	def bond(self):'Access bond reference data.';return BondReference()
	@property
	def events(self):'Access events reference data (calendar, etc.).';return EventsReference()
	@property
	def search(self):'Access global symbol search.';return SearchReference()
	def derivatives(B):"\n        [DEPRECATED] Access derivatives reference data.\n        \n        To fix: Replace `Reference().derivatives().futures(symbol).info()` with `Reference().futures(symbol).info()`\n        \n        Examples:\n            # Old way (will raise warning):\n            r.derivatives().futures('VN30F2503').info()\n            \n            # New direct way:\n            r.futures('VN30F2503').info()\n        \n        Returns:\n            DerivativesReference: Provides access to .warrant() and .futures() sub-domains.\n        ";import warnings as A;A.warn("Reference.derivatives() is deprecated. Use Reference.futures(symbol) or Reference.warrant(symbol) directly. Example: r.futures('VN30F2503').info() instead of r.derivatives().futures('VN30F2503').info(). Deprecated method will be removed after 31/8/2026.",DeprecationWarning,stacklevel=2);return DerivativesReference()