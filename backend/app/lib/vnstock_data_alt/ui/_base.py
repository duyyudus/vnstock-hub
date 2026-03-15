'\nBase classes for UI domains and details.\n'
from typing import Optional,Any,Dict,List
import pandas as pd
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
from app.lib.vnstock_data_alt.ui._registry import REFERENCE_SOURCES,MARKET_SOURCES,FUNDAMENTAL_SOURCES
class BaseDomain:
	'\n    Abstract base class for domain objects (e.g., Equity, Industry).\n    Handles dispatching calls to the registered data provider.\n    '
	def __init__(A,domain_name,layer_sources):A._domain_name=domain_name;A._sources_config=layer_sources
	def _dispatch(D,method_name,**E):
		'\n        Dispatches a method call to the underlying provider.\n        ';J=method_name;from app.lib.vnstock_data_alt.ui.config import get_route as N;A,B,O,F=N(D._domain_name,J,D._sources_config);from app.lib.vnstock_data_alt.core.registry import ProviderRegistry as G
		if not G.is_registered(B,A):
			import importlib as P
			try:P.import_module(f"vnstock_data.explorer.{A}.{B}")
			except ImportError:pass
		C=G.get(B,A)
		if not C:C=G.get(B,A)
		if not C:raise ValueError(f"Provider not found for {A}.{B}")
		try:H=C()
		except TypeError:H=C()
		if not hasattr(H,F):raise AttributeError(f"Provider {O} has no method {F}")
		K=getattr(H,F);import inspect as L;Q=L.signature(K);M=Q.parameters;R=any(A.kind==L.Parameter.VAR_KEYWORD for A in M.values())
		if not R:E={A:B for(A,B)in E.items()if A in M}
		I=K(**E);S=f"{D._domain_name}.{J}";from app.lib.vnstock_data_alt.ui.schemas import standardize_columns as T;I=T(I,S,A,strict=True);return I
class BaseDetail:
	'\n    Abstract base class for symbol-specific detailed views (e.g., Company Profile).\n    '
	def __init__(A,symbol,domain_name,layer_sources):A.symbol=symbol;A._domain_name=domain_name;A._sources_config=layer_sources
	def _dispatch(A,method_name,**D):
		'\n        Dispatches a method call to the underlying provider for a specific symbol.\n        ';I=method_name;from app.lib.vnstock_data_alt.ui.config import get_route as N;B,C,O,E=N(A._domain_name,I,A._sources_config);from app.lib.vnstock_data_alt.core.registry import ProviderRegistry as J
		if not J.is_registered(C,B):
			import importlib as P
			try:P.import_module(f"vnstock_data.explorer.{B}.{C}")
			except ImportError:pass
		F=J.get(C,B)
		if not F:raise ValueError(f"Provider not found for {B}.{C}")
		try:G=F(symbol=A.symbol)
		except TypeError:G=F(A.symbol)
		if not hasattr(G,E):raise AttributeError(f"Provider {O} has no method {E}")
		K=getattr(G,E);import inspect as L;Q=L.signature(K);M=Q.parameters;R=any(A.kind==L.Parameter.VAR_KEYWORD for A in M.values())
		if not R:D={A:B for(A,B)in D.items()if A in M}
		H=K(**D);S=f"{A._domain_name}.{I}";from app.lib.vnstock_data_alt.ui.schemas import standardize_columns as T;H=T(H,S,B,strict=True);return H