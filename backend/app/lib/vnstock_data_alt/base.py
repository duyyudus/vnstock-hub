from typing import Optional
import inspect
from abc import ABC
from functools import wraps
from tenacity import retry,stop_after_attempt,wait_exponential
from app.lib.vnstock_data_alt.config import Config
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
def dynamic_method(func):
	'\n    Decorator for adapter methods:\n    - Ensures the loaded provider supports this method\n    - Filters kwargs to only those the provider’s signature accepts\n    ';A=func.__name__
	@wraps(func)
	def B(self,*D,**E):
		B=self
		if not hasattr(B._provider,A):raise NotImplementedError("Source '"+B.source+"' does not support '"+A+"'")
		C=getattr(B._provider,A);F=inspect.signature(C);G={A:B for(A,B)in E.items()if A in F.parameters};return C(*D,**G)
	return B
class BaseAdapter(ABC):
	'\n    Base adapter that uses ProviderRegistry to discover and instantiate\n    providers from both explorer and connector packages.\n    ';_module_name:str
	def __init__(A,source,symbol=None,**J):
		I='symbol';D=source;B=symbol;A.source=D;A.symbol=B
		try:E=ProviderRegistry.get(A._module_name,D)
		except ValueError as F:raise ValueError(str(F))from F
		G=inspect.signature(E.__init__);C={}
		if B is not None and I in G.parameters:C[I]=B
		for(H,K)in J.items():
			if H in G.parameters:C[H]=K
		A._provider=E(**C)
	def __getattr__(A,name):return getattr(A._provider,name)
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	def history(self,*A,**B):return self._provider.history(*A,**B)