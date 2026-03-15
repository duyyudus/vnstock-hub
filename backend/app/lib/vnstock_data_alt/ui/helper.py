'\nHelper utilities for vnstock_data Unified UI.\nProvides tools to easily explore API structure and documentation.\n'
_K='Series'
_J='DataFrame'
_I='derivatives'
_H='__name__'
_G='Market'
_F='Reference'
_E='return'
_D=None
_C='Macro'
_B=True
_A=False
import inspect,pandas as pd
from typing import Any,Optional,Set
_DEPRECATED_METHODS={'pe','pb','evaluation','gdp','cpi','industry_prod','import_export','retail','fdi','money_supply','population_labor','exchange_rate','interest_rate',_I}
_BACKWARD_COMPAT_ALIASES={'price_board','history','intraday','price_depth','trading_stats','put_through','matched_by_price','by_group','by_exchange'}
_MARKET_TOPLEVEL_ENDPOINTS_HIDE={'quote'}
def _is_endpoint_method(method):
	'\n    Check if a method is an endpoint (returns data/DataFrame, not a domain object).\n    Endpoints are actual data-returning methods that dispatch to providers,\n    not relay methods that call other layer methods.\n    ';C=method
	try:
		E=getattr(C,'__annotations__',{});D=E.get(_E)
		if D:
			B=str(D)
			if _J in B or _K in B or'dict'in B or'list'in B:
				try:
					A=inspect.getsource(C)
					if'Analytics()'in A or'Macro()'in A or'Reference()'in A:return _A
					return _B
				except(OSError,TypeError):return _B
		try:
			A=inspect.getsource(C)
			if'_dispatch'in A:return _B
		except(OSError,TypeError):pass
		return _A
	except(OSError,TypeError):return _A
def _should_include_method(name,method,is_macro_layer=_A,parent_name=''):
	"Determine if a method should be displayed in the API tree.\n    \n    Args:\n        name: Method name\n        method: Method object\n        is_macro_layer: True if this is the Macro layer\n        parent_name: Parent class/layer name (e.g., 'Market' at top level)\n    ";C=parent_name;B=method;A=name
	if A.startswith('_'):return _A
	if not callable(B)and not isinstance(B,property):return _A
	if A in _BACKWARD_COMPAT_ALIASES:return _A
	if A in _DEPRECATED_METHODS:
		if is_macro_layer or A==_I and C==_F:return _A
	if C==_G and A in _MARKET_TOPLEVEL_ENDPOINTS_HIDE:return _A
	return _B
def _get_public_methods(obj,include_intermediate=_A,parent_name=''):
	"\n    Helper to extract public callable methods from an object, excluding dunders and unwanted methods.\n    \n    Args:\n        obj: Object to extract methods from\n        include_intermediate: If True, include navigation methods (returns domain objects).\n                            If False, only include endpoint methods (returns data).\n        parent_name: Parent class name for context-aware filtering (e.g., 'Market')\n    ";B=[];D=type(obj).__name__;E=D==_C
	for(C,A)in inspect.getmembers(obj,predicate=inspect.ismethod):
		if not _should_include_method(C,A,E,parent_name):continue
		if not include_intermediate:
			if not _is_endpoint_method(A):continue
		B.append((C,A))
	return B
def show_doc(obj):
	"\n    Prints the complete docstring and signature of a function or class.\n    \n    Args:\n        obj: Function, method, class or its name as string (e.g., 'Market', 'Reference').\n             When using strings, the object will be automatically resolved from vnstock_data UI.\n    ";A=obj
	if isinstance(A,str):
		try:
			from app.lib.vnstock_data_alt import ui;B=A.replace('()','').strip()
			if hasattr(ui,B):A=getattr(ui,B)
		except Exception:pass
	if isinstance(A,str):print(f"Could not resolve documentation for: '{A}'");print("Tip: Use the object directly (if imported) or its name as a string (e.g., show_doc('Market')).");return
	try:D=inspect.signature(A);print(f"Signature: [92m{A.__name__}{D}[0m\n")
	except(ValueError,TypeError,AttributeError):pass
	C=inspect.getdoc(A)
	if C:print(C)
	else:E=getattr(A,_H,type(A).__name__);print(f"No documentation available for {E}.")
def show_api(layer=_D,show_navigation=_B):
	"\n    Displays a visual API Tree of available endpoints.\n    Only shows endpoint methods (returning data), hiding backward compatible aliases.\n    \n    Args:\n        layer: (Optional) Limit display to a specific Layer (e.g., Market(), 'Market'). \n               If empty (None), displays all 6 library layers.\n        show_navigation: If True, displays intermediate navigation methods (returning domain objects).\n                        Default is True.\n    ";I='\n\x1b[90mTip: Sử dụng show_doc(node) để đọc docstring.\x1b[0m';H='Analytics';G='Fundamental';F='Insights';R='vnstock_data';B=show_navigation;A=layer
	if isinstance(A,str):
		try:
			from app.lib.vnstock_data_alt import ui;D=A.replace('()','').strip()
			if hasattr(ui,D):
				C=getattr(ui,D)
				if inspect.isclass(C):A=C()
				else:A=C
		except Exception:pass
	def n(method):
		'Check if method returns a domain object (navigation, not data).'
		try:
			A=method.__annotations__.get(_E)
			if A:
				if isinstance(A,str):
					C=_G,_F,F,G,_C,H
					if any(B in A for B in C):return _B
					return _A
				B=str(A)
				if _J not in B and _K not in B and'dict'not in B and'list'not in B:
					try:
						if A.__module__!='builtins':return _B
					except AttributeError:pass
			return _A
		except(AttributeError,TypeError):return _A
	def L(node,prefix='',is_last=_B,title='',level=0,show_nav=_A,parent_name='',title_suffix=''):
		m='property';l='method';k='...';j='│   ';i='    ';h='__class__';Y=show_nav;X=level;T=is_last;S=prefix;Q='├── ';P='└── ';A=node
		if not hasattr(A,h)or type(A).__module__.split('.')[0]not in(R,'__main__'):return
		o=P if T else Q;p=title or(A.__name__ if hasattr(A,_H)else type(A).__name__);Z=type(A).__name__==_C;U=parent_name or type(A).__name__;print(f"{S}{o}[96m{p}[0m{title_suffix}")
		if hasattr(A,h)and not inspect.isroutine(A):
			a=_get_public_methods(A,include_intermediate=_A,parent_name=U)
			for(V,(B,F))in enumerate(a):
				G=V==len(a)-1 if not Y else _A;H=S+(i if T else j);b=''
				try:
					M=F.__annotations__.get(_E)
					if M:
						if hasattr(M,_H):J=M.__name__
						else:
							J=str(M).replace('typing.','').replace('pandas.core.frame.','').replace('pandas.core.series.','')
							if'Optional'in J:J=J.replace('Optional[','').replace(']','')+' | None'
						b=f" -> [90m{J}[0m"
				except Exception:pass
				c=''
				try:
					if hasattr(A,'_domain_name')and hasattr(A,'_sources_config'):from app.lib.vnstock_data_alt.ui.config import get_route as q;r=q(A._domain_name,B,A._sources_config);c=f" [93m[{r[0].upper()}][0m"
				except Exception:pass
				d='';I=inspect.getdoc(F)
				if I:
					D=I.split('\n')[0].strip()
					if len(D)>100:D=D[:97]+k
					d=f" [90m# {D}[0m"
				print(f"{H}{P if G else Q}[92m{B}()[0m{c}{b}{d}")
			if Y:
				K=[]
				for(B,F)in inspect.getmembers(A,predicate=inspect.ismethod):
					if not _should_include_method(B,F,is_macro_layer=Z,parent_name=U):continue
					if n(F):K.append((B,F,l))
				for(B,e)in inspect.getmembers(type(A)):
					if isinstance(e,property):
						if not _should_include_method(B,e,is_macro_layer=Z,parent_name=U):continue
						try:
							W=getattr(A,B)
							if W is not _D and type(W).__module__.startswith(R):K.append((B,W,m))
						except Exception as f:pass
				K.sort(key=lambda x:x[0])
				for(V,(B,N,g))in enumerate(K):
					G=V==len(K)-1;H=S+(i if T else j);O=''
					if g==l:I=inspect.getdoc(N)
					else:s=getattr(type(A),B);I=inspect.getdoc(s)
					if I:
						D=I.split('\n')[0].strip()
						if len(D)>100:D=D[:97]+k
						O=f" [90m# {D}[0m"
					if g==m:L(N,H,G,B,X+1,show_nav=_B,parent_name='',title_suffix=O)
					else:
						try:
							t=inspect.signature(N);E={}
							for C in t.parameters.values():
								if C.default==inspect.Parameter.empty and C.name!='self':
									if C.name=='symbol':
										if'crypto'in B:E[C.name]='BTC'
										elif'forex'in B:E[C.name]='USDVND'
										elif'commodity'in B:E[C.name]='GC=F'
										else:E[C.name]='VIC'
									elif C.name=='index':E[C.name]='VNINDEX'
									else:E[C.name]=_D
							u=N(**E);L(u,H,G,B+'()',X+1,show_nav=_B,parent_name='',title_suffix=O)
						except Exception as f:print(f"{H}{P if G else Q}[94m{B}()[0m{O}")
						except Exception as f:print(f"{H}{P if G else Q}[94m{B}()  [Navigation][0m")
	if A is not _D:print('\n\x1b[1mAPI STRUCTURE TREE\x1b[0m');L(A,'',_B,'',0,show_nav=B)
	else:
		from app.lib.vnstock_data_alt.ui import Reference as J,Market as K,Insights as M,Fundamental as N,Macro,Analytics as O;print('\n\x1b[1mAPI STRUCTURE TREE - VNSTOCK_DATA (Unified UI Endpoints)\x1b[0m');print(R);E=[(_F,J()),(_G,K()),(G,N()),(H,O()),(_C,Macro()),(F,M())]
		for(P,(Q,S))in enumerate(E):T=P==len(E)-1;L(S,'',T,Q,0,show_nav=B)
	if B:print(I);print('\x1b[90m[Navigation] = Intermediate methods returning domain objects\x1b[0m\n')
	else:print(I);print('\x1b[90mHint: show_api() để hiển thị cây đầy đủ với navigation methods.\x1b[0m\n')
