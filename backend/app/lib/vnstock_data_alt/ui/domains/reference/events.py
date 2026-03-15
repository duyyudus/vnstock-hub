'\nEvents Reference domain.\n'
_A=None
from typing import Optional
import pandas as pd
from app.lib.vnstock_data_alt.ui._base import BaseDomain
from app.lib.vnstock_data_alt.ui._registry import REFERENCE_SOURCES
class EventsReference(BaseDomain):
	'\n    Events Reference Data.\n    Provides access to market events, news, and other time-bound occurrences.\n    '
	def __init__(A):super().__init__(domain_name='events',layer_sources=REFERENCE_SOURCES)
	def calendar(A,start=_A,end=_A,event_type=_A,page=0,limit=20000):"\n        Retrieve events calendar (dividends, AGM, new listings, ...) from the default data source.\n        \n        Args:\n            start (str, optional): Start date (YYYY-MM-DD). Defaults to the current date.\n            end (str, optional): End date (YYYY-MM-DD). Defaults to the current date.\n            event_type (str, optional): Type of event or event group:\n                - 'dividend': Returns dividends, share issuance\n                - 'insider': Returns insider trading, major shareholders\n                - 'agm': Returns shareholder meetings\n                - 'others': Returns other fluctuations\n                - Or a specific eventCode (e.g., 'ISS,DIV')\n            page (int): Page index. Defaults to 0.\n            limit (int): Number of records per page. Defaults to 20000.\n            \n        Returns:\n            pd.DataFrame: Standardized list of events.\n        ";return A._dispatch('calendar',start=start,end=end,event_type=event_type,page=page,limit=limit)
	def market(b,start=_A,end=_A,event_type=_A):
		"\n        Retrieve special stock market events (holidays, system incidents, ...)\n        \n        Uses a static internal database from the vnstock library.\n        \n        Args:\n            start (str, optional): Start date (YYYY-MM-DD). Defaults to all events.\n            end (str, optional): End date (YYYY-MM-DD). Defaults to all events.\n            event_type (str, optional): Event type ('Holiday', 'Suspension', 'Compensation').\n            \n        Returns:\n            pd.DataFrame: List of market events.\n        ";P='events.market';G=event_type;F=start;D='MARKET_EVENTS';C='date';B={}
		try:from app.lib._vnstock_shared.core.utils import market_events as Q;B=getattr(Q,D,{})
		except ImportError:
			try:import importlib;E=importlib.import_module('vnstock.core.utils.market_events');B=getattr(E,D,{})
			except ImportError:
				try:E=importlib.import_module('vnstock.core.utils.market-events');B=getattr(E,D,{})
				except ImportError:
					import sys;import os
					try:
						import app.lib.vnstock_alt as R;S=os.path.join(os.path.dirname(R.__file__),'core','utils')
						for T in['market_events.py','market-events.py']:
							H=os.path.join(S,T)
							if os.path.exists(H):import importlib.util;I=importlib.util.spec_from_file_location('market_events_ext',H);J=importlib.util.module_from_spec(I);I.loader.exec_module(J);B=getattr(J,D,{});break
					except Exception as U:from app.lib._vnstock_shared.core.utils.logger import get_logger as V;W=V(__name__);W.warning(f"Failed to load market events: {str(U)}. Please ensure you have vnstock v3.4.3+ installed.");pass
		if not B:return pd.DataFrame()
		K=[]
		for(X,Y)in B.items():L={C:X};L.update(Y);K.append(L)
		A=pd.DataFrame(K);from app.lib.vnstock_data_alt.ui._registry import REFERENCE_SOURCES;from app.lib.vnstock_data_alt.ui._base import BaseDomain;from app.lib.vnstock_data_alt.ui.schemas.events import SCHEMA_MAP as Z,STANDARD_COLUMNS as a;M=Z.get(P,{}).get('vnstock',{});N=a.get(P,[])
		if not A.empty and M:A.rename(columns=M,inplace=True)
		for O in N:
			if O not in A.columns:A[O]=pd.NA
		A=A[N]
		if not A.empty:
			A[C]=pd.to_datetime(A[C])
			if F:
				try:A=A[A[C]>=pd.to_datetime(F)]
				except Exception:pass
			if end:
				try:A=A[A[C]<=pd.to_datetime(end)]
				except Exception:pass
			if G:A=A[A['event_type'].str.lower()==G.lower()]
		return A.reset_index(drop=True)