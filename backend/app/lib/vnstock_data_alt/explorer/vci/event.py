'\nVCI Events Explorer module.\n'
_B=False
_A=None
from typing import Dict,Optional,Union
import pandas as pd
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib._vnstock_shared.core.utils.parser import camel_to_snake
from app.lib.vnstock_data_alt.core.utils.client import send_request
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
from app.lib.vnstock_data_alt.explorer.vci.const import _VCI_EVENTS_URL
from app.lib._vnstock_shared.compat import agg_execution
from datetime import datetime
logger=get_logger(__name__)
class Event:
	'\n    Class to manage event information from the VCI data source.\n    '
	def __init__(A,random_agent=_B,show_log=_B):
		B=show_log;A.headers=get_headers(data_source='VCI',random_agent=random_agent);A.show_log=B
		if not B:logger.setLevel('CRITICAL')
	@agg_execution('VCI.ext')
	def calendar(self,start=_A,end=_A,event_type=_A,page=0,limit=20000,show_log=_B):
		"\n        Retrieve events calendar (dividends, AGM, new listings, ...) from VCI.\n        \n        Args:\n            start (str): Start date (YYYY-MM-DD). If None, defaults to the current date.\n            end (str): End date (YYYY-MM-DD). If None, defaults to the current date.\n            event_type (str, optional): Type of event or event group:\n                - 'dividend': Returns dividends, share issuance (ISS, DIV)\n                - 'insider': Returns insider trading, major shareholders (DDIND, DDRP, DDINS)\n                - 'agm': Returns shareholder meetings (EGME, AGME, AGMR)\n                - 'others': Returns other fluctuations (MOVE, MA, NLIS, AIS, RETU, OTHE, SUSP)\n                - Or a specific eventCode (e.g., 'ISS,DIV')\n            page (int): Page index. Defaults to 0.\n            limit (int): Number of records per page. Defaults to 20000.\n            show_log (bool): Show logs.\n            \n        Returns:\n            pd.DataFrame: List of events.\n        ";L='content';I=show_log;H='data';F=self;D=event_type;C=end;B=start
		if B is _A:B=datetime.now().strftime('%Y-%m-%d')
		if C is _A:C=B
		M=B.replace('-','').replace('/','');N=C.replace('-','').replace('/','');J=f"{_VCI_EVENTS_URL}?fromDate={M}&toDate={N}&page={page}&size={limit}";O={'dividend':'ISS,DIV','insider':'DDIND,DDRP,DDINS','agm':'EGME,AGME,AGMR','others':'MOVE,MA,NLIS,AIS,RETU,OTHE,SUSP'}
		if D:P=O.get(D.lower(),D);J+=f"&eventCode={P.replace(',','%2C')}"
		K=F.headers.copy();K.update({'Accept':'application/json','Accept-Language':'en-US,en;q=0.9,vi;q=0.8','Connection':'keep-alive','DNT':'1','Origin':'https://trading.vietcap.com.vn','Referer':'https://trading.vietcap.com.vn/','Sec-Fetch-Dest':'empty','Sec-Fetch-Mode':'cors','Sec-Fetch-Site':'same-site','sec-ch-ua':'"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"','sec-ch-ua-mobile':'?0','sec-ch-ua-platform':'"macOS"'})
		if I or F.show_log:logger.info(f"Fetching events calendar from {B} to {C} (type: {D})")
		E=send_request(url=J,headers=K,method='GET',show_log=I or F.show_log)
		if not E or H not in E or L not in E[H]:return pd.DataFrame()
		A=pd.DataFrame(E[H][L])
		if A.empty:return A
		A.columns=[camel_to_snake(A)for A in A.columns];Q=['display_date1','display_date2','public_date','issue_date','record_date','exright_date','payout_date']
		for G in Q:
			if G in A.columns:A[G]=pd.to_datetime(A[G],format='ISO8601',errors='coerce')
		return A
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('event','vci',Event)