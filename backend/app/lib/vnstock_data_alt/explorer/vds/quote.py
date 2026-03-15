_A=None
from typing import List,Dict,Optional
from datetime import datetime,timedelta
import json,requests,pandas as pd
from app.lib._vnstock_shared.compat import agg_execution
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib._vnstock_shared.core.utils.parser import get_asset_type,camel_to_snake,flatten_data
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
from app.lib.vnstock_data_alt.core.utils.validation import validate_date
from app.lib.vnstock_data_alt.core.utils.browser import get_cookie
from app.lib.vnstock_data_alt.explorer.vds.const import _BASE_URL,_ORDER_MATCH_MAPPING
logger=get_logger(__name__)
class Quote:
	'\n    Truy xuất dữ liệu giao dịch của mã chứng khoán từ nguồn dữ liệu VDSC.\n    '
	def __init__(A,symbol,cookie=_A,random_agent=False,show_log=False):
		D='Cookie';C=show_log;B=cookie;A.symbol=symbol.upper();A.asset_type=get_asset_type(A.symbol);A.headers=get_headers(data_source='VDS',random_agent=random_agent);A.headers['Content-Type']='application/x-www-form-urlencoded; charset=UTF-8'
		if B is _A:B=get_cookie('https://livedragon.vdsc.com.vn/general/intradayBoard.rv',headers=A.headers);A.headers[D]=B
		else:A.headers[D]=B
		if not C:logger.setLevel('CRITICAL')
		A.show_log=C
	@agg_execution('VDS.ext')
	def intraday(self,date=_A):
		"\n        Get the matched transactions data for a specific stock symbol and date from Live Dragon website.\n\n        Parameters:\n            date (str): The date for which to retrieve the matched transactions data, in the format 'YYYY-MM-DD'. If None, today's date will be used.\n            cookie (str): The cookie value to include in the request headers. Default is None. If the automatic method fails, you may need to copy the actual cookie value from your web browser.\n        ";A=self
		if date is _A:D=datetime.now()
		else:D=datetime.strptime(date,'%Y-%m-%d')
		H=D.strftime('%d/%m/%Y');E=f"{_BASE_URL}general/intradaySearch.rv";F=f"stockCode={A.symbol}&boardDate={H}"
		if A.show_log:logger.info(f"Request data to {E}, using payload as details: {F}. Headers values: {A.headers}")
		C=requests.request('POST',E,headers=A.headers,data=F)
		if C.status_code!=200:logger.debug(f"Error: {C.text}")
		G=C.json()
		if A.show_log:logger.info(G)
		B=pd.DataFrame(G['list']);B.columns=[camel_to_snake(A)for A in B.columns];B.rename(columns=_ORDER_MATCH_MAPPING,inplace=True);return B