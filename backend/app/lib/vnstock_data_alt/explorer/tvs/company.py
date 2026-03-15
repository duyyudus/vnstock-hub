'\nModule quản lý thông tin công ty từ nguồn dữ liệu VCI.\n'
import json,pandas as pd
from typing import Dict,Optional,Union,List
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
from app.lib.vnstock_data_alt.core.utils.client import send_request
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
from app.lib._vnstock_shared.core.utils.transform import clean_html_dict,flatten_dict_to_df,flatten_list_to_df,reorder_cols,drop_cols_by_pattern
from app.lib._vnstock_shared.core.utils.parser import get_asset_type,camel_to_snake
from app.lib._vnstock_shared.compat import agg_execution
from app.lib.vnstock_data_alt.explorer.tvs.const import _BASE_URL
import copy
logger=get_logger(__name__)
class Company:
	'\n    Class (lớp) quản lý các thông tin liên quan đến công ty từ nguồn dữ liệu VCI.\n\n    Tham số:\n        - symbol (str): Mã chứng khoán của công ty cần truy xuất thông tin.\n        - random_agent (bool): Sử dụng user-agent ngẫu nhiên hoặc không. Mặc định là False.\n        - to_df (bool): Chuyển đổi dữ liệu thành DataFrame hoặc không. Mặc định là True.\n        - show_log (bool): Hiển thị thông tin log hoặc không. Mặc định là False.\n    '
	def __init__(A,symbol,random_agent=False,to_df=True,show_log=False):
		'\n        Khởi tạo đối tượng Company với các tham số cho việc truy xuất dữ liệu.\n        ';B=show_log;A.symbol=symbol.upper();A.asset_type=get_asset_type(A.symbol)
		if A.asset_type not in['stock']:raise ValueError('Mã chứng khoán không hợp lệ. Chỉ cổ phiếu mới có thông tin.')
		A.headers=get_headers(data_source='TVS',random_agent=random_agent);A.show_log=B;A.to_df=to_df
		if not B:logger.setLevel('CRITICAL')
	def _fetch_data(A,url):
		'\n        Phương thức riêng để lấy dữ liệu công ty từ nguồn VCI.\n        \n        Returns:\n            Dict: Dữ liệu thô về công ty từ API.\n        '
		if A.show_log:logger.debug(f"Requesting data for {A.symbol} from {url}. payload: {payload}")
		B=send_request(url=url,headers=A.headers,method='POST',payload=payload,show_log=A.show_log);return B
	@agg_execution('TVS.ext')
	def overview(self):
		'\n        Lấy thông tin tổng quan về công ty từ nguồn dữ liệu VCI.\n        \n        Returns:\n            Union[Dict, pd.DataFrame]: Thông tin tổng quan về công ty dưới dạng từ điển hoặc DataFrame.\n        ';B=self;D=f"{_BASE_URL}Dashboard/GetComanyInfo?ticker={B.symbol}";C=B._fetch_data(D)
		if not C:logger.warning(f"No data available for {B.symbol}");return
		A=pd.DataFrame(C,index=[0]);A.columns=[camel_to_snake(A)for A in A.columns];A=A.rename(columns={'ticker':'symbol'})
		if B.to_df:return A
		else:return C.to_dict(orient='records')[0]if not C.empty else{}