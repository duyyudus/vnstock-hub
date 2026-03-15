'Company module for KB Securities (KBS) data source.'
_K='công ty liên kết'
_J='ownership_percent'
_I='date'
_H='coerce'
_G='GET'
_F='LaborStructure'
_E=None
_D='KBS.ext'
_C='source'
_B='symbol'
_A=False
import json,pandas as pd
from typing import Dict,Optional,List
import re
from app.lib._vnstock_shared.compat import agg_execution
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib._vnstock_shared.core.utils.parser import get_asset_type,camel_to_snake
from app.lib._vnstock_shared.core.utils.transform import clean_html_dict
from app.lib._vnstock_shared.core.utils.client import send_request,ProxyConfig
from app.lib._vnstock_shared.core.utils.user_agent import get_headers
from app.lib.vnstock_data_alt.explorer.kbs.const import _STOCK_INFO_URL,_IIS_BASE_URL,_SAS_NEWS_URL,_PROFILE_MAP,_EVENT_TYPE,_COMPANY_PROFILE_MAP,_SUBSIDIARIES_MAP,_LEADERS_MAP,_OWNERSHIP_MAP,_SHAREHOLDERS_MAP,_CHARTER_CAPITAL_MAP,_LABOR_STRUCTURE_MAP,_EXCHANGE_CODE_MAP
logger=get_logger(__name__)
class Company:
	'\n    Lớp truy cập thông tin công ty từ KB Securities (KBS).\n    \n    Tính năng:\n    - Fetch dữ liệu công ty từ API (một lần)\n    - Cache dữ liệu để tránh gọi lại\n    - Xử lý và trả về từng nhóm dữ liệu theo method được gọi\n    - Tương tự cấu trúc của VCI Company\n    '
	def __init__(A,symbol,random_agent=_A,proxy_config=_E,show_log=_A,proxy_mode=_E,proxy_list=_E):
		"\n        Khởi tạo Company client cho KBS.\n\n        Args:\n            symbol: Mã chứng khoán (VD: 'ACB', 'VNM').\n            random_agent: Sử dụng user agent ngẫu nhiên. Mặc định False.\n            proxy_config: Cấu hình proxy. Mặc định None.\n            show_log: Hiển thị log debug. Mặc định False.\n            proxy_mode: Chế độ proxy (try, rotate, random, single). Mặc định None.\n            proxy_list: Danh sách proxy URLs. Mặc định None.\n\n        Raises:\n            ValueError: Nếu mã không phải là cổ phiếu.\n        ";E=proxy_mode;D=show_log;C=proxy_config;B=proxy_list;A.symbol=symbol.upper();A.asset_type=get_asset_type(A.symbol)
		if A.asset_type not in['stock']:raise ValueError('Mã CK không hợp lệ hoặc không phải cổ phiếu.')
		A.data_source='KBS';A.headers=get_headers(data_source=A.data_source,random_agent=random_agent);A.show_log=D
		if C is _E:
			G=E if E else'try';F='direct'
			if B and len(B)>0:F='proxy'
			A.proxy_config=ProxyConfig(proxy_mode=G,proxy_list=B,request_mode=F)
		else:A.proxy_config=C
		if not D:logger.setLevel('CRITICAL')
		A._raw_data=_E;A._cache_loaded=_A
	def _load_cache(A,show_log=_A):
		'\n        Fetch và cache dữ liệu công ty từ API (một lần).\n        \n        Returns:\n            Dictionary chứa tất cả dữ liệu công ty.\n        '
		if A._cache_loaded and A._raw_data is not _E:return A._raw_data
		C=f"{_STOCK_INFO_URL}/profile/{A.symbol}";D={'l':1};B=send_request(url=C,headers=A.headers,method=_G,params=D,show_log=show_log or A.show_log,proxy_list=A.proxy_config.proxy_list,proxy_mode=A.proxy_config.proxy_mode,request_mode=A.proxy_config.request_mode,auto_fetch=A.proxy_config.auto_fetch,validate_proxies=A.proxy_config.validate_proxies,prefer_speed=A.proxy_config.prefer_speed);A._raw_data=B;A._cache_loaded=True;return B
	def _fetch_profile(A,show_log=_A):'\n        Lấy thông tin profile công ty từ cache hoặc API.\n\n        Args:\n            show_log: Hiển thị log debug.\n\n        Returns:\n            Dictionary chứa thông tin profile công ty.\n        ';return A._load_cache(show_log=show_log)
	def _process_profile_data(F,raw_data):
		'\n        Xử lý dữ liệu profile thô từ API.\n        \n        Args:\n            raw_data: Dữ liệu thô từ API\n            \n        Returns:\n            DataFrame chứa thông tin profile chuẩn hoá\n        ';I='Value';E='exchange';A=raw_data
		if not A:return pd.DataFrame()
		C={}
		for(G,J)in _COMPANY_PROFILE_MAP.items():
			if G in A:C[J]=A[G]
		C=clean_html_dict(C)
		if _F in A and A[_F]:
			D=A[_F]
			if isinstance(D,list)and len(D)>0:
				H=sum(int(A.get(I,0))for A in D if isinstance(A.get(I),(int,str)))
				if H>0:C['number_of_employees']=H
		B=pd.DataFrame([C])
		if E in B.columns:B[E]=B[E].map(lambda x:_EXCHANGE_CODE_MAP.get(x,x)if pd.notna(x)else x)
		B.attrs[_B]=F.symbol;B.attrs[_C]=F.data_source;return B
	def _process_subsidiaries(E,raw_data):
		'\n        Xử lý dữ liệu công ty con từ API.\n        \n        Args:\n            raw_data: Dữ liệu thô từ API\n            \n        Returns:\n            DataFrame chứa thông tin công ty con\n        ';D='Subsidiaries';B=raw_data
		if D not in B or not B[D]:return pd.DataFrame()
		A=pd.DataFrame(B[D]);A=A.rename(columns=_SUBSIDIARIES_MAP)
		for C in[_I]:
			if C in A.columns:A[C]=pd.to_datetime(A[C],errors=_H)
		A.attrs[_B]=E.symbol;A.attrs[_C]=E.data_source;return A
	def _process_leaders(D,raw_data):
		'\n        Xử lý dữ liệu ban lãnh đạo từ API.\n        \n        Args:\n            raw_data: Dữ liệu thô từ API\n            \n        Returns:\n            DataFrame chứa thông tin ban lãnh đạo\n        ';C='Leaders';B=raw_data
		if C not in B or not B[C]:return pd.DataFrame()
		A=pd.DataFrame(B[C]);A=A.rename(columns=_LEADERS_MAP);A.attrs[_B]=D.symbol;A.attrs[_C]=D.data_source;return A
	def _process_ownership(E,raw_data):
		'\n        Xử lý dữ liệu cơ cấu cổ đông từ API.\n        \n        Args:\n            raw_data: Dữ liệu thô từ API\n            \n        Returns:\n            DataFrame chứa thông tin cơ cấu cổ đông\n        ';D='Ownership';B=raw_data
		if D not in B or not B[D]:return pd.DataFrame()
		A=pd.DataFrame(B[D]);A=A.rename(columns=_OWNERSHIP_MAP)
		for C in[_I]:
			if C in A.columns:A[C]=pd.to_datetime(A[C],errors=_H)
		A.attrs[_B]=E.symbol;A.attrs[_C]=E.data_source;return A
	def _process_shareholders(E,raw_data):
		'\n        Xử lý dữ liệu cổ đông lớn từ API.\n        \n        Args:\n            raw_data: Dữ liệu thô từ API\n            \n        Returns:\n            DataFrame chứa thông tin cổ đông lớn\n        ';D='Shareholders';B=raw_data
		if D not in B or not B[D]:return pd.DataFrame()
		A=pd.DataFrame(B[D]);A=A.rename(columns=_SHAREHOLDERS_MAP)
		for C in[_I]:
			if C in A.columns:A[C]=pd.to_datetime(A[C],errors=_H)
		A.attrs[_B]=E.symbol;A.attrs[_C]=E.data_source;return A
	def _process_charter_capital(E,raw_data):
		'\n        Xử lý dữ liệu lịch sử vốn điều lệ từ API.\n        \n        Args:\n            raw_data: Dữ liệu thô từ API\n            \n        Returns:\n            DataFrame chứa lịch sử vốn điều lệ\n        ';D='CharterCapital';B=raw_data
		if D not in B or not B[D]:return pd.DataFrame()
		A=pd.DataFrame(B[D]);A=A.rename(columns=_CHARTER_CAPITAL_MAP)
		for C in[_I]:
			if C in A.columns:A[C]=pd.to_datetime(A[C],errors=_H)
		A.attrs[_B]=E.symbol;A.attrs[_C]=E.data_source;return A
	def _process_labor_structure(C,raw_data):
		'\n        Xử lý dữ liệu cơ cấu lao động từ API.\n        \n        Args:\n            raw_data: Dữ liệu thô từ API\n            \n        Returns:\n            DataFrame chứa cơ cấu lao động\n        ';B=raw_data
		if _F not in B or not B[_F]:return pd.DataFrame()
		A=pd.DataFrame(B[_F]);A=A.rename(columns=_LABOR_STRUCTURE_MAP);A.attrs[_B]=C.symbol;A.attrs[_C]=C.data_source;return A
	@agg_execution(_D)
	def overview(self,show_log=_A):
		"\n        Truy xuất thông tin tổng quan của công ty.\n\n        Args:\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa thông tin tổng quan công ty.\n\n        Examples:\n            >>> company = Company('ACB')\n            >>> df = company.overview()\n            >>> print(df.columns.tolist()[:5])\n            ['business_model', 'symbol', 'founded_date', 'charter_capital', 'num_employees']\n        ";B=show_log;A=self;C=A._fetch_profile(show_log=B)
		if not C:raise ValueError(f"Không tìm thấy dữ liệu profile cho mã {A.symbol}.")
		D=A._process_profile_data(C)
		if B or A.show_log:logger.info(f"Truy xuất thành công thông tin tổng quan cho {A.symbol}.")
		return D
	@agg_execution(_D)
	def officers(self,show_log=_A):
		"\n        Truy xuất thông tin lãnh đạo công ty (officers).\n\n        Args:\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa thông tin lãnh đạo.\n\n        Examples:\n            >>> company = Company('ACB')\n            >>> df = company.officers()\n            >>> print(df.columns.tolist())\n            ['from_date', 'position_name_vn', 'name', 'position_en', 'position_id']\n        ";B=show_log;A=self;C=A._fetch_profile(show_log=B)
		if not C:return pd.DataFrame()
		D=A._process_leaders(C)
		if B or A.show_log:logger.info(f"Truy xuất thành công {len(D)} lãnh đạo công ty cho {A.symbol}.")
		return D
	@agg_execution(_D)
	def shareholders(self,show_log=_A):
		"\n        Truy xuất thông tin cổ đông của công ty.\n\n        Args:\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa thông tin cổ đông.\n\n        Examples:\n            >>> company = Company('ACB')\n            >>> df = company.shareholders()\n            >>> print(df.columns.tolist())\n            ['name', 'date', 'shares', 'ownership_ratio']\n        ";B=show_log;A=self;C=A._fetch_profile(show_log=B)
		if not C:return pd.DataFrame()
		D=A._process_shareholders(C)
		if B or A.show_log:logger.info(f"Truy xuất thành công {len(D)} cổ đông cho {A.symbol}.")
		return D
	@agg_execution(_D)
	def ownership(self,show_log=_A):
		"\n        Truy xuất cơ cấu cổ đông của công ty.\n\n        Args:\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa cơ cấu cổ đông.\n\n        Examples:\n            >>> company = Company('ACB')\n            >>> df = company.ownership()\n            >>> print(df.columns.tolist())\n            ['owner_type', 'ownership_ratio', 'shares', 'date']\n        ";B=show_log;A=self;C=A._fetch_profile(show_log=B)
		if not C:return pd.DataFrame()
		D=A._process_ownership(C)
		if B or A.show_log:logger.info(f"Truy xuất thành công cơ cấu cổ đông cho {A.symbol}.")
		return D
	@agg_execution(_D)
	def subsidiaries(self,show_log=_A):
		"\n        Truy xuất thông tin công ty con và công ty liên kết của công ty.\n        \n        Bao gồm cả công ty con (ownership > 50%) và công ty liên kết (ownership ≤ 50%),\n        với cột 'type' để phân biệt.\n\n        Args:\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa thông tin công ty con và công ty liên kết.\n\n        Examples:\n            >>> company = Company('ACB')\n            >>> df = company.subsidiaries()\n            >>> print(df.columns.tolist())\n            ['date', 'name', 'charter_capital', 'ownership_ratio', 'currency', 'type']\n        ";C=show_log;B=self;D=B._fetch_profile(show_log=C)
		if not D:return pd.DataFrame()
		A=B._process_subsidiaries(D)
		if len(A)>0:A['type']=A[_J].apply(lambda x:'công ty con'if x>50 else _K)
		if C or B.show_log:logger.info(f"Truy xuất thành công {len(A)} công ty con/liên kết cho {B.symbol}.")
		return A
	@agg_execution(_D)
	def affiliate(self,show_log=_A):
		'\n        Truy xuất thông tin công ty liên kết của công ty (ownership ≤ 50%).\n        \n        Công ty liên kết được định nghĩa là các công ty có tỷ lệ sở hữu tối đa 50%.\n        Dữ liệu được lọc từ danh sách công ty con.\n\n        Args:\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa thông tin công ty liên kết.\n        ';D=show_log;A=self;E=A._fetch_profile(show_log=D)
		if not E:return pd.DataFrame()
		B=A._process_subsidiaries(E)
		if len(B)==0:return B
		C=B[B[_J]<=50].copy();C['type']=_K
		if D or A.show_log:logger.info(f"Truy xuất thành công {len(C)} công ty liên kết cho {A.symbol}.")
		return C
	@agg_execution(_D)
	def capital_history(self,show_log=_A):
		"\n        Truy xuất lịch sử vốn điều lệ của công ty.\n\n        Args:\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa lịch sử vốn điều lệ.\n\n        Examples:\n            >>> company = Company('ACB')\n            >>> df = company.capital_history()\n            >>> print(df.columns.tolist())\n            ['date', 'value', 'currency']\n        ";B=show_log;A=self;C=A._fetch_profile(show_log=B)
		if not C:return pd.DataFrame()
		D=A._process_charter_capital(C)
		if B or A.show_log:logger.info(f"Truy xuất thành công lịch sử vốn điều lệ cho {A.symbol}.")
		return D
	@agg_execution(_D)
	def events(self,event_type=_E,page=1,page_size=10,show_log=_A):
		"\n        Truy xuất danh sách sự kiện của công ty.\n\n        Args:\n            event_type: Loại sự kiện (1-5). None để lấy tất cả. \n                        1: Đại hội cổ đông, 2: Trả cổ tức, 3: Phát hành,\n                        4: Giao dịch cổ đông nội bộ, 5: Sự kiện khác.\n            page: Số trang. Mặc định 1.\n            page_size: Số lượng bản ghi mỗi trang. Mặc định 10.\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa danh sách sự kiện.\n\n        Examples:\n            >>> company = Company('ACB')\n            >>> df = company.events(event_type=2)  # Sự kiện trả cổ tức\n        ";D=show_log;C=event_type;A=self;G=f"{_STOCK_INFO_URL}/event/{A.symbol}";E={'l':1,'p':page,'s':page_size}
		if C is not _E:
			if C not in _EVENT_TYPE:raise ValueError(f"event_type không hợp lệ. Các giá trị hợp lệ: {list(_EVENT_TYPE.keys())}")
			E['eID']=C
		F=send_request(url=G,headers=A.headers,method=_G,params=E,show_log=D or A.show_log,proxy_list=A.proxy_config.proxy_list,proxy_mode=A.proxy_config.proxy_mode,request_mode=A.proxy_config.request_mode,auto_fetch=A.proxy_config.auto_fetch,validate_proxies=A.proxy_config.validate_proxies,prefer_speed=A.proxy_config.prefer_speed)
		if not F:return pd.DataFrame()
		B=pd.DataFrame(F);B.columns=[camel_to_snake(A)for A in B.columns];B.attrs[_B]=A.symbol;B.attrs[_C]=A.data_source
		if C:B.attrs['event_type']=_EVENT_TYPE[C]
		if D or A.show_log:logger.info(f"Truy xuất thành công {len(B)} sự kiện cho {A.symbol}.")
		return B
	@agg_execution(_D)
	def news(self,page=1,page_size=10,show_log=_A):
		"\n        Truy xuất tin tức liên quan đến công ty.\n\n        Args:\n            page: Số trang. Mặc định 1.\n            page_size: Số lượng bản ghi mỗi trang. Mặc định 10.\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa danh sách tin tức.\n\n        Examples:\n            >>> company = Company('ACB')\n            >>> df = company.news(page=1, page_size=20)\n        ";C=show_log;A=self;E=f"{_STOCK_INFO_URL}/news/{A.symbol}";F={'l':1,'p':page,'s':page_size};D=send_request(url=E,headers=A.headers,method=_G,params=F,show_log=C or A.show_log,proxy_list=A.proxy_config.proxy_list,proxy_mode=A.proxy_config.proxy_mode,request_mode=A.proxy_config.request_mode,auto_fetch=A.proxy_config.auto_fetch,validate_proxies=A.proxy_config.validate_proxies,prefer_speed=A.proxy_config.prefer_speed)
		if not D:return pd.DataFrame()
		B=pd.DataFrame(D);B.columns=[camel_to_snake(A)for A in B.columns];B.attrs[_B]=A.symbol;B.attrs[_C]=A.data_source
		if C or A.show_log:logger.info(f"Truy xuất thành công {len(B)} tin tức cho {A.symbol}.")
		return B
	@agg_execution(_D)
	def insider_trading(self,page=1,page_size=10,show_log=_A):
		"\n        Truy xuất thông tin giao dịch nội bộ.\n\n        Args:\n            page: Số trang. Mặc định 1.\n            page_size: Số lượng bản ghi mỗi trang. Mặc định 10.\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa thông tin giao dịch nội bộ.\n\n        Examples:\n            >>> company = Company('ACB')\n            >>> df = company.insider_trading()\n        ";C=show_log;A=self;E=f"{_STOCK_INFO_URL}/news/internal-trading/{A.symbol}";F={'l':1,'p':page,'s':page_size};D=send_request(url=E,headers=A.headers,method=_G,params=F,show_log=C or A.show_log,proxy_list=A.proxy_config.proxy_list,proxy_mode=A.proxy_config.proxy_mode,request_mode=A.proxy_config.request_mode,auto_fetch=A.proxy_config.auto_fetch,validate_proxies=A.proxy_config.validate_proxies,prefer_speed=A.proxy_config.prefer_speed)
		if not D:return pd.DataFrame()
		B=pd.DataFrame(D);B.columns=[camel_to_snake(A)for A in B.columns];B.attrs[_B]=A.symbol;B.attrs[_C]=A.data_source
		if C or A.show_log:logger.info(f"Truy xuất thành công {len(B)} bản ghi giao dịch nội bộ cho {A.symbol}.")
		return B
	@agg_execution(_D)
	def margin_ratio(self,show_log=_A):
		"\n        Truy xuất thông tin tỷ lệ cho vay ký quỹ (margin) của mã chứng khoán tại các CTCK.\n\n        Args:\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa thông tin tỷ lệ margin.\n\n        Examples:\n            >>> company = Company('ACB')\n            >>> df = company.margin_ratio()\n        ";D=show_log;C='ClosedDate';A=self;F='https://kbbuddywts.kbsec.com.vn/sas/kbsv-stock-data-store/stock/trading-margin';G={'code':A.symbol,'languageID':1};E=send_request(url=F,headers=A.headers,method=_G,params=G,show_log=D or A.show_log,proxy_list=A.proxy_config.proxy_list,proxy_mode=A.proxy_config.proxy_mode,request_mode=A.proxy_config.request_mode)
		if not E:return pd.DataFrame()
		B=pd.DataFrame(E)
		if C in B.columns:B[C]=pd.to_datetime(B[C],errors=_H).dt.normalize()
		B.attrs[_B]=A.symbol;B.attrs[_C]=A.data_source
		if D or A.show_log:logger.info(f"Truy xuất thành công tỷ lệ margin cho {A.symbol} tại {len(B)} CTCK.")
		return B
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('company','kbs',Company)