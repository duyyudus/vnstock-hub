'Trading module for KB Securities (KBS) data source.'
_e='symbols'
_d='stock'
_c='GET'
_b='vi'
_a='same-origin'
_Z='cors'
_Y='empty'
_X='application/json'
_W='keep-alive'
_V='en-US,en;q=0.9,vi;q=0.8'
_U='x-lang'
_T='Sec-Fetch-Site'
_S='Sec-Fetch-Mode'
_R='Sec-Fetch-Dest'
_Q='DNT'
_P='Content-Type'
_O='Connection'
_N='Accept-Language'
_M='ms'
_L='page'
_K='code'
_J='exchange'
_I='HOSE'
_H='source'
_G='data'
_F='KBS.ext'
_E='symbol'
_D='coerce'
_C=None
_B=False
_A='timestamp'
import pandas as pd,json,re
from datetime import datetime
from typing import Optional,List
from app.lib._vnstock_shared.compat import agg_execution
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib._vnstock_shared.core.utils.parser import get_asset_type,camel_to_snake
from app.lib._vnstock_shared.core.utils.client import send_request,ProxyConfig
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
from app.lib.vnstock_data_alt.explorer.kbs.const import _IIS_BASE_URL,_STOCK_TRADE_HISTORY_URL,_PUT_THROUGH_HISTORY_URL,_STOCK_MATCHED_BY_PRICE_URL,_STOCK_ISS_URL,_ODD_LOT_ISS_URL,_PUT_THROUGH_ISS_URL,_DERIVATIVE_ISS_URL,_PRICE_BOARD_MAP,_ODD_LOT_MAP,_TRADE_HISTORY_MAP,_PUT_THROUGH_MAP,_MATCHED_BY_PRICE_MAP,_DERIVATIVE_MAP,_INDEX_SUMMARY_URL,_INDEX_SUMMARY_MAP,_INDEX_SYMBOL_TO_CODE,_PRICE_BOARD_STANDARD_COLUMNS,_PUT_THROUGH_STANDARD_COLUMNS,_DERIVATIVE_STANDARD_COLUMNS,_KBS_TO_SCHEMA_MAP,_EXCLUDED_COLUMNS,_EXCHANGE_CODE_MAP
logger=get_logger(__name__)
class Trading:
	'\n    Lớp truy cập dữ liệu giao dịch từ KB Securities (KBS).\n    '
	def __init__(A,symbol=_C,random_agent=_B,proxy_config=_C,show_log=_B,proxy_mode=_C,proxy_list=_C):
		"\n        Khởi tạo Trading client cho KBS.\n\n        Args:\n            symbol: Mã chứng khoán (VD: 'ACB', 'VNM'). Optional cho market-wide queries.\n            random_agent: Sử dụng user agent ngẫu nhiên. Mặc định False.\n            proxy_config: Cấu hình proxy. Mặc định None.\n            show_log: Hiển thị log debug. Mặc định False.\n            proxy_mode: Chế độ proxy (try, rotate, random, single). Mặc định None.\n            proxy_list: Danh sách proxy URLs. Mặc định None.\n        ";F=proxy_mode;E=show_log;D=proxy_config;C=symbol;B=proxy_list;A.symbol=C.upper()if C else _C;A.data_source='KBS';A.base_url=_IIS_BASE_URL;A.headers=get_headers(data_source=A.data_source,random_agent=random_agent);A.show_log=E
		if D is _C:
			H=F if F else'try';G='direct'
			if B and len(B)>0:G='proxy'
			A.proxy_config=ProxyConfig(proxy_mode=H,proxy_list=B,request_mode=G)
		else:A.proxy_config=D
		if A.symbol:A.asset_type=get_asset_type(A.symbol)
		if not E:logger.setLevel('CRITICAL')
	@agg_execution(_F)
	def price_history(self,*A,**B):return self.trade_history(*A,**B)
	@agg_execution(_F)
	def trade_history(self,page=1,page_size=1000,show_log=_B):
		"\n        Truy xuất lịch sử giao dịch của mã chứng khoán.\n\n        Args:\n            page: Số trang. Mặc định 1.\n            page_size: Số lượng bản ghi mỗi trang. Mặc định 1000.\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa lịch sử giao dịch với các cột chuẩn hóa.\n\n        Examples:\n            >>> trading = Trading('ACB')\n            >>> df = trading.trade_history(page=1, page_size=100)\n\n        Raises:\n            ValueError: Nếu không có symbol được chỉ định.\n        ";E=show_log;D=page_size;A=self
		if not A.symbol:raise ValueError('Symbol is required for trade_history method.')
		F=f"{_STOCK_TRADE_HISTORY_URL}/{A.symbol}";G={_L:page,'limit':D};C=send_request(url=F,headers=A.headers,method=_c,params=G,show_log=E or A.show_log,proxy_list=A.proxy_config.proxy_list,proxy_mode=A.proxy_config.proxy_mode,request_mode=A.proxy_config.request_mode)
		if not C:return pd.DataFrame()
		if isinstance(C,dict)and _G in C:C=C[_G]
		if not C or not isinstance(C,list):return pd.DataFrame()
		B=pd.DataFrame(C);B=B.rename(columns=_TRADE_HISTORY_MAP)
		if _A in B.columns:B[_A]=pd.to_datetime(B[_A],unit=_M,errors=_D)
		B.attrs[_E]=A.symbol;B.attrs[_H]=A.data_source;B.attrs[_L]=page;B.attrs['page_size']=D
		if E or A.show_log:logger.info(f"Truy xuất thành công {len(B)} bản ghi lịch sử giao dịch cho {A.symbol}.")
		return B
	@agg_execution(_F)
	def matched_by_price(self,show_log=_B):
		"\n        Truy xuất dữ liệu khớp lệnh theo từng mức giá.\n\n        Args:\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa dữ liệu khớp lệnh theo giá với các cột chuẩn hóa.\n\n        Examples:\n            >>> trading = Trading('ACB')\n            >>> df = trading.matched_by_price()\n\n        Raises:\n            ValueError: Nếu không có symbol được chỉ định.\n        ";C=show_log;A=self
		if not A.symbol:raise ValueError('Symbol is required for matched_by_price method.')
		E=f"{_STOCK_MATCHED_BY_PRICE_URL}/{A.symbol}";D=send_request(url=E,headers=A.headers,method=_c,show_log=C or A.show_log,proxy_list=A.proxy_config.proxy_list,proxy_mode=A.proxy_config.proxy_mode,request_mode=A.proxy_config.request_mode)
		if not D:return pd.DataFrame()
		B=pd.DataFrame(D);B=B.rename(columns=_MATCHED_BY_PRICE_MAP);B.attrs[_E]=A.symbol;B.attrs[_H]=A.data_source
		if C or A.show_log:logger.info(f"Truy xuất thành công dữ liệu khớp lệnh theo giá cho {A.symbol}.")
		return B
	@agg_execution(_F)
	def index_summary(self,show_log=_B):
		'\n        Truy xuất thông tin tóm tắt (Snapshot) cho các chỉ số thị trường.\n\n        Chấp nhận mã symbol (VNINDEX, VN30, ...) và tự động mapping sang mã KBS (HOSE, 30, ...).\n        Nếu không có symbol, mặc định lấy toàn bộ các chỉ số chính.\n\n        Args:\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa thông tin tóm tắt chỉ số với các cột chuẩn hóa.\n        ';F=show_log;B=self;import requests as J;K=_INDEX_SUMMARY_URL
		if B.symbol:
			E=_INDEX_SYMBOL_TO_CODE.get(B.symbol)
			if not E:E=B.symbol
			G={_K:E}
		else:G={_K:'HOSE,30,100,HNX,HNX30,UPCOM'}
		try:
			H=B.headers.copy();H.update({_N:_V,_O:_W,_P:_X,_Q:'1',_R:_Y,_S:_Z,_T:_a,_U:_b});I=J.post(K,headers=H,data=json.dumps(G),timeout=30)
			if I.status_code in[200,201]:D=I.json()
			else:return pd.DataFrame()
		except Exception as L:
			if F or B.show_log:logger.error(f"Failed to fetch index summary data: {str(L)}")
			return pd.DataFrame()
		if not D or not isinstance(D,list):return pd.DataFrame()
		A=pd.DataFrame(D);A=A.rename(columns=_INDEX_SUMMARY_MAP);M={B:A for(A,B)in _INDEX_SYMBOL_TO_CODE.items()}
		if'MC'in D[0]:A[_E]=A['MC'].map(lambda x:M.get(str(x),x))
		N=['close_price','price_change','open_price','high_price','low_price','reference_price','previous_close']
		for C in N:
			if C in A.columns:A[C]=pd.to_numeric(A[C],errors=_D)
		O=['accumulated_volume','accumulated_value','total_volume','put_through_value','put_through_volume','advances','declines','no_change']
		for C in O:
			if C in A.columns:A[C]=pd.to_numeric(A[C],errors=_D)
		if _A in A.columns:A[_A]=pd.to_datetime(A[_A],unit=_M,errors=_D)
		A.attrs[_H]=B.data_source
		if B.symbol:A.attrs[_E]=B.symbol
		if F or B.show_log:logger.info(f"Truy xuất thành công tóm tắt cho {len(A)} chỉ số.")
		return A
	def _fetch_stock_board(C,symbols_list,show_log=_B):
		'\n        Fetch stock board (lô chẵn) data from /stock/iss endpoint.\n        \n        Args:\n            symbols_list: List of stock symbols.\n            show_log: Show debug logs.\n            \n        Returns:\n            DataFrame with stock board data.\n        ';import requests as F;G=_STOCK_ISS_URL;H={_K:','.join(symbols_list)}
		try:
			D=C.headers.copy();D.update({_N:_V,_O:_W,_P:_X,_Q:'1',_R:_Y,_S:_Z,_T:_a,_U:_b});E=F.post(G,headers=D,data=json.dumps(H),timeout=30)
			if E.status_code in[200,201]:B=E.json()
			else:return pd.DataFrame()
		except Exception as I:
			if show_log or C.show_log:logger.error(f"Failed to fetch stock board data: {str(I)}")
			return pd.DataFrame()
		if not B or not isinstance(B,list):return pd.DataFrame()
		A=pd.DataFrame(B);A=A.rename(columns=_PRICE_BOARD_MAP)
		if _A in A.columns:A[_A]=pd.to_datetime(A[_A],unit=_M,errors=_D)
		return A
	def _fetch_put_through_board(B,symbols_list,show_log=_B):
		"\n        Fetch put-through board (thỏa thuận) data.\n        \n        Note: The put-through ISS endpoint doesn't exist. Use put_through() method instead\n        which fetches from /put-through/trade/history endpoint and returns all put-through data.\n        \n        Args:\n            symbols_list: List of stock symbols.\n            show_log: Show debug logs.\n            \n        Returns:\n            DataFrame with put-through board data filtered by symbols.\n        ";A=B.put_through(exchange=_I,page=1,show_log=show_log);A=A.loc[:,~A.columns.duplicated(keep='first')]
		if _E in A.columns and len(A)>0:A=A[A[_E].isin(symbols_list)].reset_index(drop=True)
		return A
	def _fetch_derivative_board(D,symbols_list,show_log=_B):
		'\n        Fetch derivative board data.\n        \n        Args:\n            symbols_list: List of derivative symbols.\n            show_log: Show debug logs.\n            \n        Returns:\n            DataFrame with derivative board data.\n        ';C='time';import requests as G;H=_DERIVATIVE_ISS_URL;I={_K:','.join(symbols_list)}
		try:
			E=D.headers.copy();E.update({_N:_V,_O:_W,_P:_X,_Q:'1',_R:_Y,_S:_Z,_T:_a,_U:_b});F=G.post(H,headers=E,data=json.dumps(I),timeout=30)
			if F.status_code in[200,201]:A=F.json()
			else:return pd.DataFrame()
		except Exception as J:
			if show_log or D.show_log:logger.error(f"Failed to fetch derivative board data: {str(J)}")
			return pd.DataFrame()
		if not A:return pd.DataFrame()
		if isinstance(A,dict)and _G in A:A=A[_G]
		if not isinstance(A,list):return pd.DataFrame()
		B=pd.DataFrame(A);B=B.rename(columns=_DERIVATIVE_MAP)
		if C in B.columns:B[C]=pd.to_datetime(B[C],unit=_M,errors=_D)
		return B
	@agg_execution(_F)
	def price_board(self,symbols_list,board=_d,exchange=_I,show_log=_B,get_all=_B):
		"\n        Fetch real-time price board for a list of symbols.\n\n        Unified interface for fetching price data from various board types:\n        - stock: Standard board (even lots)\n        - odd_lot: Odd-lot trades\n        - put_through: Negotiated/Put-through trades\n        - derivatives: Index futures\n\n        Args:\n            symbols_list (List[str]): List of symbols (e.g., ['ACB', 'VNM']).\n            board (str): Board type ('stock', 'odd_lot', 'put_through', 'derivatives').\n            exchange (str): Exchange ('HOSE', 'HNX', 'UPCOM').\n            show_log (bool): Display debug logs.\n            get_all (bool): If True, return all raw columns. Otherwise, standard columns.\n        ";S='phái sinh';R='derivatives';Q='odd_lot';O=get_all;F=board;D=show_log;C=self;B=symbols_list
		if not B:raise ValueError('symbols_list không được để trống.')
		P=[_d,Q,'put_through',R]
		if F not in P:raise ValueError(f"board không hợp lệ. Các giá trị hợp lệ: {P}")
		B=[A.upper()for A in B];E=[]
		if F==_d:
			from app.lib._vnstock_shared.core.utils.parser import get_asset_type as T;from app.lib.vnstock_data_alt.core.utils.parser import safe_convert_derivative_symbol as J;H=[];G=[]
			for K in B:
				if T(K)=='derivative':G.append(K)
				else:H.append(K)
			if H:U=C._fetch_stock_board(H,D);E.append((U,_PRICE_BOARD_STANDARD_COLUMNS))
			if G:L=[J(A)for A in G];M=C._fetch_derivative_board(L,D);E.append((M,_DERIVATIVE_STANDARD_COLUMNS))
			I='hỗn hợp (cơ sở & phái sinh)'if H and G else S if G else'lô chẵn'
		elif F==Q:V=C.odd_lot(symbols_list=B,exchange=exchange,show_log=D);E.append((V,_PRICE_BOARD_STANDARD_COLUMNS));I='lô lẻ'
		elif F==R:from app.lib.vnstock_data_alt.core.utils.parser import safe_convert_derivative_symbol as J;L=[J(A)for A in B];M=C._fetch_derivative_board(L,D);E.append((M,_DERIVATIVE_STANDARD_COLUMNS));I=S
		else:W=C._fetch_put_through_board(B,D);E.append((W,_PUT_THROUGH_STANDARD_COLUMNS));I='thỏa thuận'
		N=[]
		for(A,X)in E:
			if len(A)>0:
				if not O:Y=[B for B in X if B in A.columns];A=A[Y]
				else:Z=[A for A in A.columns if A not in _EXCLUDED_COLUMNS];A=A[Z]
				N.append(A)
		if N:
			A=pd.concat(N,ignore_index=True)
			if _J in A.columns:A[_J]=A[_J].map(lambda x:_EXCHANGE_CODE_MAP.get(x,x)if pd.notna(x)else x)
		else:A=pd.DataFrame()
		A.attrs[_e]=B;A.attrs['board']=F;A.attrs[_H]=C.data_source;A.attrs['get_all']=O
		if D or C.show_log:logger.info(f"Truy xuất thành công bảng giá {I} cho {len(B)} mã chứng khoán.")
		return A
	@agg_execution(_F)
	def odd_lot(self,symbols_list=_C,exchange=_I,show_log=_B):
		"\n        Truy xuất dữ liệu giao dịch lô lẻ (odd-lot) cho danh sách mã chứng khoán.\n\n        Note: Phương thức này là alias cho price_board() với data_type='odd_lot'.\n        Khuyến nghị sử dụng price_board(symbols_list, data_type='odd_lot') thay vì phương thức này.\n\n        Args:\n            symbols_list: Danh sách mã chứng khoán. Nếu None, truy xuất toàn bộ sàn.\n            exchange: Sàn giao dịch ('HOSE', 'HNX', 'UPCOM'). Mặc định 'HOSE'.\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa dữ liệu giao dịch lô lẻ với các cột chuẩn hóa.\n\n        Examples:\n            >>> trading = Trading()\n            >>> df = trading.odd_lot(symbols_list=['AAA', 'AAM'])\n            >>> df_all = trading.odd_lot(exchange='HOSE')\n\n        Raises:\n            ValueError: Nếu exchange không hợp lệ.\n        ";F=show_log;D=exchange;C=self;B=symbols_list;G=[_I,'HNX','UPCOM']
		if D not in G:raise ValueError(f"Exchange không hợp lệ. Các giá trị hợp lệ: {G}")
		K=f"{_IIS_BASE_URL}/odd-lot/iss"
		if B:B=[A.upper()for A in B];H={_K:','.join(B)}
		else:H={_J:D}
		import requests as L
		try:
			I=C.headers.copy();I.update({_N:_V,_O:_W,_P:_X,_Q:'1',_R:_Y,_S:_Z,_T:_a,_U:_b});J=L.post(K,headers=I,data=json.dumps(H),timeout=30)
			if J.status_code in[200,201]:E=J.json()
			else:return pd.DataFrame()
		except Exception as M:
			if F or C.show_log:logger.error(f"Failed to fetch odd_lot data: {str(M)}")
			return pd.DataFrame()
		if not E:return pd.DataFrame()
		if not isinstance(E,list):return pd.DataFrame()
		A=pd.DataFrame(E);A=A.rename(columns=_ODD_LOT_MAP)
		if _A in A.columns:A[_A]=pd.to_datetime(A[_A],errors=_D)
		if B:A.attrs[_e]=B
		A.attrs[_J]=D;A.attrs[_H]=C.data_source
		if F or C.show_log:
			if B:logger.info(f"Truy xuất thành công {len(A)} bản ghi giao dịch lô lẻ cho {len(B)} mã chứng khoán.")
			else:logger.info(f"Truy xuất thành công {len(A)} bản ghi giao dịch lô lẻ cho sàn {D}.")
		return A
	@agg_execution(_F)
	def put_through(self,exchange=_I,symbol=_C,page=1,page_size=1000,show_log=_B):
		"\n        Truy xuất dữ liệu giao dịch thỏa thuận (put-through) theo sàn.\n\n        Args:\n            exchange: Sàn giao dịch ('HOSE', 'HNX', 'UPCOM'). Mặc định 'HOSE'.\n            symbol: Mã chứng khoán để lọc (VD: 'ACB'). Nếu None, lấy toàn bộ sàn.\n            page: Số trang. Mặc định 1.\n            page_size: Số lượng bản ghi mỗi trang. Mặc định 1000.\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa dữ liệu giao dịch thỏa thuận với các cột chuẩn hóa.\n\n        Examples:\n            >>> trading = Trading()\n            >>> df = trading.put_through(exchange='HOSE', page=1)\n\n        Raises:\n            ValueError: Nếu exchange không hợp lệ.\n        ";E=show_log;D=exchange;B=self;F=[_I,'HNX','UPCOM']
		if D not in F:raise ValueError(f"Exchange không hợp lệ. Các giá trị hợp lệ: {F}")
		H=f"{_PUT_THROUGH_HISTORY_URL}/{D}";I={_L:page,'pageSize':page_size};C=send_request(url=H,headers=B.headers,method=_c,params=I,show_log=E or B.show_log,proxy_list=B.proxy_config.proxy_list,proxy_mode=B.proxy_config.proxy_mode,request_mode=B.proxy_config.request_mode)
		if not C:return pd.DataFrame()
		if isinstance(C,dict)and _G in C:C=C[_G]
		if not C or not isinstance(C,list):return pd.DataFrame()
		A=pd.DataFrame(C);A=A.rename(columns=_PUT_THROUGH_MAP);A=A.loc[:,~A.columns.duplicated()]
		if _A in A.columns:A[_A]=pd.to_datetime(A[_A],errors=_D)
		G=symbol or B.symbol
		if G and _E in A.columns:A=A[A[_E]==G.upper()].reset_index(drop=True)
		A.attrs[_J]=D;A.attrs[_H]=B.data_source;A.attrs[_L]=page
		if E or B.show_log:logger.info(f"Truy xuất thành công {len(A)} bản ghi giao dịch thỏa thuận cho sàn {D}.")
		return A
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('trading','kbs',Trading)