'History module for MAS.'
_N='volume'
_M='Vui lòng nhập mã chứng khoán cần truy xuất khi khởi tạo Trading Class.'
_L=': Dữ liệu khớp lệnh không thể truy cập trong thời gian chuẩn bị phiên mới. Vui lòng quay lại sau.'
_K='preparing'
_J='data_status'
_I='is_trading_hour'
_H='time'
_G='MAS.ext'
_F='records'
_E='GET'
_D='symbol'
_C=True
_B=False
_A=None
from typing import Dict,Optional,Union,List,Dict,Optional,Union,List
from datetime import datetime,timedelta
import pandas as pd
from app.lib._vnstock_shared.core.utils.lookback import get_start_date_from_lookback
from app.lib._vnstock_shared.compat import agg_execution
from app.lib.vnstock_data_alt.explorer.mas.const import _CHART_URL,_OHLC_MAP,_INTERVAL_MAP,_OHLC_DTYPE,_RESAMPLE_MAP,_INDEX_MAPPING,_INTRADAY_MAP,_INTRADAY_DTYPE,_PRICE_DEPTH_MAP
from app.lib._vnstock_shared.core.models import TickerModel
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib._vnstock_shared.core.utils.market import trading_hours
from app.lib._vnstock_shared.core.utils.parser import get_asset_type
from app.lib._vnstock_shared.core.utils.interval import normalize_interval
from app.lib.vnstock_data_alt.core.utils.client import send_request,ProxyConfig
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
from app.lib.vnstock_data_alt.core.utils.transform import ohlc_to_df,intraday_to_df
from app.lib.vnstock_data_alt.core.types import SubscriptionTier,DataQuality,UpdateFrequency
logger=get_logger(__name__)
class Quote:
	'\n    Cấu hình truy cập dữ liệu lịch sử giá chứng khoán từ MAS.\n    \n    Hỗ trợ các tính năng proxy nâng cao:\n    - auto_fetch: Tự động lấy proxy từ proxyscrape API\n    - validate_proxies: Kiểm tra tính hợp lệ của proxy\n    - prefer_speed: Ưu tiên proxy có tốc độ tốt nhất\n    '
	def __init__(A,symbol,random_agent=_B,proxy_config=_A,show_log=_C,proxy_mode=_A,proxy_list=_A):
		E=proxy_mode;D=show_log;C=proxy_config;B=proxy_list;A.symbol=symbol.upper();A.data_source='MAS';A._history=_A;A.asset_type=get_asset_type(A.symbol);A.base_url=_CHART_URL;A.headers=get_headers(data_source=A.data_source,random_agent=random_agent);A.interval_map=_INTERVAL_MAP;A.show_log=D
		if C is _A:
			G=E if E else'try';F='direct'
			if B and len(B)>0:F='proxy'
			A.proxy_config=ProxyConfig(proxy_mode=G,proxy_list=B,request_mode=F)
		else:A.proxy_config=C
		if not D:logger.setLevel('CRITICAL')
		if'INDEX'in A.symbol:A.symbol=A._index_validation()
	def _index_validation(A):
		"\n        If symbol contains 'INDEX' substring, validate it with _INDEX_MAPPING.\n        "
		if A.symbol not in _INDEX_MAPPING.keys():raise ValueError('Không tìm thấy mã chứng khoán '+A.symbol+'. Các giá trị hợp lệ: '+', '.join(_INDEX_MAPPING.keys()))
		return _INDEX_MAPPING[A.symbol]
	def _input_validation(B,start,end,interval):
		'\n        Validate input data\n        ';C=interval
		try:D=normalize_interval(C);E=D.value
		except ValueError:raise ValueError(f"Giá trị interval không hợp lệ: {C}. Vui lòng chọn: 1m, 5m, 15m, 30m, 1H, 4h, 1D, 1W, 1M")
		A=TickerModel(symbol=B.symbol,start=start,end=end,interval=E)
		if A.interval not in B.interval_map:raise ValueError(f"Giá trị interval không hỗ trợ bởi MAS: {A.interval}. Các interval được hỗ trợ: 1m, 5m, 15m, 30m, 1H, 1D, 1W, 1M")
		return A
	@agg_execution(_G)
	def history(self,start=_A,end=_A,interval='1D',to_df=_C,show_log=_B,count_back=_A,floating=2,length=_A):
		'\n        Tải lịch sử giá của mã chứng khoán từ nguồn dữ liệu MAS.\n\n        Tham số:\n            - start (tùy chọn): thời gian bắt đầu lấy dữ liệu (YYYY-MM-DD). Bắt buộc nếu không có length hoặc count_back.\n            - end (tùy chọn): thời gian kết thúc lấy dữ liệu. Mặc định là None, chương trình tự động lấy thời điểm hiện tại.\n            - interval (tùy chọn): Khung thời gian trích xuất dữ liệu. Mặc định là "1D".\n            - to_df (tùy chọn): Chuyển đổi dữ liệu lịch sử trả về dưới dạng DataFrame. Mặc định là True.\n            - show_log (tùy chọn): Hiển thị thông tin log. Mặc định là False.\n            - count_back (tùy chọn): Số lượng dữ liệu trả về từ thời điểm cuối.\n            - floating (tùy chọn): Số chữ số thập phân cho giá. Mặc định là 2.\n            - length (tùy chọn): Khoảng thời gian phân tích (vd: \'3M\', 150, \'100b\').\n        ';J=length;F=interval;E=count_back;D=start;C=self;B='%Y-%m-%d';A=end
		if A is _A:A=datetime.now().strftime(B)
		if D is _A:
			if J is not _A:
				G=str(J)
				if G.endswith('b'):E=int(G[:-1]);D=get_start_date_from_lookback(lookback_length=G,end_date=A)
				else:D=get_start_date_from_lookback(lookback_length=G,end_date=A)
			elif E is not _A:
				if F=='1D':D=(datetime.strptime(A,B)-timedelta(days=E*2)).strftime(B)
				elif F=='1H':D=(datetime.strptime(A,B)-timedelta(days=E//6)).strftime(B)
				elif F=='1m':D=(datetime.strptime(A,B)-timedelta(days=1)).strftime(B)
				else:D=get_start_date_from_lookback(lookback_length='1M',end_date=A)
			else:raise ValueError("Tham số 'start' là bắt buộc nếu không cung cấp 'length' hoặc 'count_back'.")
		H=C._input_validation(D,A,F);K=datetime.strptime(H.start,B)
		if A is not _A:
			L=datetime.strptime(H.end,B)+pd.Timedelta(days=1)
			if K>L:raise ValueError('Thời gian bắt đầu không thể lớn hơn thời gian kết thúc.')
			M=int(L.timestamp())
		else:M=int((datetime.now()+pd.Timedelta(days=1)).timestamp())
		O=int(K.timestamp());P=C.interval_map[H.interval];Q=C.base_url+'tradingview/history';R={_D:[C.symbol],'resolution':P,'from':O,'to':M};N=send_request(url=Q,headers=C.headers,method=_E,params=R,payload=_A,show_log=show_log)
		if not N:raise ValueError('Không tìm thấy dữ liệu. Vui lòng kiểm tra lại mã chứng khoán hoặc thời gian truy xuất.')
		I=ohlc_to_df(data=N,column_map=_OHLC_MAP,dtype_map=_OHLC_DTYPE,asset_type=C.asset_type,symbol=C.symbol,source=C.data_source,interval=H.interval,floating=floating,resample_map=_RESAMPLE_MAP)
		if E is not _A:I=I.tail(E)
		if to_df:return I
		else:return I.to_json(orient=_F)
	@agg_execution(_G)
	def intraday(self,page_size=100,last_time=_A,to_df=_C,get_all=_B,show_log=_B):
		'\n        Truy xuất dữ liệu khớp lệnh của mã chứng khoán bất kỳ từ nguồn dữ liệu MAS\n\n        Tham số:\n            - page_size (tùy chọn): Số lượng dữ liệu trả về trong một lần request. Mặc định là 100. \n            - last_time (tùy chọn): Thời gian cắt dữ liệu, dùng để lấy dữ liệu sau thời gian cắt. Mặc định là None.\n            - to_df (tùy chọn): Chuyển đổi dữ liệu lịch sử trả về dưới dạng DataFrame. Mặc định là True.\n            - get_all (tùy chọn): Lấy tất cả các cột trả về từ API thay vì chỉ các cột chuẩn hoá. Mặc định là False.\n            - show_log (tùy chọn): Hiển thị thông tin log giúp debug dễ dàng. Mặc định là False.\n        ';D=page_size;A=self
		if A.asset_type=='index':raise ValueError(f"Dữ liệu intraday không được hỗ trợ cho chỉ số {A.symbol}.")
		C=trading_hours(_A)
		if C[_I]is _B and C[_J]==_K:raise ValueError(str(C[_H])+_L)
		if A.symbol is _A:raise ValueError(_M)
		if D>30000:logger.warning('Bạn đang yêu cầu truy xuất quá nhiều dữ liệu, điều này có thể gây lỗi quá tải.')
		E=A.base_url+f"market/{A.symbol}/quote";F={_D:A.symbol,'fetchCount':D};G=send_request(url=E,headers=A.headers,method=_E,params=F,payload=_A,show_log=show_log);B=intraday_to_df(data=G['data'],column_map=_INTRADAY_MAP,dtype_map=_INTRADAY_DTYPE,symbol=A.symbol,asset_type=A.asset_type,source=A.data_source)
		if get_all:return B
		else:H=[_H,'price',_N,'match_type'];B=B[H]
		if to_df:return B
		else:return B.to_json(orient=_F)
	@agg_execution(_G)
	def price_depth(self,get_all=_B,to_df=_C,show_log=_B):
		'\n        Truy xuất thống kê độ bước giá & khối lượng khớp lệnh của mã chứng khoán bất kỳ từ nguồn dữ liệu MAS.\n\n        Tham số:\n            - get_all (tùy chọn): Lấy tất cả các cột trả về từ API thay vì chỉ các cột chuẩn hoá. Mặc định là False.\n            - to_df (tùy chọn): Chuyển đổi dữ liệu lịch sử trả về dưới dạng DataFrame. Mặc định là True.\n            - show_log (tùy chọn): Hiển thị thông tin log giúp debug dễ dàng. Mặc định là False.\n        ';B=self;C=trading_hours(_A)
		if C[_I]is _B and C[_J]==_K:raise ValueError(str(C[_H])+_L)
		if B.symbol is _A:raise ValueError(_M)
		D=B.base_url+'market/quoteSummary';E={_D:B.symbol};F=send_request(url=D,headers=B.headers,method=_E,params=E,payload=_A,show_log=show_log);A=pd.DataFrame(F);A=A[_PRICE_DEPTH_MAP.keys()];A.rename(columns=_PRICE_DEPTH_MAP,inplace=_C)
		if get_all==_B:G=['price',_N,'buy_volume','sell_volume','undefined_volume'];A=A[G]
		A.source=B.data_source
		if to_df:return A
		else:return A.to_json(orient=_F)
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('quote','mas',Quote)