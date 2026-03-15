'History module for VCI.'
_M='symbol'
_L='Vui lòng nhập mã chứng khoán cần truy xuất khi khởi tạo Trading Class.'
_K=': Dữ liệu khớp lệnh không thể truy cập trong thời gian chuẩn bị phiên mới. Vui lòng quay lại sau.'
_J='preparing'
_I='data_status'
_H='is_trading_hour'
_G='VCI.ext'
_F='records'
_E='time'
_D='POST'
_C=True
_B=False
_A=None
import pandas as pd
from datetime import datetime,timedelta
from app.lib._vnstock_shared.core.utils.lookback import get_start_date_from_lookback
from app.lib._vnstock_shared.compat import agg_execution
from typing import Dict,Optional,Union,List
from app.lib.vnstock_alt.explorer.vci.const import _TRADING_URL,_INTERVAL_MAP,_OHLC_MAP,_RESAMPLE_MAP,_OHLC_DTYPE,_INTRADAY_URL,_INTRADAY_MAP,_INTRADAY_DTYPE,_INDEX_MAPPING
from app.lib.vnstock_data_alt.explorer.vci.const import _PRICE_DEPTH_MAP
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
	'\n    Cấu hình truy cập dữ liệu lịch sử giá chứng khoán từ VCI.\n    Cho phép cấu hình proxy thông qua object ProxyConfig.\n    \n    Hỗ trợ các tính năng proxy nâng cao:\n    - auto_fetch: Tự động lấy proxy từ proxyscrape API\n    - validate_proxies: Kiểm tra tính hợp lệ của proxy\n    - prefer_speed: Ưu tiên proxy có tốc độ tốt nhất\n    '
	def __init__(A,symbol,random_agent=_B,proxy_config=_A,show_log=_C,proxy_mode=_A,proxy_list=_A):
		E=proxy_mode;D=show_log;C=proxy_config;B=proxy_list;A.symbol=symbol.upper();A.data_source='VCI';A._history=_A;A.asset_type=get_asset_type(A.symbol);A.base_url=_TRADING_URL;A.headers=get_headers(data_source=A.data_source,random_agent=random_agent);A.interval_map=_INTERVAL_MAP;A.show_log=D
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
		if A.interval not in B.interval_map:raise ValueError(f"Giá trị interval không hỗ trợ bởi VCI: {A.interval}. Các interval được hỗ trợ: 1m, 5m, 15m, 30m, 1H, 1D, 1W, 1M")
		return A
	@agg_execution(_G)
	def history(self,start=_A,end=_A,interval='1D',to_df=_C,show_log=_B,count_back=_A,floating=2,length=_A):
		'\n        Tải lịch sử giá của mã chứng khoán từ nguồn dữ liệu VCI.\n\n        Tham số:\n            - start (tùy chọn): thời gian bắt đầu lấy dữ liệu (YYYY-MM-DD). Bắt buộc nếu không có length hoặc count_back.\n            - end (tùy chọn): thời gian kết thúc lấy dữ liệu. Mặc định là None, chương trình tự động lấy thời điểm hiện tại.\n            - interval (tùy chọn): Khung thời gian trích xuất dữ liệu. Mặc định là "1D".\n            - to_df (tùy chọn): Chuyển đổi dữ liệu lịch sử trả về dưới dạng DataFrame. Mặc định là True.\n            - show_log (tùy chọn): Hiển thị thông tin log. Mặc định là False.\n            - count_back (tùy chọn): Số lượng dữ liệu trả về từ thời điểm cuối.\n            - floating (tùy chọn): Số chữ số thập phân cho giá. Mặc định là 2.\n            - length (tùy chọn): Khoảng thời gian phân tích (vd: \'3M\', 150, \'100b\').\n        ';O=length;I=interval;E=start;D='%Y-%m-%d';C=count_back;B=end;A=self
		if B is _A:B=datetime.now().strftime(D)
		if E is _A:
			if O is not _A:
				J=str(O)
				if J.endswith('b'):C=int(J[:-1]);E=get_start_date_from_lookback(lookback_length=J,end_date=B)
				else:E=get_start_date_from_lookback(lookback_length=J,end_date=B)
			elif C is not _A:
				if I=='1D':E=(datetime.strptime(B,D)-timedelta(days=C*2)).strftime(D)
				elif I=='1H':E=(datetime.strptime(B,D)-timedelta(days=C//6)).strftime(D)
				elif I=='1m':E=(datetime.strptime(B,D)-timedelta(days=1)).strftime(D)
				else:E=get_start_date_from_lookback(lookback_length='1M',end_date=B)
			else:raise ValueError("Tham số 'start' là bắt buộc nếu không cung cấp 'length' hoặc 'count_back'.")
		K=A._input_validation(E,B,I);L=datetime.strptime(K.start,D)
		if B is not _A:
			G=datetime.strptime(K.end,D)+pd.Timedelta(days=1)
			if L>G:raise ValueError('Thời gian bắt đầu không thể lớn hơn thời gian kết thúc.')
			P=int(G.timestamp())
		else:G=datetime.now()+pd.Timedelta(days=1);P=int(G.timestamp())
		Q=A.interval_map[K.interval];H=1000;M=pd.bdate_range(start=L,end=G)
		if C is _A and B is not _A:
			N=Q
			if N=='ONE_DAY':H=len(M)+1
			elif N=='ONE_HOUR':H=len(M)*6.5+1
			elif N=='ONE_MINUTE':H=len(M)*6.5*60+1
		else:H=C if C is not _A else 1000
		S=f"{A.base_url}chart/OHLCChart/gap-chart";T={'timeFrame':Q,'symbols':[A.symbol],'to':P,'countBack':H};R=send_request(url=S,headers=A.headers,method=_D,payload=T,show_log=show_log,proxy_list=A.proxy_config.proxy_list,proxy_mode=A.proxy_config.proxy_mode,request_mode=A.proxy_config.request_mode)
		if not R:raise ValueError('Không tìm thấy dữ liệu. Vui lòng kiểm tra lại mã chứng khoán hoặc thời gian truy xuất.')
		F=ohlc_to_df(data=R[0],column_map=_OHLC_MAP,dtype_map=_OHLC_DTYPE,asset_type=A.asset_type,symbol=A.symbol,source=A.data_source,interval=K.interval,floating=floating,resample_map=_RESAMPLE_MAP);F=F[F[_E]>=L].reset_index(drop=_C)
		if C is not _A:F=F.tail(C)
		if to_df:return F
		else:return F.to_json(orient=_F)
	@agg_execution(_G)
	def intraday(self,page_size=100,last_time=_A,to_df=_C,show_log=_B):
		'\n        Truy xuất dữ liệu khớp lệnh của mã chứng khoán bất kỳ từ nguồn dữ liệu VCI\n\n        Tham số:\n            - page_size (tùy chọn): Số lượng dữ liệu trả về trong một lần request. Mặc định là 100. \n            - last_time (tùy chọn): Thời gian cắt dữ liệu, dùng để lấy dữ liệu sau thời gian cắt. Mặc định là None.\n            - to_df (tùy chọn): Chuyển đổi dữ liệu lịch sử trả về dưới dạng DataFrame. Mặc định là True.\n            - show_log (tùy chọn): Hiển thị thông tin log giúp debug dễ dàng. Mặc định là False.\n        ';C=page_size;A=self
		if A.asset_type=='index':raise ValueError(f"Dữ liệu intraday không được hỗ trợ cho chỉ số {A.symbol}.")
		B=trading_hours(_A)
		if B[_H]is _B and B[_I]==_J:raise ValueError(str(B[_E])+_K)
		if A.symbol is _A:raise ValueError(_L)
		if C>30000:logger.warning('Bạn đang yêu cầu truy xuất quá nhiều dữ liệu, điều này có thể gây lỗi quá tải.')
		E=f"{A.base_url}{_INTRADAY_URL}/LEData/getAll";F={_M:A.symbol,'limit':C,'truncTime':last_time};G=send_request(url=E,headers=A.headers,method=_D,payload=F,show_log=show_log,proxy_list=A.proxy_config.proxy_list,proxy_mode=A.proxy_config.proxy_mode,request_mode=A.proxy_config.request_mode,auto_fetch=A.proxy_config.auto_fetch,validate_proxies=A.proxy_config.validate_proxies,prefer_speed=A.proxy_config.prefer_speed);D=intraday_to_df(data=G,column_map=_INTRADAY_MAP,dtype_map=_INTRADAY_DTYPE,symbol=A.symbol,asset_type=A.asset_type,source=A.data_source)
		if to_df:return D
		else:return D.to_json(orient=_F)
	@agg_execution(_G)
	def price_depth(self,to_df=_C,show_log=_B):
		'\n        Truy xuất thống kê độ bước giá & khối lượng khớp lệnh của mã chứng khoán bất kỳ từ nguồn dữ liệu VCI.\n\n        Tham số:\n            - to_df (tùy chọn): Chuyển đổi dữ liệu lịch sử trả về dưới dạng DataFrame. Mặc định là True.\n            - show_log (tùy chọn): Hiển thị thông tin log giúp debug dễ dàng. Mặc định là False.\n        ';A=self;C=trading_hours(_A)
		if C[_H]is _B and C[_I]==_J:raise ValueError(str(C[_E])+_K)
		if A.symbol is _A:raise ValueError(_L)
		D=f"{A.base_url}{_INTRADAY_URL}/AccumulatedPriceStepVol/getSymbolData";E={_M:A.symbol};F=send_request(url=D,headers=A.headers,method=_D,payload=E,show_log=show_log,proxy_list=A.proxy_config.proxy_list,proxy_mode=A.proxy_config.proxy_mode,request_mode=A.proxy_config.request_mode,auto_fetch=A.proxy_config.auto_fetch,validate_proxies=A.proxy_config.validate_proxies,prefer_speed=A.proxy_config.prefer_speed);B=pd.DataFrame(F);B=B[_PRICE_DEPTH_MAP.keys()];B.rename(columns=_PRICE_DEPTH_MAP,inplace=_C);B.source=A.data_source
		if to_df:return B
		else:return B.to_json(orient=_F)
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('quote','vci',Quote)