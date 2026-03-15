'History module for vnd.'
_F='VND.ext'
_E='VND'
_D=True
_C=False
_B='time'
_A=None
from typing import List,Dict,Optional
from datetime import datetime,timedelta
from typing import List,Dict,Optional,Union
from app.lib._vnstock_shared.core.utils.lookback import get_start_date_from_lookback
from.const import _CHART_BASE,_INTERVAL_MAP,_OHLC_MAP,_OHLC_DTYPE,_INTRADAY_MAP,_INTRADAY_DTYPE,_INDEX_MAPPING
from app.lib._vnstock_shared.core.models import TickerModel
from app.lib._vnstock_shared.core.utils.interval import normalize_interval
import requests,pandas as pd
from app.lib._vnstock_shared.compat import agg_execution
from app.lib._vnstock_shared.core.utils.parser import get_asset_type
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
logger=get_logger(__name__)
class Quote:
	'\n    VND data source for fetching stock market data, accommodating requests with large date ranges.\n    '
	def __init__(A,symbol,random_agent=_C,show_log=_C):
		A.symbol=symbol.upper();A._history=_A;A.asset_type=get_asset_type(A.symbol);A.base_url=_CHART_BASE;A.headers=get_headers(data_source=_E,random_agent=random_agent);A.interval_map=_INTERVAL_MAP;A.data_source=_E
		if not show_log:logger.setLevel('CRITICAL')
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
		if A.interval not in B.interval_map:raise ValueError(f"Giá trị interval không hỗ trợ bởi VND: {A.interval}. Các interval được hỗ trợ: 1m, 5m, 15m, 30m, 1H, 1D, 1W, 1M")
		return A
	@agg_execution(_F)
	def history(self,start=_A,end=_A,interval='1D',to_df=_D,show_log=_C,count_back=_A,length=_A):
		'\n        Tải lịch sử giá của mã chứng khoán từ nguồn dữ liệu VN Direct.\n\n        Tham số:\n            - start (tùy chọn): thời gian bắt đầu lấy dữ liệu (YYYY-MM-DD). Bắt buộc nếu không có length hoặc count_back.\n            - end (tùy chọn): thời gian kết thúc lấy dữ liệu. Mặc định là None, sẽ lấy thời điểm hiện tại.\n            - interval (tùy chọn): Khung thời gian trích xuất dữ liệu. Mặc định là "1D".\n            - to_df (tùy chọn): Chuyển đổi dữ liệu lịch sử trả về dưới dạng DataFrame. Mặc định là True.\n            - show_log (tùy chọn): Hiển thị thông tin log. Mặc định là False.\n            - count_back (tùy chọn): Số lượng dữ liệu trả về từ thời điểm cuối. Mặc định là 365.\n            - length (tùy chọn): Khoảng thời gian phân tích (vd: \'3M\', 150, \'100b\').\n        ';M=length;L=show_log;G=count_back;F=interval;E=self;C=start;B='%Y-%m-%d';A=end
		if A is _A:A=datetime.now().strftime(B)
		if C is _A:
			if M is not _A:
				I=str(M)
				if I.endswith('b'):G=int(I[:-1]);C=get_start_date_from_lookback(lookback_length=I,end_date=A)
				else:C=get_start_date_from_lookback(lookback_length=I,end_date=A)
			elif G is not _A:
				if F=='1D':C=(datetime.strptime(A,B)-timedelta(days=G*2)).strftime(B)
				elif F=='1H':C=(datetime.strptime(A,B)-timedelta(days=G//6)).strftime(B)
				elif F=='1m':C=(datetime.strptime(A,B)-timedelta(days=1)).strftime(B)
				else:C=get_start_date_from_lookback(lookback_length='1M',end_date=A)
			else:raise ValueError("Tham số 'start' là bắt buộc nếu không cung cấp 'length' hoặc 'count_back'.")
		D=E._input_validation(C,A,F)
		if A is _A:N=int(datetime.now().timestamp())
		else:N=int(datetime.strptime(D.end,B).timestamp())
		P=int(datetime.strptime(D.start,B).timestamp());F=E.interval_map[D.interval];O=f"{E.base_url}/dchart/history?resolution={F}&symbol={E.symbol}&from={P}&to={N}"
		if L:logger.info(f"Tải dữ liệu từ {O}")
		J=requests.get(O,headers=E.headers)
		if J.status_code!=200:raise ConnectionError(f"Failed to fetch data: {J.status_code} - {J.reason}")
		K=J.json()
		if L:logger.info(f"Truy xuất thành công dữ liệu {D.symbol} từ {D.start} đến {D.end}, khung thời gian {D.interval}.")
		H=E._as_df(K,E.asset_type)
		if D.interval not in['1D','1W','1M']:H[_B]=H[_B]+pd.Timedelta(hours=7)
		if G is not _A:H=H.tail(G)
		if to_df:return H
		else:K=H.to_json(orient='records');return K
	def _as_df(C,history_data,asset_type):
		'\n        Chuyển đổi dữ liệu lịch sử giá chứng khoán từ dạng JSON sang DataFrame.\n\n        Tham số:\n            - history_data: Dữ liệu lịch sử giá chứng khoán dạng JSON.\n        Trả về:\n            - DataFrame: Dữ liệu lịch sử giá chứng khoán dưới dạng DataFrame.\n        ';A=pd.DataFrame(history_data);A.drop(columns=['s'],inplace=_D);A.rename(columns=_OHLC_MAP,inplace=_D);A[_B]=pd.to_datetime(A[_B],unit='s')
		for(B,D)in _OHLC_DTYPE.items():A[B]=A[B].astype(D)
		A.attrs['name']=C.symbol;A.attrs['category']=asset_type;A.attrs['source']=_E;A=A[[_B,'open','high','low','close','volume']];return A
	@agg_execution(_F)
	def intraday(self,page_size=100000,to_df=_D,show_log=_C):
		'\n        Truy xuất dữ liệu khớp lệnh của mã chứng khoán bất kỳ từ nguồn dữ liệu VCI\n\n        Tham số:\n            - page_size (tùy chọn): Số lượng dữ liệu trả về trong một lần request. Mặc định là 100. Không giới hạn số lượng tối đa. Tăng số này lên để lấy toàn bộ dữ liêu, ví dụ 10_000.\n            - trunc_time (tùy chọn): Thời gian cắt dữ liệu, dùng để lấy dữ liệu sau thời gian cắt. Mặc định là None.\n            - to_df (tùy chọn): Chuyển đổi dữ liệu lịch sử trả về dưới dạng DataFrame. Mặc định là True. Đặt là False để trả về dữ liệu dạng JSON.\n            - show_log (tùy chọn): Hiển thị thông tin log giúp debug dễ dàng. Mặc định là False.\n        '
		if self.asset_type=='index':raise ValueError(f"Dữ liệu intraday không được hỗ trợ cho chỉ số {self.symbol}.")
		logger.error('Dữ liệu từ VND không còn khả dụng cho Intraday. Chúng tôi đang nghiên cứu cách khắc phục.')
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('quote','vnd',Quote)