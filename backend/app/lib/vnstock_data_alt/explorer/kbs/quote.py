'Quote module for KB Securities (KBS) data source.'
_J='index'
_I='KBS.ext'
_H='source'
_G='GET'
_F='symbol'
_E=True
_D='time'
_C='%Y-%m-%d'
_B=False
_A=None
import pandas as pd,json
from datetime import datetime,timedelta
from typing import Optional,Union,List
from app.lib._vnstock_shared.compat import agg_execution
from app.lib._vnstock_shared.core.models import TickerModel
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib._vnstock_shared.core.utils.parser import get_asset_type
from app.lib.vnstock_data_alt.core.utils.parser import convert_derivative_symbol
from app.lib._vnstock_shared.core.utils.lookback import get_start_date_from_lookback,interpret_lookback_length
from app.lib.vnstock_data_alt.core.utils.client import send_request,ProxyConfig
from app.lib.vnstock_data_alt.core.utils.transform import process_match_types
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
from app.lib.vnstock_data_alt.explorer.kbs.const import _IIS_BASE_URL,_SAS_HISTORICAL_QUOTES_URL,_INDEX_MAPPING,_OHLC_MAP,_OHLC_DTYPE,_INTERVAL_MAP,_RESAMPLE_MAP,_INTRADAY_MAP,_INTRADAY_DTYPE
logger=get_logger(__name__)
class Quote:
	'\n    Lớp truy cập dữ liệu giá lịch sử từ KB Securities (KBS).\n    '
	def __init__(A,symbol,random_agent=_B,proxy_config=_A,show_log=_B,proxy_mode=_A,proxy_list=_A):
		"\n        Khởi tạo Quote client cho KBS.\n\n        Args:\n            symbol: Mã chứng khoán (VD: 'ACB', 'VNM').\n            random_agent: Sử dụng user agent ngẫu nhiên. Mặc định False.\n            proxy_config: Cấu hình proxy. Mặc định None.\n            show_log: Hiển thị log debug. Mặc định False.\n            proxy_mode: Chế độ proxy (try, rotate, random, single). Mặc định None.\n            proxy_list: Danh sách proxy URLs. Mặc định None.\n        ";E=proxy_mode;D=show_log;C=proxy_config;B=proxy_list;A.symbol=symbol.upper();A.data_source='KBS';A.asset_type=get_asset_type(A.symbol)
		if A.asset_type=='derivative':
			try:F=convert_derivative_symbol(A.symbol);logger.info(f"Converted derivative symbol {A.symbol} to {F} (KRX format)");A.symbol=F
			except Exception as H:logger.debug(f"Symbol conversion skipped for {A.symbol}: {H}")
		A.base_url=_IIS_BASE_URL;A.headers=get_headers(data_source=A.data_source,random_agent=random_agent);A.show_log=D;A.interval_map=_INTERVAL_MAP
		if C is _A:
			I=E if E else'try';G='direct'
			if B and len(B)>0:G='proxy'
			A.proxy_config=ProxyConfig(proxy_mode=I,proxy_list=B,request_mode=G)
		else:A.proxy_config=C
		if not D:logger.setLevel('CRITICAL')
		if A.asset_type==_J:
			if A.symbol not in _INDEX_MAPPING:J=', '.join(_INDEX_MAPPING.keys());raise ValueError(f"Mã chỉ số '{A.symbol}' không được hỗ trợ bởi KBS. Các chỉ số hợp lệ: {J}")
	def _input_validation(A,start,end,interval):
		'\n        Validate input parameters.\n\n        Args:\n            start: Ngày bắt đầu (YYYY-MM-DD hoặc DD-MM-YYYY).\n            end: Ngày kết thúc (YYYY-MM-DD hoặc DD-MM-YYYY).\n            interval: Khung thời gian (1m, 5m, 15m, 30m, 1H, 1D, 1W, 1M).\n\n        Returns:\n            TickerModel instance với dữ liệu đã validate.\n\n        Raises:\n            ValueError: Nếu interval không hợp lệ.\n        ';B=interval;C=TickerModel(symbol=A.symbol,start=start,end=end,interval=B)
		if B not in A.interval_map:D=', '.join(A.interval_map.keys());raise ValueError(f"Giá trị interval không hợp lệ: {B}. Vui lòng chọn: {D}")
		return C
	def _format_date_for_api(C,date_str):
		'\n        Chuyển đổi ngày từ YYYY-MM-DD sang DD-MM-YYYY cho API KBS.\n\n        Args:\n            date_str: Ngày dạng YYYY-MM-DD.\n\n        Returns:\n            Ngày dạng DD-MM-YYYY.\n        ';A=date_str
		try:B=datetime.strptime(A,_C);return B.strftime('%d-%m-%Y')
		except ValueError:return A
	@agg_execution(_I)
	def history(self,start=_A,end=_A,interval='1D',to_df=_E,show_log=_B,count_back=_A,floating=2,length=_A,get_all=_B):
		'\n        Tải lịch sử giá của mã chứng khoán từ KBS.\n\n        Args:\n            start: Ngày bắt đầu (YYYY-MM-DD hoặc DD-MM-YYYY). Bắt buộc nếu không có length hoặc count_back.\n            end: Ngày kết thúc (YYYY-MM-DD hoặc DD-MM-YYYY). Mặc định None (lấy đến hiện tại).\n            interval: Khung thời gian trích xuất dữ liệu. Giá trị nhận: 1m, 5m, 15m, 30m, 1H, 1D, 1W, 1M. Mặc định "1D".\n            to_df: Trả về DataFrame. Mặc định True. False để trả về JSON.\n            show_log: Hiển thị log debug.\n            count_back: Số lượng nến (bars) cần lấy.\n            floating: Số chữ số thập phân cho giá. Mặc định 2.\n            length: Khoảng thời gian phân tích (vd: \'3M\', 150, \'150\'). Nhận giá trị chuỗi (vd 3M), số ngày (int/str), hoặc số bars (vd \'100b\').\n            get_all: Lấy tất cả các cột từ API response. Mặc định False (chỉ lấy cột chuẩn hóa).\n\n        Returns:\n            DataFrame hoặc JSON string chứa dữ liệu OHLCV.\n\n        Examples:\n            >>> quote = Quote(\'ACB\')\n            >>> df = quote.history(start=\'2024-01-01\', end=\'2024-12-31\', interval=\'1D\')\n            >>> print(df.columns.tolist())\n            [\'time\', \'open\', \'high\', \'low\', \'close\', \'volume\']\n            \n            >>> # Sử dụng length để lấy dữ liệu 1 tháng gần nhất\n            >>> df_1m = quote.history(length=\'1M\', interval=\'1D\')\n            \n            >>> # Lấy 100 nến dữ liệu\n            >>> df_100 = quote.history(count_back=100, interval=\'1D\')\n            \n            >>> # Lấy dữ liệu 150 ngày\n            >>> df_150d = quote.history(length=150, interval=\'1D\')\n            \n            >>> # Lấy dữ liệu 3 tháng với kết thúc vào ngày cụ thể\n            >>> df_3m = quote.history(end=\'2024-12-31\', length=\'3M\', interval=\'1D\')\n            \n            >>> # Lấy tất cả các cột (bao gồm cả cột value)\n            >>> df_all = quote.history(length=\'1M\', interval=\'1D\', get_all=True)\n        ';R=get_all;Q=floating;P=show_log;O='va';J=length;I='value';G=count_back;F=interval;D=end;C=start;B=self
		if D is _A:D=datetime.now().strftime(_C)
		if C is _A:
			if J is not _A:
				H=str(J)
				if H.endswith('b'):G=int(H[:-1]);C=get_start_date_from_lookback(lookback_length=H,end_date=D)
				else:C=get_start_date_from_lookback(lookback_length=H,end_date=D)
			elif G is not _A:
				if F=='1D':C=(datetime.strptime(D,_C)-timedelta(days=G*2)).strftime(_C)
				elif F=='1H':C=(datetime.strptime(D,_C)-timedelta(days=G//6)).strftime(_C)
				elif F=='1m':C=(datetime.strptime(D,_C)-timedelta(days=1)).strftime(_C)
				else:C=get_start_date_from_lookback(lookback_length='1M',end_date=D)
			else:raise ValueError("Tham số 'start' là bắt buộc nếu không cung cấp 'length' hoặc 'count_back'.")
		if C is not _A:S=B._input_validation(C,D,F)
		else:S=TickerModel(symbol=B.symbol,start=D,end=D,interval=F);S.start=C
		T=B._format_date_for_api(C);U=B._format_date_for_api(D);K=B.interval_map[F]
		if B.asset_type==_J:V=f"{_IIS_BASE_URL}/index/{B.symbol}/data_{K}"
		else:V=f"{_IIS_BASE_URL}/stocks/{B.symbol}/data_{K}"
		Y={'sdate':T,'edate':U};L=send_request(url=V,headers=B.headers,method=_G,params=Y,show_log=P or B.show_log,proxy_list=B.proxy_config.proxy_list,proxy_mode=B.proxy_config.proxy_mode,request_mode=B.proxy_config.request_mode)
		if not L:raise ValueError(f"Không tìm thấy dữ liệu cho mã {B.symbol}. Vui lòng kiểm tra lại mã chứng khoán hoặc khoảng thời gian.")
		W=f"data_{K}"
		if W not in L:raise ValueError(f"Không tìm thấy dữ liệu cho interval {F}. Vui lòng kiểm tra lại khoảng thời gian hoặc interval.")
		M=L[W]
		if not M:raise ValueError(f"Dữ liệu trống cho mã {B.symbol} với interval {F}.")
		if not to_df:return json.dumps(M)
		A=pd.DataFrame(M);N=_A
		if O in A.columns:
			N=A[O].copy()
			if not R:A=A.drop(columns=[O])
		A=A.rename(columns=_OHLC_MAP)
		if R and N is not _A:A[I]=N
		A[_D]=pd.to_datetime(A[_D]);A=A.sort_values(_D).reset_index(drop=_E)
		for(E,X)in _OHLC_DTYPE.items():
			if E in A.columns:
				try:A[E]=A[E].astype(X)
				except ValueError:A[E]=A[E].astype(float).astype(X)
		if I in A.columns:A[I]=A[I].astype('float64')
		Z=['open','high','low','close']
		for E in Z:
			if E in A.columns:
				A[E]=A[E]/1000
				if Q is not _A:A[E]=A[E].round(Q)
		A=A[A[_D]>=pd.to_datetime(C)].reset_index(drop=_E)
		if G is not _A:A=A.tail(G).reset_index(drop=_E)
		A.attrs[_F]=B.symbol;A.attrs[_H]=B.data_source;A.attrs['interval']=F;A.attrs['start']=C;A.attrs['end']=D;A.attrs['length']=J
		if P or B.show_log:logger.info(f"Truy xuất thành công {len(A)} bản ghi giá cho {B.symbol} ({F}) từ {T} đến {U}.")
		return A
	@agg_execution(_I)
	def intraday(self,page_size=1000,page=1,to_df=_E,get_all=_B,show_log=_B,floating=2):
		"\n        Truy xuất dữ liệu khớp lệnh intraday (real-time matching data) của mã chứng khoán từ KBS.\n        \n        Mặc định trả về các cột chuẩn hóa (time, price, volume, match_type, id).\n        Sử dụng get_all=True để lấy tất cả các cột từ API response.\n\n        Args:\n            page_size: Số lượng bản ghi trên mỗi trang (mặc định 1000).\n                       Thường 1 ngày có thể lên đến 100K dòng (VN30 derivatives) hoặc 50-70K (cổ phiếu cơ sở).\n            page: Trang dữ liệu (mặc định 1).\n            to_df: Trả về DataFrame. Mặc định True. False để trả về JSON.\n            get_all: Lấy tất cả các cột từ API response. Mặc định False (chỉ lấy cột chuẩn hóa).\n            show_log: Hiển thị log debug.\n            floating: Số chữ số thập phân cho giá. Mặc định 2. Nếu None sẽ không làm tròn.\n\n        Returns:\n            DataFrame hoặc JSON string chứa dữ liệu khớp lệnh intraday.\n            \n            **Cột chuẩn hóa (Core columns):**\n            - time: Thời gian giao dịch (YYYY-MM-DD HH:MM:SS)\n            - price: Giá khớp\n            - volume: Khối lượng khớp\n            - match_type: Loại khớp lệnh (buy, sell, atc, ato)\n            - id: ID giao dịch (từ KBS: timestamp + price + volume)\n            \n            **Cột bổ sung (nếu get_all=True):**\n            - trading_date: Ngày giao dịch (DD/MM/YYYY)\n            - symbol: Mã chứng khoán\n            - price_change: Thay đổi giá so với lần trước\n            - accumulated_volume: Khối lượng tích lũy\n            - accumulated_value: Giá trị tích lũy\n\n        Examples:\n            >>> quote = Quote('ACB')\n            \n            >>> # Lấy 1000 bản ghi (cột chuẩn hóa)\n            >>> df = quote.intraday(page_size=1000)\n            \n            >>> # Lấy trang thứ 2\n            >>> df_page2 = quote.intraday(page=2, page_size=100)\n            \n            >>> # Lấy tất cả các cột\n            >>> df_all = quote.intraday(get_all=True)\n        ";U='side';T='volume';S='data';R='page';Q=floating;P=show_log;O=get_all;N='_';M='match_volume';J=page;I=page_size;H='match_type';G='price';E='timestamp';C=self
		if C.asset_type==_J:raise ValueError(f"Dữ liệu intraday không được hỗ trợ cho chỉ số {C.symbol}.")
		V=f"{_IIS_BASE_URL}/trade/history/{C.symbol}";W={R:J,'limit':I};K=send_request(url=V,headers=C.headers,method=_G,params=W,show_log=P or C.show_log,proxy_list=C.proxy_config.proxy_list,proxy_mode=C.proxy_config.proxy_mode,request_mode=C.proxy_config.request_mode)
		if not K or S not in K:raise ValueError(f"Không tìm thấy dữ liệu intraday cho mã {C.symbol}. Vui lòng kiểm tra lại mã chứng khoán hoặc thử lại sau.")
		L=K.get(S,[])
		if not L:raise ValueError(f"Dữ liệu intraday trống cho mã {C.symbol}.")
		if not to_df:return json.dumps(L)
		A=pd.DataFrame(L);A=A.rename(columns=_INTRADAY_MAP)
		if E in A.columns:A[E]=A[E].apply(lambda x:pd.to_datetime(x.rsplit(':',1)[0])if isinstance(x,str)else x)
		for(F,X)in _INTRADAY_DTYPE.items():
			if F in A.columns:
				try:A[F]=A[F].astype(X)
				except(ValueError,TypeError):pass
		A=A.sort_values(E,ascending=_B).reset_index(drop=_E);B=pd.DataFrame()
		if E in A.columns:B[_D]=A[E]
		if G in A.columns:
			B[G]=A[G]/1000
			if Q is not _A:B[G]=B[G].round(Q)
		if M in A.columns:B[T]=A[M]
		if U in A.columns:B[H]=A[U].fillna('')
		if H in B.columns:B[_D]=pd.to_datetime(B[_D]);B=process_match_types(B,asset_type=C.asset_type,source='KBS');B[H]=B[H].str.lower()
		if E in A.columns:B['id']=A[E].astype(str).str.replace(' ',N).str.replace(':','')+N+A[G].astype(str).str.replace('.','')+N+A[M].astype(str)
		if O:
			Y=['trading_date',_F,'price_change','accumulated_volume','accumulated_value']
			for F in Y:
				if F in A.columns:B[F]=A[F]
			D=B
		else:D=B[[_D,G,T,H,'id']]
		D.attrs[_F]=C.symbol;D.attrs[_H]=C.data_source;D.attrs[R]=J;D.attrs['page_size']=I;D.attrs['get_all']=O
		if P or C.show_log:logger.info(f"Truy xuất thành công {len(D)} bản ghi khớp lệnh intraday cho {C.symbol} (trang {J}, page_size={I}).")
		return D
	@agg_execution(_I)
	def price_depth(self,show_log=_B):
		'\n        Lấy thông tin độ sâu giá (price depth/order book) của mã chứng khoán.\n\n        Args:\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa thông tin độ sâu giá.\n\n        Note:\n            API endpoint này có thể cần authentication hoặc chưa được công khai.\n        ';C=show_log;A=self;E=f"{_IIS_BASE_URL}/stock/matched-by-price/{A.symbol}";D=send_request(url=E,headers=A.headers,method=_G,show_log=C or A.show_log,proxy_list=A.proxy_config.proxy_list,proxy_mode=A.proxy_config.proxy_mode,request_mode=A.proxy_config.request_mode,auto_fetch=A.proxy_config.auto_fetch,validate_proxies=A.proxy_config.validate_proxies,prefer_speed=A.proxy_config.prefer_speed)
		if not D:raise ValueError(f"Không tìm thấy dữ liệu độ sâu giá cho mã {A.symbol}.")
		B=pd.DataFrame(D);B.attrs[_F]=A.symbol;B.attrs[_H]=A.data_source
		if C or A.show_log:logger.info(f"Truy xuất thành công dữ liệu độ sâu giá cho {A.symbol}.")
		return B
	@agg_execution(_I)
	def summary(self,show_log=_B,to_df=_E):
		'\n        Lấy thông tin tóm tắt (Snapshot) của mã chứng khoán.\n\n        Args:\n            show_log: Hiển thị log debug.\n            to_df: Trả về DataFrame. Mặc định True. False để trả về JSON (dict chuỗi).\n\n        Returns:\n            DataFrame hoặc JSON string chứa thông tin tóm tắt.\n        ';D=show_log;A=self;from app.lib.vnstock_data_alt.explorer.kbs.const import _STOCK_INFO_URL as E;F=f"{E}/info/{A.symbol}";G={'l':'1'};B=send_request(url=F,headers=A.headers,method=_G,params=G,show_log=D or A.show_log,proxy_list=A.proxy_config.proxy_list,proxy_mode=A.proxy_config.proxy_mode,request_mode=A.proxy_config.request_mode)
		if not B:raise ValueError(f"Không tìm thấy dữ liệu tóm tắt cho mã {A.symbol}.")
		if not to_df:return json.dumps(B)
		C=pd.DataFrame([B]);C.attrs[_F]=A.symbol;C.attrs[_H]=A.data_source
		if D or A.show_log:logger.info(f"Truy xuất thành công dữ liệu tóm tắt cho {A.symbol}.")
		return C
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('quote','kbs',Quote)