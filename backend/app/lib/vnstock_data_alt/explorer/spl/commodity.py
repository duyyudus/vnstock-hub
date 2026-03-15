_D='commodity'
_C='close'
_B='SPL.ext'
_A=None
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from.spl_fetcher import SPLFetcher
from typing import Dict,Any,Optional,List,Union
from datetime import timedelta
from app.lib._vnstock_shared.core.utils.lookback import get_start_date_from_lookback
from app.lib._vnstock_shared.compat import agg_execution
from app.lib._vnstock_shared.core.utils.logger import get_logger
logger=get_logger(__name__)
class CommodityPrice:
	'\n    Lớp cung cấp các phương thức để lấy dữ liệu giá hàng hóa từ nguồn SPL.\n    '
	def __init__(A,start=_A,end=_A,length=_A,show_log=False):
		"\n        Khởi tạo đối tượng CommodityPrice với tùy chọn ngày bắt đầu và kết thúc mặc định.\n\n        Các tham số:\n            start (str, optional): Ngày bắt đầu mặc định (định dạng 'YYYY-MM-DD'). Mặc định là None.\n            end (str, optional): Ngày kết thúc mặc định (định dạng 'YYYY-MM-DD'). Mặc định là None.\n            length (str, int, optional): Khoảng thời gian mặc định cần lấy dữ liệu. Mặc định là '1Y'.\n        ";B=length;A.fetcher=SPLFetcher();A.default_start=start;A.default_end=end;A.default_length=B if B is not _A else'1Y'
		if not show_log:logger.setLevel('CRITICAL')
	def _fetch_commodity(C,ticker,start=_A,end=_A,interval='1d',columns=_A,length=_A):
		"\n        Lấy dữ liệu giá hàng hóa từ API SPL.\n\n        Các tham số:\n            ticker (str): Mã hàng hóa cần lấy dữ liệu.\n            start (str, optional): Ngày bắt đầu (định dạng 'YYYY-MM-DD').\n                Ưu tiên tham số nếu có, mặc định là giá trị khởi tạo.\n            end (str, optional): Ngày kết thúc (định dạng 'YYYY-MM-DD').\n                Ưu tiên tham số nếu có, mặc định là giá trị khởi tạo.\n            interval (str, optional): Khoảng thời gian (mặc định '1d').\n            columns (List, optional): Danh sách cột cần lấy.\n                Mặc định là None (lấy tất cả).\n        \n        Giá trị trả về:\n            pd.DataFrame: Dữ liệu giá hàng hóa với time làm index.\n        ";O=length;N=columns;M='%Y-%m-%d';E=end;D=start;B='time';F=E or C.default_end
		if F is _A:F=datetime.now().strftime(M)
		G=D
		if G is _A:
			P=O if O is not _A else C.default_length
			if P is not _A:
				H=str(P).upper()
				if H.endswith('B'):H=H[:-1]+'D'
				G=get_start_date_from_lookback(lookback_length=H,end_date=F)
			else:G=C.default_start
		I={'ticker':ticker,'interval':interval,'type':_D};J=ZoneInfo('Asia/Ho_Chi_Minh');D=G;E=F
		if D:K=datetime.strptime(D,M);K=K.replace(hour=0,minute=0,second=0,microsecond=0,tzinfo=J);I['from']=int(K.timestamp())
		if E:L=datetime.strptime(E,M);L=L.replace(hour=23,minute=59,second=59,microsecond=999999,tzinfo=J);I['to']=int(L.timestamp())
		C.fetcher.validate(I);Q=C.fetcher.fetch(endpoint='/historical/prices/ohlcv',params=I);A=C.fetcher.to_dataframe(Q['data']);A[B]=pd.to_datetime(A[B])
		if A[B].dt.tz is _A:A[B]=A[B].dt.tz_localize(J)
		else:A[B]=A[B].dt.tz_convert(J)
		A[B]=A[B].dt.tz_localize(_A);A.set_index(B,inplace=True)
		if N is not _A:return A[N]
		return A
	def _gold_vn_buy(A,start=_A,end=_A,length=_A):'Lấy giá vàng Việt Nam (mua vào).';return A._fetch_commodity('GOLD:VN:BUY',start,end,columns=[_C],length=length)
	def _gold_vn_sell(A,start=_A,end=_A,length=_A):'Lấy giá vàng Việt Nam (bán ra).';return A._fetch_commodity('GOLD:VN:SELL',start,end,columns=[_C],length=length)
	@agg_execution(_B)
	def gold_vn(self,start=_A,end=_A,length=_A):'Lấy giá vàng Việt Nam.';B=length;A=start;D=self._gold_vn_buy(A,end,length=B);E=self._gold_vn_sell(A,end,length=B);C=pd.concat([D,E],axis=1);C.columns=['buy','sell'];return C
	@agg_execution(_B)
	def gold_global(self,start=_A,end=_A,length=_A):'Lấy giá vàng thế giới.';return self._fetch_commodity('GC=F',start,end,length=length)
	@agg_execution(_B)
	def _gas_ron92(self,start=_A,end=_A,length=_A):'Lấy giá xăng RON92 tại Việt Nam.';return self._fetch_commodity('GAS:RON92:VN',start,end,columns=[_C],length=length)
	@agg_execution(_B)
	def _gas_ron95(self,start=_A,end=_A,length=_A):'Lấy giá xăng RON95 tại Việt Nam.';return self._fetch_commodity('GAS:RON95:VN',start,end,columns=[_C],length=length)
	@agg_execution(_B)
	def _oil_do(self,start=_A,end=_A,length=_A):'Lấy giá dầu DO tại Việt Nam.';return self._fetch_commodity('GAS:DO:VN',start,end,columns=[_C],length=length)
	@agg_execution(_B)
	def gas_vn(self,start=_A,end=_A,length=_A):'Lấy giá xăng và dầu DO tại Việt Nam.';D=length;C=end;B=start;A=self;F=A._gas_ron92(B,C,length=D);G=A._gas_ron95(B,C,length=D);H=A._oil_do(B,C,length=D);E=pd.concat([G,F,H],axis=1);E.columns=['ron95','ron92','oil_do'];return E
	@agg_execution(_B)
	def oil_crude(self,start=_A,end=_A,length=_A):'Lấy giá dầu thô.';return self._fetch_commodity('CL=F',start,end,length=length)
	@agg_execution(_B)
	def gas_natural(self,start=_A,end=_A,length=_A):'Lấy giá khí thiên nhiên.';return self._fetch_commodity('NG=F',start,end,length=length)
	@agg_execution(_B)
	def coke(self,start=_A,end=_A,length=_A):'Lấy giá than cốc.';return self._fetch_commodity('ICEEUR:NCF1!',start,end,length=length)
	@agg_execution(_B)
	def steel_d10(self,start=_A,end=_A,length=_A):'Lấy giá thép D10 tại Việt Nam.';return self._fetch_commodity('STEEL:D10:VN',start,end,columns=[_C],length=length)
	@agg_execution(_B)
	def iron_ore(self,start=_A,end=_A,length=_A):'Lấy giá quặng sắt.';return self._fetch_commodity('COMEX:TIO1!',start,end,length=length)
	@agg_execution(_B)
	def steel_hrc(self,start=_A,end=_A,length=_A):'Lấy giá thép HRC.';return self._fetch_commodity('COMEX:HRC1!',start,end,length=length)
	@agg_execution(_B)
	def fertilizer_ure(self,start=_A,end=_A,length=_A):'Lấy giá phân ure.';return self._fetch_commodity('CBOT:UME1!',start,end,length=length)
	@agg_execution(_B)
	def soybean(self,start=_A,end=_A,length=_A):'Lấy giá đậu tương.';return self._fetch_commodity('ZM=F',start,end,length=length)
	@agg_execution(_B)
	def corn(self,start=_A,end=_A,length=_A):'Lấy giá ngô (bắp).';return self._fetch_commodity('ZC=F',start,end,length=length)
	@agg_execution(_B)
	def sugar(self,start=_A,end=_A,length=_A):'Lấy giá đường.';return self._fetch_commodity('SB=F',start,end,length=length)
	@agg_execution(_B)
	def pork_north_vn(self,start=_A,end=_A,length=_A):'Lấy giá heo hơi miền Bắc Việt Nam.';return self._fetch_commodity('PIG:NORTH:VN',start,end,columns=[_C],length=length)
	@agg_execution(_B)
	def pork_china(self,start=_A,end=_A,length=_A):'Lấy giá heo hơi Trung Quốc.';return self._fetch_commodity('PIG:CHINA',start,end,columns=[_C],length=length)
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register(_D,'spl',CommodityPrice)