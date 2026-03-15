_F='%Y-%m-%d'
_E="Không có trường 'data' trong phản hồi JSON"
_D=None
_C='data'
_B='VNINDEX'
_A='VND.ext'
import pandas as pd,requests
from typing import Union
from datetime import datetime
from app.lib._vnstock_shared.compat import agg_execution
from app.lib.vnstock_data_alt.explorer.vnd.const import _INSIGHT_BASE,_TOP_STOCK_INDEX,_TOP_STOCK_COLS
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
logger=get_logger(__name__)
class TopStock:
	"\n    Lớp để lấy dữ liệu cổ phiếu hàng đầu từ API VND.\n\n    Lớp này cung cấp các phương thức để truy xuất dữ liệu về top 10 cổ phiếu\n    dựa trên các tiêu chí hiệu suất khác nhau như tăng giá, giảm giá,\n    giá trị giao dịch, khối lượng đột biến, giao dịch thỏa thuận đột biến,\n    và mua/bán của nhà đầu tư nước ngoài.\n\n    Thuộc tính:\n    -----------\n    base_url : str\n        URL cơ sở cho API VND.\n    headers : dict\n        Headers HTTP được sử dụng trong các yêu cầu API, bao gồm user-agent.\n    data_source : str\n        Tên nguồn dữ liệu, trong trường hợp này là 'VND'.\n\n    Phương thức:\n    -----------\n    gainer(index: str='VNINDEX', limit: int=10) -> pd.DataFrame:\n        Lấy top 10 cổ phiếu có tỷ lệ tăng giá cao nhất cho chỉ số cụ thể.\n    \n    loser(index: str='VNINDEX', limit: int=10) -> pd.DataFrame:\n        Lấy top 10 cổ phiếu có tỷ lệ giảm giá cao nhất cho chỉ số cụ thể.\n    \n    trade(index: str='VNINDEX', limit: int=10) -> pd.DataFrame:\n        Lấy top 10 cổ phiếu có giá trị giao dịch lớn nhất cho chỉ số cụ thể.\n    \n    volume(index: str='VNINDEX', limit: int=10) -> pd.DataFrame:\n        Lấy top 10 cổ phiếu có khối lượng đột biến lớn nhất cho chỉ số cụ thể.\n    \n    deal(index: str='VNINDEX', limit: int=10) -> pd.DataFrame:\n        Lấy top 10 cổ phiếu có giao dịch thỏa thuận đột biến lớn nhất cho chỉ số cụ thể.\n\n    top_foreign_buy(trading_date: str, limit: int=10) -> pd.DataFrame:\n        Lấy top 10 cổ phiếu có giá trị mua ròng lớn nhất từ nhà đầu tư nước ngoài.\n\n    top_foreign_trade(trading_date: str, limit: int=10) -> pd.DataFrame:\n        Lấy top 10 cổ phiếu có giá trị giao dịch ròng lớn nhất từ nhà đầu tư nước ngoài.\n    "
	def __init__(A,show_log=False,random_agent=False):
		'\n        Khởi tạo lớp TopStock với tùy chọn sử dụng user-agent ngẫu nhiên và hiển thị log.\n\n        Tham số:\n        -----------\n        show_log : bool, tùy chọn\n            Nếu True, hiển thị thông báo log.\n        random_agent : bool, tùy chọn\n            Nếu True, sử dụng user-agent ngẫu nhiên trong headers HTTP (mặc định là False).\n        ';C='VND';B=show_log;A.show_log=B;A.base_url=_INSIGHT_BASE;A.headers=get_headers(data_source=C,random_agent=random_agent);A.data_source=C
		if not B:logger.setLevel('CRITICAL')
	def _fetch_data(A,url):
		'\n        Phương thức nội bộ để lấy dữ liệu từ URL đã cho và trả về dưới dạng DataFrame của pandas.\n\n        Tham số:\n        -----------\n        url : str\n            URL để gửi yêu cầu GET.\n\n        Trả về:\n        --------\n        pd.DataFrame\n            Một DataFrame chứa dữ liệu lấy từ API, hoặc một DataFrame trống nếu có lỗi xảy ra.\n        '
		try:
			if A.show_log:logger.info(f"Lấy dữ liệu từ URL: {url}")
			B=requests.get(url,headers=A.headers);B.raise_for_status();C=B.json()
			if _C in C:D=pd.DataFrame(C[_C]);D.rename(columns=_TOP_STOCK_COLS,inplace=True);return D
			else:logger.error(_E);return pd.DataFrame()
		except requests.exceptions.RequestException as E:logger.error(f"Lỗi khi lấy dữ liệu: {E}");return pd.DataFrame()
	def _fetch_foreign_data(A,url):
		"\n        Phương thức nội bộ để lấy dữ liệu giao dịch nước ngoài từ URL đã cho và trả về dưới dạng DataFrame.\n\n        Tham số:\n        -----------\n        url : str\n            URL để gửi yêu cầu GET.\n\n        Trả về:\n        --------\n        pd.DataFrame\n            Một DataFrame chứa dữ liệu giao dịch nước ngoài, với các cột được đổi tên thành 'symbol', 'date', và 'net_value'.\n        "
		try:
			if A.show_log:logger.info(f"Lấy dữ liệu từ URL: {url}")
			B=requests.get(url,headers=A.headers);B.raise_for_status();C=B.json()
			if _C in C:D=pd.DataFrame(C[_C]);D.rename(columns={'code':'symbol','tradingDate':'date','netVal':'net_value'},inplace=True);return D
			else:logger.error(_E);return pd.DataFrame()
		except requests.exceptions.RequestException as E:logger.error(f"Lỗi khi lấy dữ liệu: {E}");return pd.DataFrame()
	def _get_index_code(A,index):"\n        Phương thức nội bộ để lấy mã chỉ số từ tên chỉ số.\n\n        Tham số:\n        -----------\n        index : str\n            Tên chỉ số được người dùng cung cấp (ví dụ: 'VNINDEX').\n\n        Trả về:\n        --------\n        str\n            Mã chỉ số tương ứng (ví dụ: 'VNIndex').\n        ";return _TOP_STOCK_INDEX.get(index.upper(),'VNIndex')
	@agg_execution(_A)
	def gainer(self,index=_B,limit=10):"\n        Lấy top 10 cổ phiếu có tỷ lệ tăng giá cao nhất cho chỉ số cụ thể.\n\n        Tham số:\n        -----------\n        index : str, tùy chọn\n            Tên chỉ số (mặc định là 'VNINDEX').\n        limit : int, tùy chọn\n            Số lượng cổ phiếu muốn lấy (mặc định là 10).\n\n        Trả về:\n        --------\n        pd.DataFrame\n            Một DataFrame chứa top cổ phiếu tăng giá.\n        ";A=self;B=A._get_index_code(index);C=f"{A.base_url}/top_stocks?q=index:{B}~nmVolumeAvgCr20D:gte:10000~priceChgPctCr1D:gt:0&size={limit}&sort=priceChgPctCr1D";return A._fetch_data(C)
	@agg_execution(_A)
	def loser(self,index=_B,limit=10):"\n        Lấy top 10 cổ phiếu có tỷ lệ giảm giá cao nhất cho chỉ số cụ thể.\n\n        Tham số:\n        -----------\n        index : str, tùy chọn\n            Tên chỉ số (mặc định là 'VNINDEX').\n        limit : int, tùy chọn\n            Số lượng cổ phiếu muốn lấy (mặc định là 10).\n\n        Trả về:\n        --------\n        pd.DataFrame\n            Một DataFrame chứa top cổ phiếu giảm giá.\n        ";A=self;B=A._get_index_code(index);C=f"{A.base_url}/top_stocks?q=index:{B}~nmVolumeAvgCr20D:gte:10000~priceChgPctCr1D:lt:0&size={limit}&sort=priceChgPctCr1D:asc";return A._fetch_data(C)
	@agg_execution(_A)
	def value(self,index=_B,limit=10):"\n        Lấy top 10 cổ phiếu có giá trị giao dịch lớn nhất cho chỉ số cụ thể.\n\n        Tham số:\n        -----------\n        index : str, tùy chọn\n            Tên chỉ số (mặc định là 'VNINDEX').\n        limit : int, tùy chọn\n            Số lượng cổ phiếu muốn lấy (mặc định là 10).\n\n        Trả về:\n        --------\n        pd.DataFrame\n            Một DataFrame chứa top cổ phiếu có giá trị giao dịch lớn nhất.\n        ";A=self;B=A._get_index_code(index);C=f"{A.base_url}/top_stocks?q=index:{B}~accumulatedVal:gt:0&size={limit}&sort=accumulatedVal";return A._fetch_data(C)
	@agg_execution(_A)
	def volume(self,index=_B,limit=10):"\n        Lấy top 10 cổ phiếu có khối lượng đột biến lớn nhất cho chỉ số cụ thể.\n\n        Tham số:\n        -----------\n        index : str, tùy chọn\n            Tên chỉ số (mặc định là 'VNINDEX').\n        limit : int, tùy chọn\n            Số lượng cổ phiếu muốn lấy (mặc định là 10).\n\n        Trả về:\n        --------\n        pd.DataFrame\n            Một DataFrame chứa top cổ phiếu có khối lượng đột biến lớn nhất.\n        ";A=self;B=A._get_index_code(index);C=f"{A.base_url}/top_stocks?q=index:{B}~nmVolumeAvgCr20D:gte:10000~nmVolNmVolAvg20DPctCr:gte:100&size={limit}&sort=nmVolNmVolAvg20DPctCr";return A._fetch_data(C)
	@agg_execution(_A)
	def deal(self,index=_B,limit=10):"\n        Lấy top 10 cổ phiếu có giao dịch thỏa thuận đột biến lớn nhất cho chỉ số cụ thể.\n\n        Tham số:\n        -----------\n        index : str, tùy chọn\n            Tên chỉ số (mặc định là 'VNINDEX').\n        limit : int, tùy chọn\n            Số lượng cổ phiếu muốn lấy (mặc định là 10).\n\n        Trả về:\n        --------\n        pd.DataFrame\n            Một DataFrame chứa top cổ phiếu có giao dịch thỏa thuận đột biến lớn nhất.\n        ";A=self;B=A._get_index_code(index);C=f"{A.base_url}/top_stocks?size={limit}&q=index:{B}~nmVolumeAvgCr20D:gte:10000&sort=ptVolTotalVolAvg20DPctCr";return A._fetch_data(C)
	@agg_execution(_A)
	def foreign_buy(self,date=_D,limit=10):
		"\n        Lấy top 10 cổ phiếu có giá trị mua ròng lớn nhất từ nhà đầu tư nước ngoài.\n\n        Tham số:\n        -----------\n        date : str\n            Ngày giao dịch dưới dạng 'YYYY-mm-dd'. Ví dụ: '2024-08-16'.\n        limit : int, tùy chọn\n            Số lượng cổ phiếu muốn lấy (mặc định là 10).\n\n        Trả về:\n        --------\n        pd.DataFrame\n            Một DataFrame chứa top cổ phiếu có giá trị mua ròng lớn nhất từ nhà đầu tư nước ngoài.\n        ";A=date
		if A is _D:A=datetime.now().strftime(_F)
		B=f"https://api-finfo.vndirect.com.vn/v4/foreigns?q=type:STOCK,IFC,ETF~netVal:gt:0~tradingDate:{A}&sort=tradingDate~netVal:desc&size={limit}&fields=code,netVal,tradingDate";return self._fetch_foreign_data(B)
	@agg_execution(_A)
	def foreign_sell(self,date=_D,limit=10):
		"\n        Lấy top 10 cổ phiếu có giá trị giao dịch ròng lớn nhất từ nhà đầu tư nước ngoài.\n\n        Tham số:\n        -----------\n        date : str\n            Ngày giao dịch dưới dạng 'YYYY-mm-dd'. Ví dụ: '2024-08-16'.\n        limit : int, tùy chọn\n            Số lượng cổ phiếu muốn lấy (mặc định là 10).\n\n        Trả về:\n        --------\n        pd.DataFrame\n            Một DataFrame chứa top cổ phiếu có giá trị giao dịch ròng lớn nhất từ nhà đầu tư nước ngoài.\n        ";A=date
		if A is _D:A=datetime.now().strftime(_F)
		B=f"https://api-finfo.vndirect.com.vn/v4/foreigns?q=type:STOCK,IFC,ETF~netVal:lt:0~tradingDate:{A}&sort=tradingDate~netVal:asc&size={limit}&fields=code,netVal,tradingDate";return self._fetch_foreign_data(B)
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('insight','vnd',TopStock)