_K='Failed to drop change column'
_J='change'
_I=False
_H='CafeF'
_G='TotalCount'
_F='GET'
_E='%Y-%m-%d'
_D=True
_C='Data'
_B='%m/%d/%Y'
_A=None
import requests,pandas as pd
from typing import Union,Optional,Dict
from app.lib.vnstock_data_alt.explorer.cafef.const import _BASE_URL,_PRICE_HISTORY_MAP,_FOREIGN_TRADE_MAP,_PROP_TRADE_MAP,_ORDER_STATS_MAP,_INSIDER_DEAL_MAP
from app.lib.vnstock_data_alt.core.utils.parser import days_between
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
from app.lib._vnstock_shared.core.utils.logger import get_logger
logger=get_logger(__name__)
class Trading:
	def __init__(A,symbol,random_agent=_I):A.symbol=symbol.upper();A.base_url=_BASE_URL;A.headers=get_headers(data_source='CAFEF',random_agent=random_agent)
	def _df_standardized(E,history_data,mapping_dict):
		'\n        Định dạng lại dữ liệu lịch sử giá của mã chứng khoán\n\n        Tham số:\n            - history_data (DataFrame): Dữ liệu lịch sử giá của mã chứng khoán.\n        Return:\n            - DataFrame: Dữ liệu lịch sử giá của mã chứng khoán.\n        ';C=mapping_dict;B='time';A=history_data;A=A.rename(columns=C);D=C.values()
		if A.empty:logger.info('No data found');return
		else:
			try:A=A[D]
			except:logger.debug(f"Failed to sort column names. Actual columns and predefine mapping dict mismatched. Actual columns: {A.columns}. Expected columns: {D}");pass
			if B in A.columns:A[B]=pd.to_datetime(A[B],format='%d/%m/%Y');A.set_index(B,inplace=_D)
			return A
	def price_history(B,start,end,page=1,limit=_A):
		"\n        Truy xuất dữ liệu lịch sử giá của mã chứng khoán bất kỳ từ nguồn dữ liệu CafeF\n\n        Tham số:\n            - start (bắt buộc): Ngày bắt đầu lấy dữ liệu, định dạng YYYY-mm-dd. Ví dụ '2024-06-01'.\n            - end (bắt buộc): Ngày kết thúc lấy dữ liệu, định dạng YYYY-mm-dd. Ví dụ '2024-07-31'.\n            - page (tùy chọn): Trang hiện tại. Mặc định là 1.\n            - limit (tùy chọn): Số lượng dữ liệu trả về trong một lần request. Mặc định là None để đặt giới hạn tương ứng số ngày giữa khung thời gian start và end.\n        Return:\n            - DataFrame: Dữ liệu lịch sử giá của mã chứng khoán.\n        ";H='ThayDoi';F=limit;D=end;C=start
		if F is _A:F=days_between(start=C,end=D,format=_E)
		C=pd.to_datetime(C).strftime(_B);D=pd.to_datetime(D).strftime(_B);I=f"{B.base_url}/PriceHistory.ashx?Symbol={B.symbol}&StartDate={C}&EndDate={D}&PageIndex={page}&PageSize={F}";E=requests.request(_F,I,headers=B.headers)
		if E.status_code!=200:raise ConnectionError(f"Tải dữ liệu không thành công: {E.status_code} - {E.text}")
		G=E.json()[_C];J=G[_G];logger.info(f"Lịch sử giá:\nMã CK: {B.symbol}. Số bản ghi hợp lệ: {J}");A=pd.DataFrame(G[_C])
		try:
			if H in A.columns:A['change_pct']=pd.to_numeric(A[H].str.split('(',expand=_D)[1].str.replace(' %)','',regex=_I).str.replace(')','',regex=_I).str.strip(),errors='coerce')/100
		except:logger.debug('Failed to extract change_pct from ThayDoi column');pass
		try:A=B._df_standardized(A,_PRICE_HISTORY_MAP)
		except Exception as K:logger.error(f"Failed to standardize data: {K}")
		try:A.drop(columns=[_J],inplace=_D)
		except:logger.debug(_K);pass
		A.name=B.symbol;A.category='history_price';A.source=_H;return A
	def foreign_trade(B,start,end,page=1,limit=_A):
		"\n        Truy xuất dữ liệu lịch sử giao dịch nhà đầu tư nước ngoài của mã chứng khoán bất kỳ từ nguồn dữ liệu CafeF\n\n        Tham số:\n            - start (bắt buộc): Ngày bắt đầu lấy dữ liệu, định dạng YYYY-mm-dd. Ví dụ '2024-06-01'.\n            - end (bắt buộc): Ngày kết thúc lấy dữ liệu, định dạng YYYY-mm-dd. Ví dụ '2024-07-31'.\n            - page (tùy chọn): Trang hiện tại. Mặc định là 1.\n            - limit (tùy chọn): Số lượng dữ liệu trả về trong một lần request. Mặc định là None để đặt giới hạn tương ứng số ngày giữa khung thời gian start và end.\n        Return:\n            - DataFrame: Dữ liệu lịch sử giá của mã chứng khoán.\n        ";F=limit;D=end;C=start
		if F is _A:F=days_between(start=C,end=D,format=_E)
		C=pd.to_datetime(C).strftime(_B);D=pd.to_datetime(D).strftime(_B);H=f"{B.base_url}/GDKhoiNgoai.ashx?Symbol={B.symbol}&StartDate={C}&EndDate={D}&PageIndex={page}&PageSize={F}";E=requests.request(_F,H,headers=B.headers)
		if E.status_code!=200:raise ConnectionError(f"Tải dữ liệu không thành công: {E.status_code} - {E.text}")
		G=E.json()[_C];I=G[_G];logger.info(f"Lịch sử GD Nước ngoài:\nMã CK: {B.symbol}. Số bản ghi hợp lệ: {I}");A=pd.DataFrame(G[_C]);A=B._df_standardized(A,_FOREIGN_TRADE_MAP)
		try:A.drop(columns=[_J],inplace=_D)
		except:logger.debug(_K);pass
		A.name=B.symbol;A.category='foreign_trade';A.source=_H;return A
	def prop_trade(B,start,end,page=1,limit=_A):
		'\n        Truy xuất dữ liệu lịch sử giao dịch tự doanh của mã chứng khoán bất kỳ từ nguồn dữ liệu CafeF\n\n        Tham số:\n            - start (bắt buộc): Ngày bắt đầu lấy dữ liệu, định dạng YYYY-mm-dd.\n            - end (bắt buộc): Ngày kết thúc lấy dữ liệu, định dạng YYYY-mm-dd.\n            - page (tùy chọn): Trang hiện tại. Mặc định là\n            - limit (tùy chọn): Số lượng dữ liệu trả về trong một lần request. Mặc định là None để đặt giới hạn tương ứng số ngày giữa khung thời gian start và end.\n        Return:\n            - DataFrame: Dữ liệu lịch sử giá của mã chứng khoán.\n        ';F=limit;D=end;C=start
		if F is _A:F=days_between(start=C,end=D,format=_E)
		C=pd.to_datetime(C).strftime(_B);D=pd.to_datetime(D).strftime(_B);H=f"{B.base_url}/GDTuDoanh.ashx?Symbol={B.symbol}&StartDate={C}&EndDate={D}&PageIndex={page}&PageSize={F}";E=requests.request(_F,H,headers=B.headers)
		if E.status_code!=200:raise ConnectionError(f"Tải dữ liệu không thành công: {E.status_code} - {E.text}")
		G=E.json()[_C];I=G[_G];logger.info(f"Lịch sử GD Tự Doanh:\nMã CK: {B.symbol}. Số bản ghi hợp lệ: {I}");A=pd.DataFrame(G[_C]['ListDataTudoanh']);A=B._df_standardized(A,_PROP_TRADE_MAP)
		try:A.drop(columns='symbol',inplace=_D)
		except:logger.debug('Failed to drop symbol column');pass
		A.name=B.symbol;A.category='prop_trade';A.source=_H;return A
	def order_stats(B,start,end,page=1,limit=_A):
		'\n        Truy xuất dữ liệu lịch sử thống kê đặt lệnh của mã chứng khoán bất kỳ từ nguồn dữ liệu CafeF\n\n        Tham số:\n            - start (bắt buộc): Ngày bắt đầu lấy dữ liệu, định dạng YYYY-mm-dd.\n            - end (bắt buộc): Ngày kết thúc lấy dữ liệu, định dạng YYYY-mm-dd.\n            - page (tùy chọn): Trang hiện tại. Mặc định là\n            - limit (tùy chọn): Số lượng dữ liệu trả về trong một lần request. Mặc định là None để đặt giới hạn tương ứng số ngày giữa khung thời gian start và end.\n        Return:\n            - DataFrame: Dữ liệu lịch sử giá của mã chứng khoán.\n        ';F=limit;D=end;C=start
		if F is _A:F=days_between(start=C,end=D,format=_E)
		C=pd.to_datetime(C).strftime(_B);D=pd.to_datetime(D).strftime(_B);H=f"{B.base_url}/ThongKeDL.ashx?Symbol={B.symbol}&StartDate={C}&EndDate={D}&PageIndex={page}&PageSize={F}";E=requests.request(_F,H,headers=B.headers)
		if E.status_code!=200:raise ConnectionError(f"Tải dữ liệu không thành công: {E.status_code} - {E.text}")
		G=E.json()[_C];I=G[_G];logger.info(f"Thống kê đặt lệnh:\nMã CK: {B.symbol}. Số bản ghi hợp lệ: {I}");A=pd.DataFrame(G[_C]);A=B._df_standardized(A,_ORDER_STATS_MAP)
		try:A.drop(columns=[_J],inplace=_D)
		except:logger.debug(_K);pass
		A.name=B.symbol;A.category='order_stats';A.source=_H;return A
	def insider_deal(B,start,end,page=1,limit=_A):
		'\n        Truy xuất dữ liệu lịch sử thống kê giao dịch của cổ đông & nội bộ cho mã chứng khoán bất kỳ từ nguồn dữ liệu CafeF\n\n        Tham số:\n            - start (bắt buộc): Ngày bắt đầu lấy dữ liệu, định dạng YYYY-mm-dd.\n            - end (bắt buộc): Ngày kết thúc lấy dữ liệu, định dạng YYYY-mm-dd.\n            - page (tùy chọn): Trang hiện tại. Mặc định là\n            - limit (tùy chọn): Số lượng dữ liệu trả về trong một lần request. Mặc định là None để đặt giới hạn tương ứng số ngày giữa khung thời gian start và end.\n        Return:\n            - DataFrame: Dữ liệu lịch sử giá của mã chứng khoán.\n        ';G=limit;D=end;C=start
		if G is _A:G=days_between(start=C,end=D,format=_E)
		C=pd.to_datetime(C).strftime(_B);D=pd.to_datetime(D).strftime(_B);I=f"{B.base_url}/GDCoDong.ashx?Symbol={B.symbol}&StartDate={C}&EndDate={D}&PageIndex={page}&PageSize={G}";E=requests.request(_F,I,headers=B.headers)
		if E.status_code!=200:raise ConnectionError(f"Tải dữ liệu không thành công: {E.status_code} - {E.text}")
		H=E.json()[_C];J=H[_G];logger.info(f"Thống kê giao dịch Cổ Đông & Nội bộ:\nMã CK: {B.symbol}. Số bản ghi hợp lệ: {J}");A=pd.DataFrame(H[_C]);A=B._df_standardized(A,_INSIDER_DEAL_MAP)
		for F in['plan_begin_date','plan_end_date','real_end_date','published_date','order_date']:A[F]=A[F].str.replace('\\D','',regex=_D);A[F]=pd.to_datetime(pd.to_numeric(A[F]),unit='ms')
		A.name=B.symbol;A.category='insider_deals';A.source=_H;return A
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('trading','cafef',Trading)