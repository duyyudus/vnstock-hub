'Listing module for KB Securities (KBS) data source.'
_K='FU_INDEX'
_J='BOND'
_I='GET'
_H='name'
_G='data'
_F=None
_E='source'
_D='organ_name'
_C='symbol'
_B='KBS.ext'
_A=False
from typing import Dict,List,Optional,Union
import json,pandas as pd
from app.lib._vnstock_shared.compat import agg_execution
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib.vnstock_data_alt.core.utils.client import send_request,ProxyConfig
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
from app.lib.vnstock_data_alt.explorer.kbs.const import _IIS_BASE_URL,_SEARCH_URL,_SECTOR_ALL_URL,_GROUP_CODE,_INDUSTRY_CODE
from app.lib._vnstock_shared.common import indices as market_indices
logger=get_logger(__name__)
class Listing:
	'\n    Lớp truy cập dữ liệu danh sách mã chứng khoán từ KB Securities (KBS).\n    '
	def __init__(A,random_agent=_A,proxy_config=_F,show_log=_A,proxy_mode=_F,proxy_list=_F):
		'\n        Khởi tạo Listing client cho KBS.\n\n        Args:\n            random_agent: Sử dụng user agent ngẫu nhiên. Mặc định False.\n            proxy_config: Cấu hình proxy. Mặc định None (không sử dụng proxy).\n            show_log: Hiển thị log debug. Mặc định False.\n            proxy_mode: Chế độ proxy (try, rotate, random, single). Mặc định None.\n            proxy_list: Danh sách proxy URLs. Mặc định None.\n        ';E=proxy_mode;D=show_log;C=proxy_config;B=proxy_list;A.data_source='KBS';A.base_url=_IIS_BASE_URL;A.headers=get_headers(data_source=A.data_source,random_agent=random_agent);A.show_log=D
		if C is _F:
			G=E if E else'try';F='direct'
			if B and len(B)>0:F='proxy'
			A.proxy_config=ProxyConfig(proxy_mode=G,proxy_list=B,request_mode=F)
		else:A.proxy_config=C
		if not D:logger.setLevel('CRITICAL')
	@agg_execution(_B)
	def all_symbols(self,show_log=_A):
		"\n        Truy xuất danh sách toàn bộ mã chứng khoán trên thị trường Việt Nam từ KBS.\n\n        Trả về DataFrame đơn giản với mapping symbol → organ_name (tên công ty tiếng Việt).\n\n        Args:\n            show_log: Hiển thị log debug. Mặc định False.\n\n        Returns:\n            DataFrame với 2 cột: symbol, organ_name.\n            Metadata 'source' được lưu trong df.attrs['source'].\n\n        Examples:\n            >>> kbs = Listing()\n            >>> df = kbs.all_symbols()\n            >>> print(df.columns.tolist())\n            ['symbol', 'organ_name']\n            >>> print(df.attrs['source'])\n            'KBS'\n        ";B=show_log
		try:C=self._get_full_stock_data(show_log=B)
		except Exception as D:
			if B:logger.error(f"Lỗi khi lấy dữ liệu chứng khoán: {str(D)}")
			return pd.DataFrame(columns=[_C,_D])
		if not C:return pd.DataFrame(columns=[_C,_D])
		A=pd.DataFrame(C).query("type == 'stock'")
		if _H in A.columns:A=A.rename(columns={_H:_D})
		A=A[[_C,_D]];A.attrs[_E]=self.data_source
		if B:logger.info(f"Truy xuất thành công {len(A)} mã chứng khoán.")
		return A
	@agg_execution(_B)
	def symbols_by_exchange(self,get_all=_A,show_log=_A):
		"\n        Truy xuất danh sách mã chứng khoán theo sàn giao dịch.\n\n        Sử dụng endpoint /stock/search/data để lấy dữ liệu đầy đủ.\n\n        Args:\n            get_all: Lấy tất cả các cột mà API cung cấp thay vì chỉ các cột chuẩn hoá. Mặc định False.\n            show_log: Hiển thị log debug. Mặc định False.\n\n        Returns:\n            DataFrame chứa các cột từ API KBS: symbol, organ_name, en_organ_name,\n            exchange, type, id, re, ceiling, floor.\n            Metadata 'source' được lưu trong df.attrs['source'].\n            Các cột không có sẽ bỏ qua.\n        ";H='type';E='id';D='en_organ_name';C=show_log;B='exchange'
		try:F=self._get_full_stock_data(show_log=C)
		except Exception as I:
			if C:logger.error(f"Lỗi khi lấy dữ liệu chứng khoán: {str(I)}")
			return pd.DataFrame(columns=[_C,_D,B,_E])
		if not F:return pd.DataFrame(columns=[_C,_D,B,_E])
		A=pd.DataFrame(F);J={_H:_D,'nameEn':D,'index':E};A=A.rename(columns=J)
		if get_all:K=[_C,_D,D,B,H,E,'re','ceiling','floor'];G=[B for B in K if B in A.columns]
		else:G=[_C,_D,D,B,H,E]
		A=A[G];A.attrs[_E]=self.data_source
		if C:logger.info(f"Truy xuất thành công {len(A)} mã chứng khoán theo sàn.")
		return A
	@agg_execution(_B)
	def symbols_by_industries(self,lang='vi',show_log=_A):
		"\n        Truy xuất danh sách mã chứng khoán theo nhóm ngành. Thông tin mã ngành là quy định riêng của KBS không theo chuẩn ICB thường gặp.\n\n        Args:\n            lang: Ngôn ngữ ('vi' hoặc 'en'). Mặc định 'vi'.\n            show_log: Hiển thị log debug. Mặc định False.\n\n        Returns:\n            DataFrame chứa thông tin mã chứng khoán theo ngành.\n\n        Raises:\n            ValueError: Nếu ngôn ngữ không hợp lệ.\n        ";H='industry_name';G='industry_code';C=show_log;B=self
		if lang not in['vi','en']:raise ValueError("Ngôn ngữ phải là 'vi' hoặc 'en'.")
		try:K=B._get_industries_internal(show_log=C)
		except Exception as D:
			if C:logger.error(f"Lỗi khi lấy danh sách ngành: {str(D)}")
			A=pd.DataFrame(columns=[_C,G,H]);A.attrs[_E]=B.data_source;return A
		E=[]
		for I in K:
			F=I['code'];J=I[_H]
			try:
				L=B._get_symbols_by_industry_internal(industry_code=F,show_log=C)
				for M in L:E.append({_C:M,G:F,H:J})
			except Exception as D:
				if C:logger.warning(f"Lỗi khi lấy mã từ ngành {J} ({F}): {str(D)}")
		if E:A=pd.DataFrame(E);A.attrs[_E]=B.data_source;return A
		else:A=pd.DataFrame(columns=[_C,G,H]);A.attrs[_E]=B.data_source;return A
	@agg_execution(_B)
	def symbols_by_group(self,group='VN30',show_log=_A):
		"\n        Truy xuất danh sách mã chứng khoán theo nhóm chỉ số.\n\n        Hỗ trợ lọc theo các nhóm/sàn: chỉ số VN (VN30, VN100, VNMidCap, VNSmallCap, VNSI, VNX50, VNXALL),\n        sàn giao dịch (HOSE, HNX, UPCOM), chỉ số HNX30, ETF, chứng quyền (CW), trái phiếu (BOND),\n        và phái sinh (DER).\n\n        Để xem danh sách tất cả các nhóm được hỗ trợ, gọi `get_supported_groups()`.\n\n        Args:\n            group: Tên nhóm được hỗ trợ. Mặc định 'VN30'.\n                   Ví dụ: 'VN30', 'VN100', 'HOSE', 'HNX', 'UPCOM', 'ETF', 'BOND', 'CW', 'FU_INDEX'.\n            show_log: Hiển thị log debug. Mặc định False.\n\n        Returns:\n            Series chứa mã chứng khoán theo nhóm.\n\n        Raises:\n            ValueError: Nếu tên nhóm không hợp lệ.\n\n        Example:\n            >>> from app.lib.vnstock_data_alt.explorer.kbs import Listing\n            >>> kbs = Listing()\n            >>> # Lấy danh sách VN30\n            >>> vn30 = kbs.symbols_by_group('VN30')\n            >>> # Lấy tất cả ETF\n            >>> etf_symbols = kbs.symbols_by_group('ETF')\n            >>> # Xem tất cả nhóm được hỗ trợ\n            >>> groups = kbs.get_supported_groups()\n        ";A=group
		if A not in _GROUP_CODE:raise ValueError(f"Nhóm không hợp lệ. Sử dụng get_supported_groups() để xem danh sách nhóm được hỗ trợ.")
		C=self._get_symbols_by_group_internal(group=A,show_log=show_log);B=pd.Series(C,name=_C);B.attrs[_E]=self.data_source;B.attrs['group']=A;return B
	@agg_execution(_B)
	def industries_icb(self,show_log=_A):'\n        Truy xuất thông tin danh sách các ngành công nghiệp.\n\n        Note: **KBS không cung cấp ICB classification.**\n        \n        Để lấy danh sách mã theo ngành, hãy sử dụng `symbols_by_industries()`.\n\n        Raises:\n            NotImplementedError: KBS không hỗ trợ ICB classification.\n        ';raise NotImplementedError('KBS không cung cấp ICB classification. Sử dụng symbols_by_industries() để lấy mã theo ngành.')
	@agg_execution(_B)
	def get_supported_groups(self):"\n        Liệt kê tất cả các nhóm/sàn được hỗ trợ bởi phương thức symbols_by_group().\n\n        Các mô tả chỉ số tuân theo chuẩn của vnstock library.\n\n        Returns:\n            DataFrame với các cột:\n            - group_name: Tên nhóm có thể truyền vào symbols_by_group()\n            - group_code: Mã nội bộ của KBS\n            - category: Danh mục (Chỉ số VN, Sàn giao dịch, ETF/Quỹ, Chứng quyền, Trái phiếu, Phái sinh)\n            - description: Mô tả chi tiết theo chuẩn vnstock\n\n        Example:\n            >>> from app.lib.vnstock_data_alt.explorer.kbs import Listing\n            >>> kbs = Listing()\n            >>> groups = kbs.get_supported_groups()\n            >>> print(groups)\n            >>> # Lọc chỉ các chỉ số VN\n            >>> vn_indices = groups[groups['category'] == 'Chỉ số VN']\n        ";G='UPCOM';F='HNX';E='HOSE';D='HNX30';B='Sàn giao dịch';A='Chỉ số VN';H={('VN30','30',A,'30 cổ phiếu vốn hóa lớn nhất & thanh khoản tốt nhất HOSE'),('VN100','100',A,'100 cổ phiếu có vốn hoá lớn nhất HOSE'),('VNMidCap','MID',A,'Mid-Cap Index - nhóm cổ phiếu vốn hóa trung bình'),('VNSmallCap','SML',A,'Small-Cap Index - nhóm cổ phiếu vốn hóa nhỏ'),('VNSI','SI',A,'Vietnam Small-Cap Index'),('VNX50','X50',A,'50 cổ phiếu vốn hóa lớn nhất trên toàn bộ thị trường HOSE và HNX'),('VNXALL','XALL',A,'Tất cả cổ phiếu trên toàn bộ thị trường HOSE và HNX'),('VNALL','ALL',A,'Tất cả cổ phiếu trên HOSE và HNX'),(D,D,A,'Chỉ số 30 cổ phiếu hàng đầu HNX'),(E,E,B,'Sở giao dịch chứng khoán TP. Hồ Chí Minh'),(F,F,B,'Sàn Giao dịch Chứng khoán Hà Nội'),(G,G,B,'Sàn Giao dịch OTC (UPCoM - Unlisted Public Company Market)'),('ETF','FUND','ETF/Quỹ','Exchange-Traded Fund - Quỹ chỉ số và quỹ trao đổi'),('CW','CW','Chứng quyền','Covered Warrant - Chứng quyền phát hành bởi các tổ chức tài chính'),(_J,_J,'Trái phiếu','Corporate Bond - Trái phiếu doanh nghiệp niêm yết'),(_K,'DER','Phái sinh','Futures - Hợp đồng tương lai chỉ số')};I=[{'group_name':A,'group_code':B,'category':C,'description':D}for(A,B,C,D)in sorted(H)];C=pd.DataFrame(I);C.attrs[_E]=self.data_source;return C
	@agg_execution(_B)
	def all_future_indices(self,show_log=_A):'\n        Truy xuất danh sách mã phái sinh hợp đồng tương lai.\n\n        Args:\n            show_log: Hiển thị log debug. Mặc định False.\n\n        Returns:\n            Series chứa mã phái sinh.\n        ';return self.symbols_by_group(group=_K,show_log=show_log)
	@agg_execution(_B)
	def all_covered_warrant(self,show_log=_A):'\n        Truy xuất danh sách mã chứng quyền.\n\n        Args:\n            show_log: Hiển thị log debug. Mặc định False.\n\n        Returns:\n            Series chứa mã chứng quyền.\n        ';return self.symbols_by_group(group='CW',show_log=show_log)
	@agg_execution(_B)
	def all_bonds(self,show_log=_A):'\n        Truy xuất danh sách mã trái phiếu.\n\n        Args:\n            show_log: Hiển thị log debug. Mặc định False.\n\n        Returns:\n            Series chứa mã trái phiếu.\n        ';return self.symbols_by_group(group=_J,show_log=show_log)
	@agg_execution(_B)
	def all_etf(self,show_log=_A):'\n        Truy xuất danh sách mã quỹ ETF.\n\n        Args:\n            show_log: Hiển thị log debug. Mặc định False.\n\n        Returns:\n            Series chứa mã ETF.\n        ';return self.symbols_by_group(group='ETF',show_log=show_log)
	@agg_execution(_B)
	def all_government_bonds(self,show_log=_A):'\n        Truy xuất danh sách mã trái phiếu chính phủ.\n\n        Note: **KBS không cung cấp dữ liệu trái phiếu chính phủ.**\n\n        Để lấy danh sách trái phiếu doanh nghiệp, hãy sử dụng `all_bonds()`.\n\n        Raises:\n            NotImplementedError: KBS không hỗ trợ trái phiếu chính phủ.\n        ';raise NotImplementedError('KBS không cung cấp dữ liệu trái phiếu chính phủ. Sử dụng all_bonds() để lấy trái phiếu doanh nghiệp.')
	def _get_full_stock_data(B,show_log=_A):
		'\n        Internal method để lấy dữ liệu đầy đủ về tất cả chứng khoán từ /stock/search/data endpoint.\n\n        Trả về danh sách chứng khoán với tất cả thông tin: symbol, name, nameEn, exchange, \n        type, index, re, ceiling, floor.\n\n        Args:\n            show_log: Hiển thị log debug. Mặc định False.\n\n        Returns:\n            List[Dict] chứa thông tin đầy đủ của tất cả chứng khoán, hoặc [] nếu lỗi.\n        ';D=show_log;E=_SEARCH_URL
		try:
			A=send_request(url=E,headers=B.headers,method=_I,payload=_F,show_log=D,proxy_list=B.proxy_config.proxy_list,proxy_mode=B.proxy_config.proxy_mode,request_mode=B.proxy_config.request_mode)
			if not A:raise ValueError('Không tìm thấy dữ liệu chứng khoán.')
			if isinstance(A,list):C=A
			elif isinstance(A,dict)and _G in A:C=A[_G]
			else:C=[]
			if D:logger.info(f"Truy xuất thành công {len(C)} chứng khoán.")
			return C
		except Exception as F:
			if D:logger.error(f"Lỗi khi lấy dữ liệu chứng khoán: {str(F)}")
			return[]
	def _get_symbols_by_group_internal(C,group,show_log=_A):
		'\n        Internal method để lấy danh sách mã theo nhóm/sàn.\n\n        Args:\n            group: Tên nhóm hoặc sàn.\n            show_log: Hiển thị log debug.\n\n        Returns:\n            List[str] chứa danh sách mã.\n        ';E=show_log;B=group;F=_GROUP_CODE.get(B,B);G=f"{_IIS_BASE_URL}/index/{F}/stocks"
		try:
			A=send_request(url=G,headers=C.headers,method=_I,payload=_F,show_log=E,proxy_list=C.proxy_config.proxy_list,proxy_mode=C.proxy_config.proxy_mode,request_mode=C.proxy_config.request_mode)
			if not A:raise ValueError(f"Không tìm thấy dữ liệu cho nhóm {B}.")
			if isinstance(A,dict)and _G in A:D=A[_G]
			elif isinstance(A,list):D=A
			else:D=[]
			if E:logger.info(f"Truy xuất thành công {len(D)} mã từ nhóm {B}.")
			return D
		except Exception as H:
			if E:logger.error(f"Lỗi khi lấy dữ liệu từ nhóm {B}: {str(H)}")
			raise
	def _get_industries_internal(B,show_log=_A):
		'\n        Internal method để lấy danh sách các ngành.\n\n        Args:\n            show_log: Hiển thị log debug.\n\n        Returns:\n            List[Dict] chứa thông tin ngành.\n        ';D=show_log;E=f"{_SECTOR_ALL_URL}"
		try:
			A=send_request(url=E,headers=B.headers,method=_I,payload=_F,show_log=D,proxy_list=B.proxy_config.proxy_list,proxy_mode=B.proxy_config.proxy_mode,request_mode=B.proxy_config.request_mode)
			if not A:raise ValueError('Không tìm thấy dữ liệu ngành.')
			if isinstance(A,list):C=A
			elif isinstance(A,dict)and _G in A:C=A[_G]
			else:C=[]
			if D:logger.info(f"Truy xuất thành công {len(C)} ngành.")
			return C
		except Exception as F:
			if D:logger.error(f"Lỗi khi lấy danh sách ngành: {str(F)}")
			return[]
	def _get_symbols_by_industry_internal(B,industry_code,show_log=_A):
		'\n        Internal method để lấy danh sách mã theo mã ngành.\n\n        Args:\n            industry_code: Mã ngành.\n            show_log: Hiển thị log debug.\n\n        Returns:\n            List[str] chứa danh sách mã.\n        ';E=show_log;D=industry_code;F=f"{_SECTOR_ALL_URL}?code={D}&l=1"
		try:
			A=send_request(url=F,headers=B.headers,method=_I,payload=_F,show_log=E,proxy_list=B.proxy_config.proxy_list,proxy_mode=B.proxy_config.proxy_mode,request_mode=B.proxy_config.request_mode)
			if not A:return[]
			if isinstance(A,dict)and _G in A:C=A[_G]
			elif isinstance(A,list):C=A
			else:C=[]
			if E:logger.info(f"Truy xuất thành công {len(C)} mã từ ngành {D}.")
			return C
		except Exception as G:
			if E:logger.error(f"Lỗi khi lấy mã từ ngành {D}: {str(G)}")
			return[]
	@agg_execution(_B)
	def all_indices(self,show_log=_A):'\n        Lấy danh sách tất cả các chỉ số tiêu chuẩn hóa với thông tin\n        đầy đủ từ dữ liệu HOSE.\n\n        Returns:\n            DataFrame: Columns [symbol, name, description, full_name,\n                                group, index_id, sector_id (for sectors)]\n        ';return market_indices.get_all_indices()
	@agg_execution(_B)
	def indices_by_group(self,group,show_log=_A):"\n        Lấy danh sách chỉ số theo nhóm tiêu chuẩn hóa từ dữ liệu HOSE.\n\n        Args:\n            group: Tên nhóm (VD: 'HOSE Indices', 'Sector Indices', etc.)\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame: Danh sách chỉ số trong nhóm hoặc None\n                       (Sector indices include sector_id mapping)\n        ";return market_indices.get_indices_by_group(group)
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('listing','kbs',Listing)