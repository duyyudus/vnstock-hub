'Listing module.'
_N='__typename'
_M="Tham số lang phải là 'vi' hoặc 'en'."
_L='Không tìm thấy dữ liệu. Vui lòng kiểm tra lại.'
_K='icb_name'
_J=None
_I='icb_code'
_H='records'
_G='organ_name'
_F='VCI'
_E='symbol'
_D='vi'
_C='VCI.ext'
_B=False
_A=True
from typing import Dict,Optional
from datetime import datetime
from app.lib.vnstock_alt.explorer.vci.const import _GROUP_CODE,_TRADING_URL,_GRAPHQL_URL
import json,pandas as pd
from app.lib._vnstock_shared.core.utils.parser import camel_to_snake
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib._vnstock_shared.core.utils.transform import drop_cols_by_pattern,reorder_cols
from app.lib.vnstock_data_alt.core.utils.client import send_request
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
from app.lib._vnstock_shared.compat import agg_execution
logger=get_logger(__name__)
class Listing:
	'\n    Cấu hình truy cập dữ liệu lịch sử giá chứng khoán từ VCI.\n    '
	def __init__(A,random_agent=_B,show_log=_B):
		B=show_log;A.data_source=_F;A.base_url=_TRADING_URL;A.headers=get_headers(data_source=A.data_source,random_agent=random_agent);A.show_log=B
		if not B:logger.setLevel('CRITICAL')
	@agg_execution(_C)
	def all_symbols(self,show_log=_B,to_df=_A):
		'\n        Truy xuất danh sách toàn. bộ mã và tên các cổ phiếu trên thị trường Việt Nam.\n\n        Tham số:\n            - show_log (tùy chọn): Hiển thị thông tin log giúp debug dễ dàng. Mặc định là False.\n            - to_df (tùy chọn): Chuyển đổi dữ liệu danh sách mã cổ phiếu trả về dưới dạng DataFrame. Mặc định là True. Đặt là False để trả về dữ liệu dạng JSON.\n        ';A=self.symbols_by_exchange(show_log=show_log,to_df=_A);A=A.query('type == "STOCK"').reset_index(drop=_A);A=A[[_E,_G]]
		if to_df:return A
		else:B=A.to_json(orient=_H);return B
	@agg_execution(_C)
	def symbols_by_industries(self,lang=_D,show_log=_B,to_df=_A):
		'\n        Truy xuất thông tin phân ngành icb của các mã cổ phiếu trên thị trường Việt Nam.\n\n        Tham số:\n            - show_log (tùy chọn): Hiển thị thông tin log giúp debug dễ dàng. Mặc định là False.\n            - to_df (tùy chọn): Chuyển đổi dữ liệu danh sách mã cổ phiếu trả về dưới dạng DataFrame. Mặc định là True. Đặt là False để trả về dữ liệu dạng JSON.\n        ';M='com_type_code';K=show_log;J='icb_level';D=lang
		if D not in[_D,'en']:raise ValueError(_M)
		F='{"query":"{\\n  CompaniesListingInfo {\\n    ticker\\n    organName\\n    enOrganName\\n    icbName3\\n    enIcbName3\\n    icbName2\\n    enIcbName2\\n    icbName4\\n    enIcbName4\\n    comTypeCode\\n    icbCode1\\n    icbCode2\\n    icbCode3\\n    icbCode4\\n    __typename\\n  }\\n}\\n","variables":{}}';F=json.loads(F);E=send_request(url=_GRAPHQL_URL,headers=self.headers,method='POST',payload=F,show_log=K)
		if not E:raise ValueError(_L)
		if K:logger.info(f"Truy xuất thành công dữ liệu danh sách phân ngành icb.")
		A=pd.DataFrame(E['data']['CompaniesListingInfo']);A.columns=[camel_to_snake(A)for A in A.columns];A=A.drop(columns=[_N]);A=A.rename(columns={'ticker':_E});L=_G if D==_D else'en_organ_name';N='icb_name2'if D==_D else'en_icb_name2';O='icb_name3'if D==_D else'en_icb_name3';P='icb_name4'if D==_D else'en_icb_name4';Q=[_E,L,M];G=[]
		for(R,H,I)in[(1,'icb_code1',_J),(2,'icb_code2',N),(3,'icb_code3',O),(4,'icb_code4',P)]:
			if H in A.columns:
				C=A[Q+[H]].copy();C[J]=R;C=C.rename(columns={H:_I})
				if I and I in A.columns:C[_K]=A[I]
				else:C[_K]=_J
				G.append(C)
		if G:B=pd.concat(G,ignore_index=_A);B=B[B[_I].notna()&(B[_I]!='')];B=B.rename(columns={L:_G});B=B.sort_values(by=[_E,J]).reset_index(drop=_A);A=B[[_E,_G,M,J,_I,_K]]
		A.source=_F
		if to_df:return A
		else:E=A.to_json(orient=_H);return E
	@agg_execution(_C)
	def symbols_by_exchange(self,lang=_D,show_log=_B,to_df=_A):
		'\n        Truy xuất thông tin niêm yết theo sàn của các mã cổ phiếu trên thị trường Việt Nam.\n\n        Tham số:\n                - show_log (tùy chọn): Hiển thị thông tin log giúp debug dễ dàng. Mặc định là False.\n                - to_df (tùy chọn): Chuyển đổi dữ liệu danh sách mã cổ phiếu trả về dưới dạng DataFrame. Mặc định là True. Đặt là False để trả về dữ liệu dạng JSON.\n        ';E='en_';D='exchange';C=show_log
		if lang not in[_D,'en']:raise ValueError(_M)
		F=self.base_url+'/price/symbols/getAll';B=send_request(url=F,headers=self.headers,method='GET',payload=_J,show_log=C)
		if not B:raise ValueError(_L)
		if C:logger.info(f"Truy xuất dữ liệu thành công cho {len(B)} mã.")
		A=pd.DataFrame(B);A.columns=[camel_to_snake(A)for A in A.columns];A=A.rename(columns={'board':D});A=reorder_cols(A,[_E,D,'type'],position='first');A=A.drop(columns=['id'])
		if lang==_D:A=drop_cols_by_pattern(A,[E])
		else:A=A.drop(columns=[_G,'organ_short_name']);A.columns=[A.replace(E,'')for A in A.columns]
		if to_df:A.source=_F;return A
		else:B=A.to_json(orient=_H);return B
	@agg_execution(_C)
	def industries_icb(self,show_log=_B,to_df=_A):
		'\n        Truy xuất thông tin phân ngành icb của các mã cổ phiếu trên thị trường Việt Nam.\n\n        Tham số:\n            - show_log (tùy chọn): Hiển thị thông tin log giúp debug dễ dàng. Mặc định là False.\n            - to_df (tùy chọn): Chuyển đổi dữ liệu danh sách mã cổ phiếu trả về dưới dạng DataFrame. Mặc định là True. Đặt là False để trả về dữ liệu dạng JSON.\n        ';D=show_log;C='{"query":"query Query {\\n  ListIcbCode {\\n    icbCode\\n    level\\n    icbName\\n    enIcbName\\n    __typename\\n  }\\n  CompaniesListingInfo {\\n    ticker\\n    icbCode1\\n    icbCode2\\n    icbCode3\\n    icbCode4\\n    __typename\\n  }\\n}","variables":{}}';C=json.loads(C);B=send_request(url=_GRAPHQL_URL,headers=self.headers,method='POST',payload=C,show_log=D)
		if not B:raise ValueError(_L)
		if D:logger.info(f"Truy xuất thành công dữ liệu danh sách phân ngành icb.")
		A=pd.DataFrame(B['data']['ListIcbCode']);A.columns=[camel_to_snake(A)for A in A.columns];A=A.drop(columns=[_N]);A=A[[_K,'en_icb_name',_I,'level']];A.source=_F
		if to_df:return A
		else:B=A.to_json(orient=_H);return B
	@agg_execution(_C)
	def symbols_by_group(self,group='VN30',show_log=_B,to_df=_A):
		"\n        Truy xuất danh sách các mã cổ phiếu theo tên nhóm trên thị trường Việt Nam.\n\n        Tham số:\n            - group (tùy chọn): Tên nhóm cổ phiếu. Mặc định là 'VN30'. Các mã có thể là: HOSE, VN30, VNMidCap, VNSmallCap, VNAllShare, VN100, ETF, HNX, HNX30, HNXCon, HNXFin, HNXLCap, HNXMSCap, HNXMan, UPCOM, FU_INDEX (mã chỉ số hợp đồng tương lai), CW (chứng quyền).\n            - show_log (tùy chọn): Hiển thị thông tin log giúp debug dễ dàng. Mặc định là False.\n            - to_df (tùy chọn): Chuyển đổi dữ liệu danh sách mã cổ phiếu trả về dưới dạng DataFrame. Mặc định là True. Đặt là False để trả về dữ liệu dạng JSON.\n        ";D=show_log;C=group
		if C not in _GROUP_CODE:raise ValueError(f"Invalid group. Group must be in {_GROUP_CODE}")
		E=self.base_url+f"/price/symbols/getByGroup?group={C}";A=send_request(url=E,headers=self.headers,method='GET',payload=_J,show_log=D)
		if D:logger.info(f"Truy xuất thành công dữ liệu danh sách mã CP theo nhóm.")
		B=pd.DataFrame(A)
		if to_df:
			if not A:raise ValueError('JSON data is empty or not provided.')
			B.source=_F;return B[_E]
		else:A=B.to_json(orient=_H);return A
	@agg_execution(_C)
	def all_future_indices(self,show_log=_B,to_df=_A):return self.symbols_by_group(group='FU_INDEX',show_log=show_log,to_df=to_df)
	@agg_execution(_C)
	def all_government_bonds(self,show_log=_B,to_df=_A):return self.symbols_by_group(group='FU_BOND',show_log=show_log,to_df=to_df)
	@agg_execution(_C)
	def all_covered_warrant(self,show_log=_B,to_df=_A):return self.symbols_by_group(group='CW',show_log=show_log,to_df=to_df)
	@agg_execution(_C)
	def all_bonds(self,show_log=_B,to_df=_A):return self.symbols_by_group(group='BOND',show_log=show_log,to_df=to_df)
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('listing','vci',Listing)