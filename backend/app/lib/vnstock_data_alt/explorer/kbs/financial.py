'Financial module for KB Securities (KBS) data source.'
_X='ignore'
_W='LCTT'
_V='financial_ratios'
_U='cash_flow'
_T='balance_sheet'
_S='income_statement'
_R='KBS.ext'
_Q='period'
_P='report_type'
_O='source'
_N='symbol'
_M='Content'
_L='row_number'
_K='levels'
_J='unit'
_I='item_en'
_H='item'
_G='year'
_F='unit_type'
_E='audit_status'
_D=None
_C='item_id'
_B='periods'
_A=False
import json,pandas as pd
from typing import Optional,List,Dict,Tuple,Union
from enum import Enum
from app.lib._vnstock_shared.compat import agg_execution
import logging
from app.lib.vnstock_data_alt.core.utils.parser import get_asset_type,vn_to_snake_case
from app.lib.vnstock_data_alt.core.utils.client import send_request,ProxyConfig
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
from app.lib.vnstock_data_alt.explorer.kbs.const import _SAS_FINANCE_INFO_URL,_INCOME_STATEMENT_MAP,_BALANCE_SHEET_MAP,_CASH_FLOW_MAP,_FINANCIAL_RATIOS_MAP,_FINANCIAL_REPORT_TYPE_MAP,_FINANCIAL_PERIOD_TYPE_MAP
logger=logging.getLogger(__name__)
class FieldDisplayMode(Enum):'Field display modes.';STD='std';ALL='all';AUTO='auto'
class Finance:
	'\n    Lớp truy cập dữ liệu tài chính từ KB Securities (KBS).\n    '
	def __init__(A,symbol,period=_D,random_agent=_A,proxy_config=_D,show_log=_A,standardize_columns=True,proxy_mode=_D,proxy_list=_D):
		"\n        Khởi tạo Finance client cho KBS.\n\n        Args:\n            symbol: Mã chứng khoán (VD: 'ACB', 'VNM').\n            period: Kỳ báo cáo mặc định ('year', 'quarter' hoặc None).\n            random_agent: Sử dụng user agent ngẫu nhiên. Mặc định False.\n            proxy_config: Cấu hình proxy. Mặc định None.\n            show_log: Hiển thị log debug. Mặc định False.\n            standardize_columns: Chuẩn hoá tên cột theo schema. Mặc định True.\n            proxy_mode: Chế độ proxy (try, rotate, random, single). Mặc định None.\n            proxy_list: Danh sách proxy URLs. Mặc định None.\n\n        Raises:\n            ValueError: Nếu mã không phải là cổ phiếu.\n        ";F=proxy_mode;E=show_log;D=proxy_config;C=proxy_list;B=period;A.symbol=symbol.upper();A.asset_type=get_asset_type(A.symbol)
		if B is not _D and B not in[_G,'quarter']:raise ValueError("Kỳ báo cáo tài chính không hợp lệ. Chỉ chấp nhận 'year' hoặc 'quarter' hoặc None.")
		A.period=B
		if A.asset_type not in['stock']:raise ValueError('Mã CK không hợp lệ hoặc không phải cổ phiếu.')
		A.data_source='KBS';A.headers=get_headers(data_source=A.data_source,random_agent=random_agent);A.show_log=E;A.standardize_columns=standardize_columns
		if D is _D:
			H=F if F else'try';G='direct'
			if C and len(C)>0:G='proxy'
			A.proxy_config=ProxyConfig(proxy_mode=H,proxy_list=C,request_mode=G)
		else:A.proxy_config=D
		if E:logger.setLevel('INFO')
		else:logger.setLevel('CRITICAL')
	def _get_column_mapping(B,report_type):'\n        Lấy column mapping cho loại báo cáo.\n        \n        Args:\n            report_type: Loại báo cáo (income_statement, balance_sheet, cash_flow, financial_ratios)\n            \n        Returns:\n            Dictionary chứa mapping từ cột gốc sang cột chuẩn hoá\n        ';A={_S:_INCOME_STATEMENT_MAP,_T:_BALANCE_SHEET_MAP,_U:_CASH_FLOW_MAP,_V:_FINANCIAL_RATIOS_MAP};return A.get(report_type,{})
	def _parse_financial_response(t,response,report_key,include_metadata=_A):
		"\n        Parse KBS API response and extract financial data with proper structure.\n        \n        Args:\n            response: API response containing Audit, Unit, Head, Content\n            report_key: Key in Content (e.g., 'Kết quả kinh doanh')\n            include_metadata: Whether to include Audit and Unit info as rows in DataFrame\n            \n        Returns:\n            DataFrame with proper financial data structure\n        ";k='Quý';j='Unit';X=report_key;G=response;B='';Y=G.get('Audit',[]);Z=G.get(j,[]);a=G.get('Head',[]);l=G.get(_M,{});b=l.get(X,[])
		if not b:return pd.DataFrame()
		H=[];O={};P={}
		if a:
			m=sorted(a,key=lambda x:x.get('ID',0))
			for F in m:
				if isinstance(F,dict):
					c=F.get('YearPeriod',B);Q=F.get('TermName',B)
					if Q and k in Q:n=Q.replace(k,B).strip();D=f"{c}-Q{n}"
					else:D=str(c)
					H.append(D);O[D]=F.get('AuditedStatus',B);P[D]=F.get('United',B)
		R={}
		if Y:
			for S in Y:
				if isinstance(S,dict):R[S.get('AuditedStatusCode')]=S.get('Description')
		T={}
		if Z:
			for U in Z:
				if isinstance(U,dict):T[U.get('UnitedCode')]=U.get('UnitedName')
		I=[];J={}
		for E in b:
			K=E.get('Name',B);L=E.get('NameEn',B)
			if L and L.strip():A=vn_to_snake_case(L)
			elif K and K.strip():A=vn_to_snake_case(K)
			else:A=B
			if A and A in J:J[A]+=1;A=A+'_'+str(J[A])
			elif A:J[A]=1
			d={_H:K,_I:L,_C:A,_J:E.get(j,B),_K:E.get('Levels',0),_L:E.get('ID',0)}
			for(o,D)in enumerate(H,1):
				p=f"Value{o}";M=E.get(p)
				if M is not _D:
					try:M=float(M)
					except(ValueError,TypeError):pass
				d[D]=M
			I.append(d)
		if include_metadata:
			e={_H:'Kiểm toán',_I:'Audit Status',_C:_E,_J:B,_K:0,_L:-2};f={_H:'Đơn vị',_I:'Unit Type',_C:_F,_J:B,_K:0,_L:-1}
			for N in H:g=O.get(N);e[N]=R.get(g,g);h=P.get(N);f[N]=T.get(h,h)
			I.append(e);I.append(f)
		C=pd.DataFrame(I);q=[_H,_I,_C,_J,_K,_L];r=[A for A in q if A in C.columns];s=[A for A in H if A in C.columns];V=[]
		for i in s:
			if not C[i].isnull().all():V.append(i)
		C=C[r+V];W=V;C.attrs[_E]={A:R.get(B,B)for(A,B)in O.items()if A in W};C.attrs[_F]={A:T.get(B,B)for(A,B)in P.items()if A in W};C.attrs[_B]=W;C.attrs['report_key']=X;return C
	def _apply_schema_standardization(B,df,report_type):
		'\n        Áp dụng chuẩn hoá schema cho DataFrame.\n        \n        Args:\n            df: DataFrame cần chuẩn hoá\n            report_type: Loại báo cáo\n            \n        Returns:\n            DataFrame với cột chuẩn hoá\n        ';A=df
		if not B.standardize_columns or A.empty:return A
		D=B._get_column_mapping(report_type);C={B:C for(B,C)in D.items()if B in A.columns}
		if C:
			A=A.rename(columns=C)
			if B.show_log:logger.info(f"Applied schema standardization: {len(C)} columns renamed")
		return A
	def _filter_columns_by_lang(G,df,display_mode=FieldDisplayMode.STD):
		"\n        Filter DataFrame columns based on field display mode.\n        \n        Args:\n            df: DataFrame to filter\n            display_mode: Field display mode\n                - FieldDisplayMode.STD: Keep only 'item' and 'item_id' columns (standardized)\n                - FieldDisplayMode.ALL: Keep all item columns (item, item_en, item_id)\n                - FieldDisplayMode.AUTO: Auto-convert based on data type\n                - 'vi': Keep Vietnamese names only (backward compatibility)\n                - 'en': Keep English names only (backward compatibility)\n                - None: Keep all item columns (backward compatibility)\n            \n        Returns:\n            DataFrame with filtered columns\n        ";C=df;A=display_mode
		if C.empty:return C
		if isinstance(A,str):
			if A=='vi':A=FieldDisplayMode.STD
			elif A=='en':A=FieldDisplayMode.STD
			else:A=FieldDisplayMode.ALL
		F=C.attrs.get(_B,[]);E=[A for A in C.columns if A not in F];D=C.copy()
		if A==FieldDisplayMode.ALL:B=E
		elif A==FieldDisplayMode.AUTO:B=E
		else:
			B=[A for A in E if A in[_H,_C]]
			if isinstance(A,str)and A=='en'and _I in D.columns:D[_H]=D[_I];B=[_H,_C]
		B.extend(F);B=[A for A in B if A in D.columns];return D[B]
	def _fetch_financial_data(A,report_type='KQKD',period_type=1,page=1,page_size=4,show_log=_A):
		'\n        Lấy dữ liệu tài chính từ API SAS với các tham số chính xác.\n\n        Args:\n            report_type: Loại báo cáo (CDKT, KQKD, LCTT, CSTC, CTKH, BCTT)\n            period_type: Loại kỳ báo cáo (1=năm, 2=quý)\n            page: Trang (mặc định 1)\n            page_size: Số bản ghi trên trang (mặc định 4)\n            show_log: Hiển thị log debug.\n\n        Returns:\n            Dictionary chứa dữ liệu tài chính đầy đủ.\n        ';G='data';F=period_type;E=report_type;B=show_log;H=f"{_SAS_FINANCE_INFO_URL}/{A.symbol}";C={'page':page,'pageSize':page_size,'type':E,_J:1000,'termtype':F}
		if E!=_W:C['languageid']=1
		else:C['code']=A.symbol;C['termType']=F
		if B or A.show_log:logger.info(f"KBS Financial API Request: {A.symbol} - {E} - Period: {F}")
		try:
			D=send_request(url=H,headers=A.headers,method='GET',params=C,show_log=B or A.show_log,proxy_list=A.proxy_config.proxy_list,proxy_mode=A.proxy_config.proxy_mode,request_mode=A.proxy_config.request_mode)
			if B or A.show_log:
				if isinstance(D,dict)and G in D:logger.info('API Response received: '+str(len(D.get(G,[])))+' records')
			return D
		except Exception as I:
			if B or A.show_log:logger.error(f"API Request Failed: {str(I)}")
			raise
	def _fetch_series_data(H,report_type,period_type,report_key,limit=12,include_metadata=_A,show_log=_A):
		'\n        Helper to fetch data across multiple pages to satisfy the limit.\n        ';C=limit;D=[];I=[];E=1;L=max(C,4)
		while len(I)<C:
			M=H._fetch_financial_data(report_type=report_type,period_type=period_type,page=E,page_size=L,show_log=show_log);F=H._parse_financial_response(M,report_key,include_metadata=include_metadata)
			if F.empty:break
			J=F.attrs.get(_B,[])
			if not J:break
			D.append(F);I.extend(J);E+=1
			if E>50:break
		if not D:return pd.DataFrame()
		A=D[0];S=[_H,_I,_C,_J,_K,_L]
		for N in range(1,len(D)):
			B=D[N];K=B.attrs[_B];O=[_C]+K
			if _C in B.columns:
				P=A.attrs;A=pd.merge(A,B[O],on=_C,how='outer');A.attrs=P
				if _E in B.attrs:
					if _E not in A.attrs:A.attrs[_E]={}
					A.attrs[_E].update(B.attrs[_E])
				if _F in B.attrs:
					if _F not in A.attrs:A.attrs[_F]={}
					A.attrs[_F].update(B.attrs[_F])
				if _B in A.attrs:A.attrs[_B].extend(K)
		G=A.attrs[_B]
		if len(G)>C:Q=G[:C];R=G[C:];A=A.drop(columns=R,errors=_X);A.attrs[_B]=Q
		return A
	@agg_execution(_R)
	def income_statement(self,period=_D,limit=12,include_metadata=_A,display_mode=FieldDisplayMode.STD,show_log=_A):
		"\n        Truy xuất báo cáo kết quả kinh doanh (income statement).\n\n        Args:\n            period: Loại kỳ báo cáo ('year' hoặc 'quarter'). Mặc định 'year'.\n            limit: Số kỳ báo cáo tối đa cần lấy. Mặc định 4.\n            include_metadata: Bao gồm thông tin audit và unit trong rows. Mặc định False.\n            display_mode: Chế độ hiển thị trường dữ liệu. Mặc định FieldDisplayMode.STD.\n                - FieldDisplayMode.STD: Chỉ giữ cột 'item' và 'item_id' (đã chuẩn hóa)\n                - FieldDisplayMode.ALL: Giữ tất cả cột item (item, item_en, item_id)\n                - 'vi': Chỉ giữ tên tiếng Việt (tương thích ngược)\n                - 'en': Chỉ giữ tên tiếng Anh (tương thích ngược)\n                - None: Giữ tất cả cột (tương thích ngược)\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa báo cáo kết quả kinh doanh.\n\n        Examples:\n            >>> finance = Finance('ACB')\n            >>> df = finance.income_statement(period='year', display_mode=FieldDisplayMode.STD)\n            >>> # Returns DataFrame with columns: item, item_id, unit, periods...\n            >>> df_all = finance.income_statement(period='year', display_mode=FieldDisplayMode.ALL)\n            >>> # Returns DataFrame with all item columns\n            >>> # Backward compatibility:\n            >>> df_vi = finance.income_statement(period='year', display_mode='vi')\n            >>> df_en = finance.income_statement(period='year', display_mode='en')\n        ";D=show_log;C=period;A=self;E=C if C else A.period if A.period else _G;F=1 if E==_G else 2;B=A._fetch_series_data(report_type='KQKD',period_type=F,report_key='Kết quả kinh doanh',limit=limit,include_metadata=include_metadata,show_log=D)
		if B.empty:logger.warning(f"Không tìm thấy báo cáo kết quả kinh doanh cho {A.symbol}.");return pd.DataFrame()
		if A.standardize_columns:B=A._apply_schema_standardization(B,_S)
		B=A._filter_columns_by_lang(B,display_mode);B.attrs[_N]=A.symbol;B.attrs[_O]=A.data_source;B.attrs[_P]=_S;B.attrs[_Q]=E
		if D or A.show_log:logger.info(f"Truy xuất thành công báo cáo kết quả kinh doanh cho {A.symbol}.")
		return B
	@agg_execution(_R)
	def balance_sheet(self,period=_D,limit=12,include_metadata=_A,display_mode=FieldDisplayMode.STD,show_log=_A):
		"\n        Truy xuất bảng cân đối kế toán (balance sheet).\n\n        Args:\n            period: Loại kỳ báo cáo ('year' hoặc 'quarter'). Mặc định 'year'.\n            limit: Số kỳ báo cáo tối đa cần lấy. Mặc định 4.\n            include_metadata: Bao gồm thông tin audit và unit trong rows. Mặc định False.\n            display_mode: Chế độ hiển thị trường dữ liệu. Mặc định FieldDisplayMode.STD.\n                - FieldDisplayMode.STD: Chỉ giữ cột 'item' và 'item_id' (đã chuẩn hóa)\n                - FieldDisplayMode.ALL: Giữ tất cả cột item (item, item_en, item_id)\n                - 'vi': Chỉ giữ tên tiếng Việt (tương thích ngược)\n                - 'en': Chỉ giữ tên tiếng Anh (tương thích ngược)\n                - None: Giữ tất cả cột (tương thích ngược)\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa bảng cân đối kế toán.\n\n        Examples:\n            >>> finance = Finance('ACB')\n            >>> df = finance.balance_sheet(period='year', display_mode=FieldDisplayMode.STD)\n            >>> df_all = finance.balance_sheet(period='year', display_mode=FieldDisplayMode.ALL)\n            >>> # Backward compatibility:\n            >>> df_vi = finance.balance_sheet(period='year', display_mode='vi')\n            >>> df_en = finance.balance_sheet(period='year', display_mode='en')\n        ";D=show_log;C=period;A=self;E=C if C else A.period if A.period else _G;F=1 if E==_G else 2;B=A._fetch_series_data(report_type='CDKT',period_type=F,report_key='Cân đối kế toán',limit=limit,include_metadata=include_metadata,show_log=D)
		if B.empty:logger.warning(f"Không tìm thấy bảng cân đối kế toán cho {A.symbol}.");return pd.DataFrame()
		if A.standardize_columns:B=A._apply_schema_standardization(B,_T)
		B=A._filter_columns_by_lang(B,display_mode);B.attrs[_N]=A.symbol;B.attrs[_O]=A.data_source;B.attrs[_P]=_T;B.attrs[_Q]=E
		if D or A.show_log:logger.info(f"Truy xuất thành công bảng cân đối kế toán cho {A.symbol}.")
		return B
	@agg_execution(_R)
	def cash_flow(self,period=_D,limit=12,include_metadata=_A,display_mode=FieldDisplayMode.STD,show_log=_A):
		"\n        Truy xuất báo cáo lưu chuyển tiền tệ (cash flow statement).\n\n        Args:\n            period: Loại kỳ báo cáo ('year' hoặc 'quarter'). Mặc định 'year'.\n            limit: Số kỳ báo cáo tối đa cần lấy. Mặc định 4.\n            include_metadata: Bao gồm thông tin audit và unit trong rows. Mặc định False.\n            display_mode: Chế độ hiển thị trường dữ liệu. Mặc định FieldDisplayMode.STD.\n                - FieldDisplayMode.STD: Chỉ giữ cột 'item' và 'item_id' (đã chuẩn hóa)\n                - FieldDisplayMode.ALL: Giữ tất cả cột item (item, item_en, item_id)\n                - 'vi': Chỉ giữ tên tiếng Việt (tương thích ngược)\n                - 'en': Chỉ giữ tên tiếng Anh (tương thích ngược)\n                - None: Giữ tất cả cột (tương thích ngược)\n            show_log: Hiển thị log debug.\n\n        Returns:\n            DataFrame chứa báo cáo lưu chuyển tiền tệ.\n\n        Examples:\n            >>> finance = Finance('ACB')\n            >>> df = finance.cash_flow(period='year', display_mode=FieldDisplayMode.STD)\n            >>> df_all = finance.cash_flow(period='year', display_mode=FieldDisplayMode.ALL)\n            >>> # Backward compatibility:\n            >>> df_vi = finance.cash_flow(period='year', display_mode='vi')\n            >>> df_en = finance.cash_flow(period='year', display_mode='en')\n        ";K='Lưu chuyển tiền tệ trực tiếp';J='Lưu chuyển tiền tệ gián tiếp';E=show_log;D=period;A=self;F=D if D else A.period if A.period else _G;G=1 if F==_G else 2;H=A._fetch_financial_data(report_type=_W,period_type=G,page_size=1,show_log=_A)
		if not H:raise ValueError(f"Không tìm thấy dữ liệu tài chính cho mã {A.symbol}.")
		I=H.get(_M,{});C=_D
		if J in I:C=J
		elif K in I:C=K
		if not C:logger.warning(f"Không tìm thấy báo cáo lưu chuyển tiền tệ cho {A.symbol}.");return pd.DataFrame()
		B=A._fetch_series_data(report_type=_W,period_type=G,report_key=C,limit=limit,include_metadata=include_metadata,show_log=E)
		if B.empty:logger.warning(f"Không tìm thấy báo cáo lưu chuyển tiền tệ cho {A.symbol}.");return pd.DataFrame()
		if A.standardize_columns:B=A._apply_schema_standardization(B,_U)
		B=A._filter_columns_by_lang(B,display_mode);B.attrs[_N]=A.symbol;B.attrs[_O]=A.data_source;B.attrs[_P]=_U;B.attrs[_Q]=F
		if E or A.show_log:logger.info(f"Truy xuất thành công báo cáo lưu chuyển tiền tệ cho {A.symbol}.")
		return B
	@agg_execution(_R)
	def ratio(self,period=_D,limit=12,include_metadata=_A,display_mode=FieldDisplayMode.STD,show_log=_A):
		"\n        Truy xuất các chỉ số tài chính (financial ratios).\n\n        Args:\n            period: Loại kỳ báo cáo ('year' hoặc 'quarter'). Mặc định 'year'.\n            limit: Số kỳ báo cáo tối đa cần lấy. Mặc định 4.\n            include_metadata: Bao gồm thông tin audit và unit trong rows. Mặc định False.\n            display_mode: Chế độ hiển thị trường dữ liệu. Mặc định FieldDisplayMode.STD.\n                - FieldDisplayMode.STD: Chỉ giữ cột 'item' và 'item_id' (đã chuẩn hóa)\n                - FieldDisplayMode.ALL: Giữ tất cả cột item (item, item_en, item_id)\n                - 'vi': Chỉ giữ tên tiếng Việt (tương thích ngược)\n                - 'en': Chỉ giữ tên tiếng Anh (tương thích ngược)\n                - None: Giữ tất cả cột (tương thích ngược)\n# Register provider\nfrom app.lib.vnstock_data_alt.core.registry import ProviderRegistry\nProviderRegistry.register('financial', 'kbs', Finance)\n\n\n        Returns:\n            DataFrame chứa các chỉ số tài chính.\n\n        Examples:\n            >>> finance = Finance('ACB')\n            >>> df = finance.ratio(period='year', display_mode=FieldDisplayMode.STD)\n            >>> df_all = finance.ratio(period='year', display_mode=FieldDisplayMode.ALL)\n            >>> # Backward compatibility:\n            >>> df_vi = finance.ratio(period='year', display_mode='vi')\n            >>> df_en = finance.ratio(period='year', display_mode='en')\n        ";R='Financial Ratios Combined';L=show_log;K=period;D=limit;B=self;M=K if K else B.period if B.period else _G;S=1 if M==_G else 2;E=[];N=[];G=1;T=max(D,4)
		while len(N)<D:
			F=B._fetch_financial_data(report_type='CSTC',period_type=S,page=G,page_size=T,show_log=L)
			if not F:break
			U=F.get(_M,{});V=['Nhóm chỉ số Định giá','Nhóm chỉ số Sinh lợi','Nhóm chỉ số Tăng trưởng','Nhóm chỉ số Thanh khoản','Nhóm chỉ số Chất lượng tài sản'];H=[]
			for W in V:
				O=U.get(W,[])
				if O:H.extend(O)
			if not H:break
			F[_M][R]=H;I=B._parse_financial_response(F,R,include_metadata=include_metadata)
			if I.empty:break
			P=I.attrs.get(_B,[])
			if not P:break
			E.append(I);N.extend(P);G+=1
			if G>50:break
		if not E:logger.warning(f"Không tìm thấy chỉ số tài chính cho {B.symbol}.");return pd.DataFrame()
		A=E[0]
		for X in range(1,len(E)):
			C=E[X];Q=C.attrs[_B];Y=[_C]+Q
			if _C in C.columns:
				Z=A.attrs;A=pd.merge(A,C[Y],on=_C,how='outer');A.attrs=Z
				if _E in C.attrs:
					if _E not in A.attrs:A.attrs[_E]={}
					A.attrs[_E].update(C.attrs[_E])
				if _F in C.attrs:
					if _F not in A.attrs:A.attrs[_F]={}
					A.attrs[_F].update(C.attrs[_F])
				if _B in A.attrs:A.attrs[_B].extend(Q)
		J=A.attrs[_B]
		if len(J)>D:a=J[:D];b=J[D:];A=A.drop(columns=b,errors=_X);A.attrs[_B]=a
		if B.standardize_columns:A=B._apply_schema_standardization(A,_V)
		A=B._filter_columns_by_lang(A,display_mode);A.attrs[_N]=B.symbol;A.attrs[_O]=B.data_source;A.attrs[_P]=_V;A.attrs[_Q]=M
		if L or B.show_log:logger.info(f"Truy xuất thành công chỉ số tài chính cho {B.symbol}.")
		return A
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('financial','kbs',Finance)