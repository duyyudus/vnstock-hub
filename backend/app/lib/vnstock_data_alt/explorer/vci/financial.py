'\nModule quản lý thông tin báo cáo tài chính từ nguồn dữ liệu VCI.\n'
_X='cash_flow'
_W='income_statement'
_V="'. Các ngôn ngữ được hỗ trợ: "
_U="Ngôn ngữ không hợp lệ: '"
_T="Loại báo cáo tài chính không hợp lệ: '"
_S='en_name'
_R='dict'
_Q='balance_sheet'
_P='data'
_O=', '
_N='vi'
_M='name'
_L='VCI.ext'
_K='code'
_J='report_period'
_I='field_name'
_H='year'
_G='final'
_F='en'
_E=True
_D='quarter'
_C='readable'
_B=None
_A=False
import json,pandas as pd
from typing import Optional,Dict,Tuple,Union
from.const import _VCIQ_URL,_IQ_FINANCE_REPORT
from app.lib.vnstock_alt.explorer.vci.const import _GRAPHQL_URL,_UNIT_MAP,SUPPORTED_LANGUAGES
from app.lib._vnstock_shared.core.utils.parser import get_asset_type,camel_to_snake
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib._vnstock_shared.core.utils.transform import reorder_cols
from app.lib.vnstock_data_alt.core.utils.client import send_request,ProxyConfig
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
from app.lib.vnstock_data_alt.core.utils.parser import vn_to_snake_case
from app.lib.vnstock_data_alt.core.utils.transform import generate_period,remove_pattern_columns
from app.lib._vnstock_shared.compat import agg_execution
logger=get_logger(__name__)
class Finance:
	"\n    Truy xuất thông tin báo cáo tài chính của một công ty theo mã chứng khoán từ nguồn dữ liệu VCI.\n\n    Tham số:\n        - symbol (str): Mã chứng khoán của công ty cần truy xuất thông tin.\n        - period (str): Chu kỳ báo cáo tài chính cần truy xuất. Mặc định là 'quarter'.\n        - get_all (bool): Trả về tất cả các trường dữ liệu hoặc chỉ các trường chọn lọc. Mặc định là True.\n        - show_log (bool): Hiển thị thông tin log hoặc không. Mặc định là True.\n    "
	def __init__(A,symbol,period=_B,get_all=_E,proxy_config=_B,show_log=_A):
		'\n        Khởi tạo đối tượng Finance với các tham số cho việc truy xuất dữ liệu báo cáo tài chính.\n        ';D=show_log;C=proxy_config;B=period;A.symbol=symbol.upper();A.asset_type=get_asset_type(A.symbol);A.headers=get_headers(data_source='VCI');A.base_url=_VCIQ_URL;A.show_log=D;A.proxy_config=C if C is not _B else ProxyConfig()
		if not D:logger.setLevel('CRITICAL')
		if B not in[_H,_D]and B!=_B:raise ValueError("Kỳ báo cáo tài chính không hợp lệ. Chỉ chấp nhận 'year' hoặc 'quarter' hoặc None.")
		if A.asset_type not in['stock']:raise ValueError('Mã chứng khoán không hợp lệ. Chỉ cổ phiếu mới có thông tin.')
		A.period=B;A.get_all=get_all
	@staticmethod
	def duplicated_columns_handling(df_or_mapping,target_col_name=_B):
		"\n        Handle duplicated column names in a DataFrame or column mapping DataFrame.\n        \n        Parameters:\n            - df_or_mapping (pd.DataFrame): Either a DataFrame with potentially duplicated columns\n            or a mapping DataFrame with columns that may have duplicated values.\n            - target_col_name (str, optional): When handling a mapping DataFrame, this is the column\n            to check for duplicates. When None, assumes we're handling DataFrame columns directly.\n        \n        Returns:\n            pd.DataFrame: DataFrame with resolved column duplications.\n        ";C=target_col_name;A=df_or_mapping
		if C is not _B:H=A[A[C].duplicated()].copy();J=A[~A[C].duplicated()].copy();H[C]=A[_M]+' - '+A[_I];return pd.concat([J,H])
		else:
			B=A.copy();K=B.columns.duplicated(keep=_A);I=B.columns[K].unique()
			if len(I)>0:
				D=B.columns.tolist()
				for F in I:
					L=[A for(A,B)in enumerate(D)if B==F]
					for(E,M)in enumerate(L):
						if E==0:continue
						G='_'*E+F
						while G in D:E+=1;G='_'*E+F
						D[M]=G
				B.columns=D
			return B
	def _get_ratio_dict(B,lang=_N,format=_R,style=_C,show_log=_A):
		"\n        Lấy từ điển ánh xạ cho tất cả các chỉ số tài chính từ nguồn VCI.\n\n        Tham số:\n            - lang (str): Ngôn ngữ của báo cáo. Mặc định là 'en'.\n            - format (str): Định dạng trả về. Mặc định là 'dict', lựa chọn khác có thể là 'dataframe'.\n            - style (str): Định dạng trả về. Mặc định là 'readable', lựa chọn khác có thể là 'code'.\n            - show_log (bool): Hiển thị thông tin log hoặc không. Mặc định là False.\n            \n        Returns:\n            pd.DataFrame: DataFrame chứa ánh xạ giữa 'field_name', 'name', 'en_name', 'type', 'order', 'unit'.\n        ";P='en_full_name';O='fullTitleEn';N='fullTitleVi';M='titleVi';L='titleEn';K='field';J='report_name';E=show_log;D=style;C=lang
		if C not in SUPPORTED_LANGUAGES:raise ValueError("Ngôn ngữ '"+C+"' không hợp lệ. Chỉ chấp nhận "+_O.join(SUPPORTED_LANGUAGES)+'.')
		if format not in[_R,'dataframe']:raise ValueError("Định dạng '"+str(format)+"' không hợp lệ. Chỉ chấp nhận 'dict' hoặc 'dataframe'.")
		F=f"{B.base_url}/v1/company/{B.symbol}/financial-statement/metrics"
		if E:logger.debug(f"Requesting financial ratio data from {F}")
		Q=send_request(url=F,headers=B.headers,method='GET',payload=_B,show_log=E,proxy_list=B.proxy_config.proxy_list,proxy_mode=B.proxy_config.proxy_mode,request_mode=B.proxy_config.request_mode);G=Q[_P];H=[]
		for I in G.keys():A=pd.DataFrame(G[I]);A[J]=I;A=A[[J,K,'parent',L,M,N,O]];H.append(A)
		A=pd.concat(H);A=A.rename(columns={K:_I,M:_M,L:_S,N:'full_name',O:P})
		if format==_R:
			if C==_N:
				if D==_C:return A.set_index(_I).to_dict()[_M]
				elif D==_K:return{vn_to_snake_case(A.lower()if A else''):vn_to_snake_case(B.lower()if B else'')for(A,B)in A.set_index(_I)[_M].to_dict().items()if A and B}
			elif C==_F:
				if D==_K:return{A.lower():B.lower().replace(' ','_')for(A,B)in A.set_index(_I)[_S].to_dict().items()if A and B}
				elif D==_C:return{A:B for(A,B)in A.set_index(_I)[P].to_dict().items()if A and B}
		else:return A
	def _get_old_ratio_dict(B,show_log=_A,get_all=_A):
		"\n        Lấy từ điển ánh xạ cho tất cả các chỉ số tài chính từ nguồn VCI.\n\n        Tham số:\n            - show_log (bool): Hiển thị thông tin log hoặc không. Mặc định là False.\n            - get_all (bool): Lấy tất cả cột hoặc không. Mặc định là False.\n            \n        Returns:\n            pd.DataFrame: DataFrame chứa ánh xạ giữa 'field_name', 'name', 'en_name', 'type', 'order', 'unit'.\n        ";E=get_all;D=show_log;C='unit';F='{"query":"query Query {\\n  ListFinancialRatio {\\n    id\\n    type\\n    name\\n    unit\\n    isDefault\\n    fieldName\\n    en_Type\\n    en_Name\\n    tagName\\n    comTypeCode\\n    order\\n    __typename\\n  }\\n}\\n","variables":{}}';G=json.loads(F)
		if D:logger.debug(f"Requesting financial ratio data from {_GRAPHQL_URL}. payload: {F}")
		H=send_request(url=_GRAPHQL_URL,headers=B.headers,method='POST',payload=G,show_log=D,proxy_list=B.proxy_config.proxy_list,proxy_mode=B.proxy_config.proxy_mode,request_mode=B.proxy_config.request_mode);I=H[_P]['ListFinancialRatio'];A=pd.DataFrame(I);A.columns=[camel_to_snake(A).replace('__','_')for A in A.columns];J=E if E is not _B else B.get_all;K=[_I,_M,_S,'type','order',C,'com_type_code'];A[C]=A[C].map(_UNIT_MAP)
		if J is _A:A=A[K]
		A.columns=[A.replace('__','_')for A in A.columns];return A
	def _get_report(D,report_type=_B,lang=_F,show_log=_A,mode=_G,style=_C,get_all=_A):
		"\n        Lấy dữ liệu báo cáo tài chính cho một công ty từ nguồn VCI.\n        \n        Tham số:\n            - report_type (str): Loại báo cáo tài chính bao gồm 'income_statement', 'balance_sheet', 'cash_flow' và 'note'\n            - lang (str): Ngôn ngữ của báo cáo. Mặc định là 'en'.\n            - show_log (bool): Hiển thị thông tin log hoặc không. Mặc định là False.\n            - mode (str): Chế độ báo cáo. Mặc định là 'final' trả về dữ liệu đã xử lý sau quá trình ánh xạ. \n              Chế độ khác là 'raw' trả về dữ liệu thô chứa tên mã cho tất cả các trường.\n            - style (str): Chế độ tên cột. Mặc định là 'readable' trả về tên cột dễ đọc. \n              Chế độ khác là 'code' trả về tên cột mã hóa.\n            - get_all (bool): Trả về tất cả các trường dữ liệu hoặc chỉ các trường chọn lọc. Mặc định là False.\n              \n        Returns:\n            Union[Tuple[Dict[str, pd.DataFrame], pd.DataFrame], pd.DataFrame]: \n                Nếu mode='final': Trả về tuple gồm dictionary các báo cáo chính và DataFrame cho các báo cáo khác\n                Nếu mode='raw': Trả về DataFrame dữ liệu thô\n        ";O='RATIO';I=mode;H=lang;E=report_type;C=show_log
		if H not in SUPPORTED_LANGUAGES:P=_O.join(SUPPORTED_LANGUAGES);raise ValueError(f"Ngôn ngữ không hợp lệ: '{H}'. Các ngôn ngữ được hỗ trợ: {P}.")
		if E not in _IQ_FINANCE_REPORT.keys():raise ValueError(_T+str(E)+"'. Các loại báo cáo tài chính được hỗ trợ: "+_O.join(_IQ_FINANCE_REPORT.keys())+'.')
		else:E=_IQ_FINANCE_REPORT[E]
		if E==O:J={};K=f"https://iq.vietcap.com.vn/api/iq-insight-service/v1/company/{D.symbol}/statistics-financial"
		else:K=f"https://iq.vietcap.com.vn/api/iq-insight-service/v1/company/{D.symbol}/financial-statement";J={'section':E}
		if C:logger.debug(f"Requesting financial report data from {K}. payload: {J}")
		G=send_request(url=K,headers=D.headers,method='GET',params=J,payload=_B,show_log=C,proxy_list=D.proxy_config.proxy_list,proxy_mode=D.proxy_config.proxy_mode,request_mode=D.proxy_config.request_mode)
		try:
			if G is _B or _P not in G or G[_P]is _B:
				A='No data received from the API'
				if C:logger.error(f"{A}. Response: {G}")
				raise ValueError(A)
			B=G[_P]
			if E==O:
				if not isinstance(B,list):
					A=f"Unexpected data format for ratio. Expected list, got {type(B).__name__}"
					if C:logger.error(f"{A}. Data: {B}")
					raise ValueError(A)
				F=pd.DataFrame(B)
				if F.empty:
					A='No valid ratio data found in the response'
					if C:logger.error(f"{A}. Data: {B}")
					raise ValueError(A)
				L=F
			else:
				if not isinstance(B,dict):
					A=f"Unexpected data format. Expected dict, got {type(B).__name__}"
					if C:logger.error(f"{A}. Data: {B}")
					raise ValueError(A)
				M=[]
				for(Q,N)in B.items():
					if N:
						F=pd.DataFrame(N)
						if not F.empty:F[_J]=Q[:-1];M.append(F)
				if not M:
					A='No valid data found in the response'
					if C:logger.error(f"{A}. Data: {B}")
					raise ValueError(A)
				L=pd.concat(M,ignore_index=_E)
			if I==_G:R=D._ratio_mapping(report_df=L,lang=H,style=style,get_all=get_all,show_log=C);return R
			elif I=='raw':return L
			else:A=f"Invalid mode: {I}. Must be 'final' or 'raw'.";logger.error(A);raise ValueError(A)
		except Exception as S:logger.error(f"Error processing financial report data: {S}",exc_info=_E);raise
	@agg_execution(_L)
	def _ratio_mapping(self,report_df,lang=_N,style=_C,get_all=_A,show_log=_A):
		"\n        A dedicated method to map the financial ratio DataFrame columns to the dictionary ratio_dict.\n\n        Parameters:\n            - report_df (pd.DataFrame): The DataFrame containing the financial ratio from the function _get_report().\n            - lang (str): The language of the report. Default is 'vi'.\n            - style (str): The style of the report. Default is 'readable'.\n            - get_all (bool): Whether to get all raw columns or just essential columns. Default is False for removing optional columns.\n            - show_log (bool): Whether to show log messages. Default is False.\n\n        Returns:\n            - pd.DataFrame: A DataFrame containing the financial ratio data.\n        ";O='ky_bao_cao';N='Kỳ báo cáo';M='publicDate';L='updateDate';K='createDate';J='organCode';G='ticker';F='period';E='yearReport';D=style;C=lang;B='lengthReport';A=report_df
		if C not in SUPPORTED_LANGUAGES:P=_O.join(SUPPORTED_LANGUAGES);raise ValueError(_U+str(C)+_V+P+'.')
		if D not in[_C,_K]:raise ValueError("Chế độ không hợp lệ: '"+str(D)+"'. Chế độ được hỗ trợ: 'readable' cho tên hiển thị hoặc 'code' cho tên mã.")
		H=self._get_ratio_dict(lang=C,style=D,format=_R);A.columns=[H[A]if A in H else A for A in A.columns]
		if B not in A.columns and _D in A.columns:A[B]=A[_D]
		if E not in A.columns and _H in A.columns:A[E]=A[_H]
		if _J not in A.columns:
			if _D in A.columns:A[_J]=A[_D].apply(lambda x:_H if x==5 else _D)
			else:A[_J]=_H
		if _D in A.columns and B in A.columns:Q=A[_D]==5;A.loc[Q,B]=4
		A=generate_period(A);A=reorder_cols(A,cols=[F,_J,J,G,K,L,E,B,M],position='first')
		if C==_F:A=A.set_index(F)
		elif C==_N:
			if D==_C:A=A.rename(columns={F:N,G:'Mã CP'});A=A.set_index(N)
			elif D==_K:A=A.rename(columns={F:O,G:'cp'});A=A.set_index(O)
		if get_all==_A:
			R=[J,K,L,E,B,M];I=[B for B in R if B in A.columns]
			if I:A=A.drop(columns=I)
			try:A=remove_pattern_columns(A,['bsa','bsb','bsi','bss','nob','nos','cfa','cfs','cfi','isa','isi','iss'])
			except Exception as S:logger.error(f"Error removing pattern columns: {S}");raise
			finally:return A
		else:return A
	def _get_financial_report(C,report_type,period=_B,lang=_F,mode=_G,style=_C,get_all=_A,dropna=_E,show_log=_A):
		'\n        Internal method to retrieve and filter financial reports by type and period.\n        ';F=lang;E=period;B=report_type
		if F not in SUPPORTED_LANGUAGES:raise ValueError(_U+str(F)+_V+_O.join(SUPPORTED_LANGUAGES)+'.')
		if B not in[_Q,_W,_X,'note','ratio']:raise ValueError(_T+str(B)+"'. Các loại báo cáo tài chính được hỗ trợ: 'balance_sheet', 'income_statement', 'cash_flow', 'note'.")
		A=C._get_report(report_type=B,lang=F,mode=mode,style=style,get_all=get_all,show_log=show_log)
		if E is _B or E not in[_H,_D]:
			if B==_Q:A=C.duplicated_columns_handling(A)
			return A
		G=E
		if _J in A.columns:
			H=A[_J].astype(str).str.contains(G,case=_A,regex=_A);D=A[H].copy()
			if D.empty:logger.warning(f"Không tìm thấy kỳ báo cáo {G} trong cột report_period.")
			if B==_Q:D=C.duplicated_columns_handling(D)
			return D
		else:
			logger.error('Không thể lọc theo kỳ báo cáo: Không tìm thấy cột report_period.')
			if B==_Q:A=C.duplicated_columns_handling(A)
			return A
	@agg_execution(_L)
	def balance_sheet(self,period=_B,lang=_F,mode=_G,style=_C,get_all=_A,dropna=_E,show_log=_A):
		"\n        Trích xuất dữ liệu bảng cân đối kế toán cho một công ty từ nguồn VCI.\n\n        Tham số:\n            - period (str): Kỳ báo cáo tài chính. Mặc định là None.\n            - lang (str): Ngôn ngữ của báo cáo. Mặc định là 'en'.\n            - mode (str): Chế độ trả về dữ liệu. Mặc định là 'final' cho dữ liệu đã qua xử lý tên, nếu mode='raw' thì trả về DataFrame dữ liệu thô cho lưu trữ cơ sở dữ liệu.\n            - style (str): Chế độ hiển thị tên cột. Mặc định là 'readable' cho tên hiển thị, hoặc 'code' cho tên mã dạng snake_case không dấu.\n            - get_all (bool): Có lấy tất cả các cột hay không. Mặc định là False để lấy các cột quan trọng.\n            - dropna (bool): Có loại bỏ các cột với tất cả giá trị 0 hay không. Mặc định là False.\n            - show_log (bool): Hiển thị thông tin log hoặc không. Mặc định là False.\n            \n        Returns:\n            pd.DataFrame: DataFrame chứa dữ liệu bảng cân đối kế toán.\n        ";B='year_period';A=self._get_financial_report(report_type=_Q,period=period,lang=lang,mode=mode,style=style,get_all=get_all,dropna=dropna,show_log=show_log)
		if B in A.columns:A=A.drop(columns=B)
		return A
	@agg_execution(_L)
	def income_statement(self,period=_B,lang=_F,mode=_G,style=_C,get_all=_A,dropna=_E,show_log=_A):"\n        Trích xuất dữ liệu báo cáo kết quả kinh doanh cho một công ty từ nguồn VCI.\n\n        Tham số:\n            - period (str): Kỳ báo cáo tài chính. Mặc định là None.\n            - lang (str): Ngôn ngữ của báo cáo. Mặc định là 'en'.\n            - mode (str): Chế độ trả về dữ liệu. Mặc định là 'final' cho dữ liệu đã qua xử lý tên, nếu mode='raw' thì trả về DataFrame dữ liệu thô cho lưu trữ cơ sở dữ liệu.\n            - style (str): Chế độ hiển thị tên cột. Mặc định là 'readable' cho tên hiển thị, hoặc 'code' cho tên mã dạng snake_case không dấu.\n            - get_all (bool): Có lấy tất cả các cột hay không. Mặc định là False để lấy các cột quan trọng.\n            - dropna (bool): Có loại bỏ các cột với tất cả giá trị 0 hay không. Mặc định là False.\n            - show_log (bool): Hiển thị thông tin log hoặc không. Mặc định là False.\n            \n        Returns:\n            pd.DataFrame: DataFrame chứa dữ liệu báo cáo kết quả kinh doanh.\n        ";return self._get_financial_report(report_type=_W,period=period,lang=lang,mode=mode,style=style,get_all=get_all,dropna=dropna,show_log=show_log)
	@agg_execution(_L)
	def cash_flow(self,period=_B,lang=_F,mode=_G,style=_C,get_all=_A,dropna=_E,show_log=_A):"\n        Trích xuất dữ liệu báo cáo lưu chuyển tiền tệ của công ty từ nguồn VCI.\n\n        Tham số:\n            - period (str): Kỳ báo cáo tài chính. Mặc định là None.\n            - lang (str): Ngôn ngữ của báo cáo. Mặc định là 'en'.\n            - mode (str): Chế độ trả về dữ liệu. Mặc định là 'final' cho dữ liệu đã qua xử lý tên, nếu mode='raw' thì trả về DataFrame dữ liệu thô cho lưu trữ cơ sở dữ liệu.\n            - style (str): Chế độ hiển thị tên cột. Mặc định là 'readable' cho tên hiển thị, hoặc 'code' cho tên mã dạng snake_case không dấu.\n            - get_all (bool): Có lấy tất cả các cột hay không. Mặc định là False để lấy các cột quan trọng.\n            - dropna (bool): Có loại bỏ các cột với tất cả giá trị 0 hay không. Mặc định là False.\n            - show_log (bool): Hiển thị thông tin log hoặc không. Mặc định là False.\n            \n        Returns:\n            pd.DataFrame: DataFrame chứa dữ liệu báo cáo lưu chuyển tiền tệ.\n        ";return self._get_financial_report(report_type=_X,period=period,lang=lang,mode=mode,style=style,get_all=get_all,dropna=dropna,show_log=show_log)
	@agg_execution(_L)
	def note(self,period=_B,lang=_F,mode=_G,style=_C,get_all=_A,dropna=_E,show_log=_A):"\n        Trích xuất dữ liệu thuyến minh báo cáo tài chính công ty từ nguồn VCI.\n\n        Tham số:\n            - period (str): Kỳ báo cáo tài chính. Mặc định là None.\n            - lang (str): Ngôn ngữ của báo cáo. Mặc định là 'en'.\n            - mode (str): Chế độ trả về dữ liệu. Mặc định là 'final' cho dữ liệu đã qua xử lý tên, nếu mode='raw' thì trả về DataFrame dữ liệu thô cho lưu trữ cơ sở dữ liệu.\n            - style (str): Chế độ hiển thị tên cột. Mặc định là 'readable' cho tên hiển thị, hoặc 'code' cho tên mã dạng snake_case không dấu.\n            - get_all (bool): Có lấy tất cả các cột hay không. Mặc định là False để lấy các cột quan trọng.\n            - dropna (bool): Có loại bỏ các cột với tất cả giá trị 0 hay không. Mặc định là False.\n            - show_log (bool): Hiển thị thông tin log hoặc không. Mặc định là False.\n            \n        Returns:\n            pd.DataFrame: DataFrame chứa dữ liệu thuyên minh báo cáo tài chính.\n        ";return self._get_financial_report(report_type='note',period=period,lang=lang,mode=mode,style=style,get_all=get_all,dropna=dropna,show_log=show_log)
	@agg_execution(_L)
	def ratio(self,period=_B,lang=_F,mode=_G,style=_C,get_all=_A,dropna=_E,show_log=_A):
		"\n        Trích xuất dữ liệu báo cáo tỷ lệ tài chính của công ty từ nguồn VCI.\n\n        Tham số:\n            - period (str): Kỳ báo cáo tài chính. Mặc định là None.\n            - lang (str): Ngôn ngữ của báo cáo. Mặc định là 'en'.\n            - mode (str): Chế độ trả về dữ liệu. Mặc định là 'final' cho dữ liệu đã qua xử lý tên, nếu mode='raw' thì trả về DataFrame dữ liệu thô cho lưu trữ cơ sở dữ liệu.\n            - style (str): Chế độ hiển thị tên cột. Mặc định là 'readable' cho tên hiển thị, hoặc 'code' cho tên mã dạng snake_case không dấu.\n            - get_all (bool): Có lấy tất cả các cột hay không. Mặc định là False để lấy các cột quan trọng.\n            - dropna (bool): Có loại bỏ các cột với tất cả giá trị 0 hay không. Mặc định là False.\n            - show_log (bool): Hiển thị thông tin log hoặc không. Mặc định là False.\n            \n        Returns:\n            pd.DataFrame: DataFrame chứa dữ liệu báo cáo tỷ lệ tài chính.\n        ";B=style;A=self._get_financial_report(report_type='ratio',period=period,lang=lang,mode=mode,style=_K,get_all=get_all,dropna=dropna,show_log=show_log)
		if B==_K:from app.lib.vnstock_data_alt.core.utils.parser import vn_to_snake_case as E;A.columns=[E(str(A))for A in A.columns]
		elif B==_C:
			from.const import RATIO_COLUMN_MAP_EN as F,RATIO_COLUMN_MAP_VI as G
			if lang==_N:C=G
			else:C=F
			A.columns=[C.get(str(A),str(A))for A in A.columns]
		for D in[_H,_D]:
			if D in A.columns:A=A.drop(columns=D)
		return A
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('financial','vci',Finance)