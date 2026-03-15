'\nModule quản lý thông tin báo cáo tài chính từ nguồn dữ liệu MAS.\n'
_K="Kỳ báo cáo tài chính không hợp lệ. Chỉ chấp nhận 'year' hoặc 'quarter'."
_J='period'
_I='ignore'
_H='quarter'
_G='MAS.ext'
_F=False
_E='vi'
_D='year'
_C=True
_B=None
_A='year_period'
import json,pandas as pd
from typing import Optional
from app.lib.vnstock_data_alt.explorer.mas.const import _FINANCIAL_URL,_FINANCIAL_REPORT_MAP,_FINANCIAL_REPORT_PERIOD_MAP,SUPPORTED_LANGUAGES
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
from app.lib.vnstock_data_alt.core.utils.parser import get_asset_type
from app.lib._vnstock_shared.core.utils.transform import replace_in_column_names,flatten_hierarchical_index,reorder_cols
from app.lib._vnstock_shared.compat import agg_execution
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib._vnstock_shared.core.utils.client import send_request
from app.lib._vnstock_shared.core.utils.parser import camel_to_snake
logger=get_logger(__name__)
class Finance:
	"\n    Truy xuất thông tin báo cáo tài chính của một công ty theo mã chứng khoán từ nguồn dữ liệu MAS.\n\n    Tham số:\n        - symbol (str): Mã chứng khoán của công ty cần truy xuất thông tin.\n        - period (str): Chu kỳ báo cáo tài chính cần truy xuất. Mặc định là 'quarter'.\n        - get_all (bool): Trả về tất cả các trường dữ liệu hoặc chỉ các trường chọn lọc. Mặc định là True.\n        - show_log (bool): Hiển thị thông tin log hoặc không. Mặc định là True.\n    "
	def __init__(A,symbol,period=_H,get_all=_C,show_log=_C):
		'\n        Khởi tạo đối tượng Finance với các tham số cho việc truy xuất dữ liệu báo cáo tài chính.\n        ';C=show_log;B=period;A.symbol=symbol.upper();A.asset_type=get_asset_type(A.symbol);A.base_url=_FINANCIAL_URL;A.query_params='query{vsFinancialReportList(StockCode:"TARGET_SYMBOL",Type:"TARGET_TYPE",TermType:"TARGET_PERIOD"){_id,ID,TermCode,YearPeriod,Content{Values{Name,NameEn,Value}}}}';A.headers=get_headers(data_source='MAS');A.show_log=C
		if not C:logger.setLevel('CRITICAL')
		if B not in[_D,_H]:raise ValueError(_K)
		A.raw_period=B;A.period=_FINANCIAL_REPORT_PERIOD_MAP.get(B)
		if A.asset_type not in['stock']:raise ValueError('Mã chứng khoán không hợp lệ. Chỉ cổ phiếu mới có thông tin.')
		A.get_all=get_all
	def _flatten_content(G,content,lang):
		"\n        Flatten a nested content structure based on the language.\n\n        Args:\n            content: A nested data structure (typically a list with a dict containing key 'Values').\n            lang (str): 'en' or 'vi'; uses 'NameEn' for 'en' and 'Name' for 'vi'.\n\n        Returns:\n            dict: Mapping of selected (trimmed) names to their values.\n        ";F='Values';C=content
		if isinstance(C,list):
			for B in C:
				if isinstance(B,dict)and F in B:
					D={}
					for A in B[F]:
						if isinstance(A,dict):
							E=A.get('NameEn','').strip()if lang.lower()=='en'else A.get('Name','').strip()
							if E:D[E]=A.get('Value')
					return D
		return{}
	def _parse_nested_data(B,df,lang):"\n        Apply _flatten_content to the 'content' column, convert to DataFrame, and\n        concatenate with the original DataFrame (dropping 'content').\n\n        Args:\n            df (pd.DataFrame): DataFrame with a 'content' column.\n            lang (str): 'en' or 'vi'.\n\n        Returns:\n            pd.DataFrame: Merged DataFrame.\n        ";A='content';C=df[A].apply(lambda content:B._flatten_content(content,lang));D=pd.DataFrame(C.tolist());E=pd.concat([df.drop(A,axis=1,errors=_I),D],axis=1);return E
	def _clean_columns(F,df,period_type):
		'\n        Clean and reformat DataFrame columns by creating a \'period\' column based on period_type\n        and dropping unneeded columns.\n\n        For "year":\n          - Use \'year_period\' as \'period\'.\n          - Drop _id, id, term_code.\n        For "quarter":\n          - Create period as "year_period-term_code".\n          - Drop \'year_period\', _id, id, and term_code.\n\n        Args:\n            df (pd.DataFrame): DataFrame that may contain \'_id\', \'id\', \'term_code\', \'year_period\'.\n            period_type (str): "year" or "quarter".\n\n        Returns:\n            pd.DataFrame: Cleaned DataFrame.\n        ';C='term_code';B=period_type;A=df;B=B.lower()
		if B==_D:A[_J]=A[_A]if _A in A.columns else _B
		elif B==_H:
			if _A in A.columns and C in A.columns:A[_J]=A[_A].astype(str)+'-'+A[C].astype(str)
			else:A[_J]=_B
			if _A in A.columns:A=A.drop(_A,axis=1,errors=_I)
		else:raise ValueError("period_type must be either 'year' or 'quarter'")
		D=['_id','id',C];E=[B for B in D if B in A.columns];return A.drop(columns=E,errors=_I)
	def _get_report(B,report_type,period,lang,dropna,show_log):
		"\n        A general method to query and process a financial report.\n\n        Args:\n            report_type (str): Report type such as 'balance_sheet', 'income_statement',\n                               'cash_flow', 'ratio', or 'annual_plan'.\n            period (str): Either 'year' or 'quarter'. If None, defaults to the instance period.\n            lang (str): 'en' or 'vi'.\n            dropna (bool): Whether to drop columns with all NaN values.\n            show_log (bool): Whether to show log details during the request.\n\n        Returns:\n            pd.DataFrame: Processed DataFrame with report data.\n        ";D=lang;C=period
		if D not in SUPPORTED_LANGUAGES:raise ValueError("Ngôn ngữ không hợp lệ: '"+D+"'. Hỗ trợ: "+', '.join(SUPPORTED_LANGUAGES)+'.')
		if C is _B:F=B.raw_period;E=B.period
		elif C not in[_D,_H]:raise ValueError(_K)
		else:F=C;E=_FINANCIAL_REPORT_PERIOD_MAP.get(C)
		G=B.base_url+'financialReport';H=B.query_params.replace('TARGET_SYMBOL',B.symbol).replace('TARGET_TYPE',_FINANCIAL_REPORT_MAP[report_type]).replace('TARGET_PERIOD',E);I={'query':H};J=send_request(url=G,headers=B.headers,method='GET',params=I,payload=_B,show_log=show_log);A=pd.DataFrame(J);A.columns=[camel_to_snake(A)for A in A.columns]
		if A.empty:logger.warning('No data found for symbol '+str(B.symbol)+' in period '+str(E)+'.');return pd.DataFrame()
		A=B._parse_nested_data(A,D);A=B._clean_columns(A,period_type=F);A=reorder_cols(A,[_J],position='first')
		if dropna:A=A.dropna(axis=1,how='all')
		return A
	@agg_execution(_G)
	def balance_sheet(self,period=_B,lang=_E,dropna=_C,show_log=_F):'\n        Trích xuất dữ liệu bảng cân đối kế toán từ nguồn MAS.\n        ';return self._get_report('balance_sheet',period,lang,dropna,show_log)
	@agg_execution(_G)
	def income_statement(self,period=_B,lang=_E,dropna=_C,show_log=_F):'\n        Trích xuất dữ liệu báo cáo kết quả kinh doanh từ nguồn MAS.\n        ';return self._get_report('income_statement',period,lang,dropna,show_log)
	@agg_execution(_G)
	def cash_flow(self,period=_B,lang=_E,dropna=_C,show_log=_F):'\n        Trích xuất dữ liệu báo cáo lưu chuyển tiền tệ từ nguồn MAS.\n        ';return self._get_report('cash_flow',period,lang,dropna,show_log)
	@agg_execution(_G)
	def ratio(self,period=_B,lang=_E,dropna=_C,show_log=_F):'\n        Trích xuất dữ liệu báo cáo tỉ lệ tài chính từ nguồn MAS.\n        ';return self._get_report('ratio',period,lang,dropna,show_log)
	@agg_execution(_G)
	def annual_plan(self,period=_D,lang=_E,dropna=_C,show_log=_F):
		'\n        Trích xuất dữ liệu báo cáo dự kiến từ nguồn MAS.\n        ';B=period
		if B not in[_D]:raise ValueError('Báo cáo chỉ tiêu kế hoạch chỉ chấp nhận giá trị year.')
		A=self._get_report('annual_plan',B,lang,dropna,show_log)
		if _A in A.columns:A=A.drop(_A,axis=1,errors=_I)
		return A
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('financial','mas',Finance)