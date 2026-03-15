'\nModule quản lý thông tin công ty từ nguồn dữ liệu VCI.\n'
_Q='ownership_percent'
_P='subOrListingInfo'
_O='working'
_N='financialRatio'
_M='type'
_L='all'
_K='percentage'
_J='%Y-%m-%d'
_I='ms'
_H='update_date'
_G='ticker'
_F='first'
_E='en_'
_D='organ_code'
_C='__typename'
_B='symbol'
_A='VCI.ext'
import json,pandas as pd
from typing import Dict,Optional,Union,List
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib._vnstock_shared.core.utils.transform import clean_html_dict,flatten_dict_to_df,flatten_list_to_df,reorder_cols,drop_cols_by_pattern
from app.lib.vnstock_data_alt.core.utils.client import send_request
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
from app.lib._vnstock_shared.core.utils.parser import get_asset_type,camel_to_snake
from app.lib._vnstock_shared.compat import agg_execution
from app.lib.vnstock_alt.explorer.vci.const import _GRAPHQL_URL,_PRICE_INFO_MAP
import copy
logger=get_logger(__name__)
class Company:
	'\n    Class (lớp) quản lý các thông tin liên quan đến công ty từ nguồn dữ liệu VCI.\n\n    Tham số:\n        - symbol (str): Mã chứng khoán của công ty cần truy xuất thông tin.\n        - random_agent (bool): Sử dụng user-agent ngẫu nhiên hoặc không. Mặc định là False.\n        - to_df (bool): Chuyển đổi dữ liệu thành DataFrame hoặc không. Mặc định là True.\n        - show_log (bool): Hiển thị thông tin log hoặc không. Mặc định là False.\n    '
	def __init__(A,symbol,random_agent=False,to_df=True,show_log=False):
		'\n        Khởi tạo đối tượng Company với các tham số cho việc truy xuất dữ liệu.\n        ';B=show_log;A.symbol=symbol.upper();A.asset_type=get_asset_type(A.symbol)
		if A.asset_type not in['stock']:raise ValueError('Mã chứng khoán không hợp lệ. Chỉ cổ phiếu mới có thông tin.')
		A.headers=get_headers(data_source='VCI',random_agent=random_agent);A.show_log=B;A.to_df=to_df;A.raw_data=A._fetch_data()
		if not B:logger.setLevel('CRITICAL')
	def _fetch_data(A):
		'\n        Phương thức riêng để lấy dữ liệu công ty từ nguồn VCI.\n        \n        Returns:\n            Dict: Dữ liệu thô về công ty từ API.\n        ';C=_GRAPHQL_URL;B='{"query":"query Query($ticker: String!, $lang: String!) {\\n  AnalysisReportFiles(ticker: $ticker, langCode: $lang) {\\n    date\\n    description\\n    link\\n    name\\n    __typename\\n  }\\n  News(ticker: $ticker, langCode: $lang) {\\n    id\\n    organCode\\n    ticker\\n    newsTitle\\n    newsSubTitle\\n    friendlySubTitle\\n    newsImageUrl\\n    newsSourceLink\\n    createdAt\\n    publicDate\\n    updatedAt\\n    langCode\\n    newsId\\n    newsShortContent\\n    newsFullContent\\n    closePrice\\n    referencePrice\\n    floorPrice\\n    ceilingPrice\\n    percentPriceChange\\n    __typename\\n  }\\n  TickerPriceInfo(ticker: $ticker) {\\n    financialRatio {\\n      yearReport\\n      lengthReport\\n      updateDate\\n      revenue\\n      revenueGrowth\\n      netProfit\\n      netProfitGrowth\\n      ebitMargin\\n      roe\\n      roic\\n      roa\\n      pe\\n      pb\\n      eps\\n      currentRatio\\n      cashRatio\\n      quickRatio\\n      interestCoverage\\n      ae\\n      fae\\n      netProfitMargin\\n      grossMargin\\n      ev\\n      issueShare\\n      ps\\n      pcf\\n      bvps\\n      evPerEbitda\\n      at\\n      fat\\n      acp\\n      dso\\n      dpo\\n      epsTTM\\n      charterCapital\\n      RTQ4\\n      charterCapitalRatio\\n      RTQ10\\n      dividend\\n      ebitda\\n      ebit\\n      le\\n      de\\n      ccc\\n      RTQ17\\n      __typename\\n    }\\n    ticker\\n    exchange\\n    ev\\n    ceilingPrice\\n    floorPrice\\n    referencePrice\\n    openPrice\\n    matchPrice\\n    closePrice\\n    priceChange\\n    percentPriceChange\\n    highestPrice\\n    lowestPrice\\n    totalVolume\\n    highestPrice1Year\\n    lowestPrice1Year\\n    percentLowestPriceChange1Year\\n    percentHighestPriceChange1Year\\n    foreignTotalVolume\\n    foreignTotalRoom\\n    averageMatchVolume2Week\\n    foreignHoldingRoom\\n    currentHoldingRatio\\n    maxHoldingRatio\\n    __typename\\n  }\\n  Subsidiary(ticker: $ticker) {\\n    id\\n    organCode\\n    subOrganCode\\n    percentage\\n    subOrListingInfo {\\n      enOrganName\\n      organName\\n      __typename\\n    }\\n    __typename\\n  }\\n  Affiliate(ticker: $ticker) {\\n    id\\n    organCode\\n    subOrganCode\\n    percentage\\n    subOrListingInfo {\\n      enOrganName\\n      organName\\n      __typename\\n    }\\n    __typename\\n  }\\n  CompanyListingInfo(ticker: $ticker) {\\n    id\\n    issueShare\\n    en_History\\n    history\\n    en_CompanyProfile\\n    companyProfile\\n    icbName3\\n    enIcbName3\\n    icbName2\\n    enIcbName2\\n    icbName4\\n    enIcbName4\\n    financialRatio {\\n      id\\n      ticker\\n      issueShare\\n      charterCapital\\n      __typename\\n    }\\n    __typename\\n  }\\n  OrganizationManagers(ticker: $ticker) {\\n    id\\n    ticker\\n    fullName\\n    positionName\\n    positionShortName\\n    en_PositionName\\n    en_PositionShortName\\n    updateDate\\n    percentage\\n    quantity\\n    __typename\\n  }\\n  OrganizationShareHolders(ticker: $ticker) {\\n    id\\n    ticker\\n    ownerFullName\\n    en_OwnerFullName\\n    quantity\\n    percentage\\n    updateDate\\n    __typename\\n  }\\n  OrganizationResignedManagers(ticker: $ticker) {\\n    id\\n    ticker\\n    fullName\\n    positionName\\n    positionShortName\\n    en_PositionName\\n    en_PositionShortName\\n    updateDate\\n    percentage\\n    quantity\\n    __typename\\n  }\\n  OrganizationEvents(ticker: $ticker) {\\n    id\\n    organCode\\n    ticker\\n    eventTitle\\n    en_EventTitle\\n    publicDate\\n    issueDate\\n    sourceUrl\\n    eventListCode\\n    ratio\\n    value\\n    recordDate\\n    exrightDate\\n    eventListName\\n    en_EventListName\\n    __typename\\n  }\\n}\\n","variables":{"ticker":"VCI","lang":"vi"}}';B=json.loads(B);B['variables'][_G]=A.symbol
		if A.show_log:logger.debug(f"Requesting data for {A.symbol} from {C}. payload: {B}")
		D=send_request(url=C,headers=A.headers,method='POST',payload=B,show_log=A.show_log);return D['data']
	def _process_data(D,data,data_key,columns_dict=None):
		'\n        Xử lý dữ liệu công ty từ API của VCI và chuyển đổi thành DataFrame.\n\n        Tham số:\n            - data (Dict): Dữ liệu từ API của VCI.\n            - data_key (str): Khóa của dữ liệu cần xử lý.\n            - columns_dict (Dict, optional): Từ điển các cột cần đổi tên.\n\n        Returns:\n            pd.DataFrame: DataFrame đã xử lý.\n        ';B=columns_dict;C=data[data_key];A=pd.DataFrame(C);A.columns=[camel_to_snake(A)for A in A.columns]
		if B:A=A.rename(columns=B)
		return A
	def _parse_price_info(B):'\n        Phân tách dữ liệu giá và tỷ lệ tài chính từ thông tin giá.\n        \n        Returns:\n            tuple: (price_data, ratio_data) - Bộ dữ liệu giá và tỷ lệ tài chính.\n        ';A=copy.deepcopy(B.raw_data['TickerPriceInfo']);C=pd.DataFrame(A[_N],index=[0]);A.pop(_N);A=pd.DataFrame(A,index=[0]);return A,C
	@agg_execution(_A)
	def overview(self):'\n        Truy xuất thông tin tổng quan của công ty.\n        \n        Returns:\n            pd.DataFrame: DataFrame chứa thông tin tổng quan của công ty.\n        ';B=self.raw_data['CompanyListingInfo'];C=clean_html_dict(B);A=flatten_dict_to_df(C,_N);A=A.map(lambda x:x.replace('\n',' ')if isinstance(x,str)else x);A.columns=[camel_to_snake(A)for A in A.columns];A=drop_cols_by_pattern(A,[_E,'__','_ratio_id']);A=A.rename(columns={_G:_B});A=reorder_cols(A,[_B],position=_F);return A
	@agg_execution(_A)
	def shareholders(self):'\n        Truy xuất thông tin cổ đông của công ty.\n        \n        Returns:\n            pd.DataFrame: DataFrame chứa thông tin cổ đông của công ty.\n        ';A=self._process_data(self.raw_data,'OrganizationShareHolders');A=drop_cols_by_pattern(A,[_C,_G,_E]);A[_H]=pd.to_datetime(A[_H],unit=_I).dt.strftime(_J);A=A.rename(columns={'owner_full_name':'share_holder',_K:'share_own_percent'});return A
	@agg_execution(_A)
	def officers(self,filter_by=_O):
		"\n        Truy xuất thông tin lãnh đạo công ty.\n\n        Tham số:\n            - filter_by (str): Lọc lãnh đạo đang làm việc hoặc đã từ nhiệm hoặc tất cả.\n                - 'working': Lọc lãnh đạo đang làm việc.\n                - 'resigned': Lọc lãnh đạo đã từ nhiệm.\n                - 'all': Lọc tất cả lãnh đạo.\n\n        Returns:\n            pd.DataFrame: DataFrame chứa thông tin lãnh đạo của công ty.\n        ";H='OrganizationResignedManagers';G='OrganizationManagers';F='resigned';C=filter_by;B=self
		if C not in[_O,F,_L]:raise ValueError("filter_by chỉ nhận giá trị 'working' hoặc 'resigned' hoặc 'all'")
		if C==_O:A=B._process_data(B.raw_data,G)
		elif C==F:A=B._process_data(B.raw_data,H)
		else:D=B._process_data(B.raw_data,G);D[_M]='đang làm việc';E=B._process_data(B.raw_data,H);E[_M]='đã từ nhiệm';A=pd.concat([D,E])
		A=drop_cols_by_pattern(A,[_E,'__',_G]);A=reorder_cols(A,[_B],position=_F);A=A.rename(columns={'full_name':'officer_name','position_name':'officer_position',_K:'officer_own_percent'});A[_H]=pd.to_datetime(A[_H],unit=_I).dt.strftime(_J);return A
	@agg_execution(_A)
	def subsidiaries(self,filter_by=_L):
		"\n        Truy xuất thông tin công ty con của công ty.\n\n        Tham số:\n            - filter_by (str): Lọc công ty con hoặc công ty liên kết.\n                - 'all': Lọc tất cả.\n                - 'subsidiary': Lọc công ty con.\n                - 'affiliate': Lọc công ty liên kết.\n\n        Returns:\n            pd.DataFrame: DataFrame chứa thông tin công ty con.\n        ";D='subsidiary';B=filter_by
		if B not in[_L,D]:raise ValueError("filter_by chỉ nhận giá trị 'all' hoặc 'subsidiary'")
		A=self.raw_data['Subsidiary'];A=flatten_list_to_df(A,_P);A.columns=[camel_to_snake(A)for A in A.columns];A=drop_cols_by_pattern(A,[_C,_E])
		if _D in A.columns:A=A.drop(columns=[_D])
		A=A.rename(columns={_K:_Q});A[_M]='công ty con'
		if B==D:return A
		elif B==_L:C=self.affiliate();C[_M]='công ty liên kết';E=pd.concat([A,C]);return E
	@agg_execution(_A)
	def affiliate(self):
		'\n        Truy xuất thông tin công ty liên kết của công ty.\n        \n        Returns:\n            pd.DataFrame: DataFrame chứa thông tin công ty liên kết.\n        ';B=self.raw_data['Affiliate'];A=flatten_list_to_df(B,_P);A.columns=[camel_to_snake(A)for A in A.columns];A=drop_cols_by_pattern(A,[_E,_C])
		if _D in A.columns:A=A.drop(columns=[_D])
		A=reorder_cols(A,['id','sub_organ_code','organ_name'],position=_F);A=A.rename(columns={_K:_Q});return A
	@agg_execution(_A)
	def news(self):
		'\n        Truy xuất tin tức liên quan đến công ty.\n        \n        Returns:\n            pd.DataFrame: DataFrame chứa tin tức liên quan đến công ty.\n        ';A=self._process_data(self.raw_data,'News')
		for(B,C)in _PRICE_INFO_MAP.items():
			if B in A.columns:A=A.rename(columns={B:C})
		A=A.drop(columns=[_D,_B,_C]);return A
	@agg_execution(_A)
	def events(self):
		'\n        Truy xuất các sự kiện của công ty.\n        \n        Returns:\n            pd.DataFrame: DataFrame chứa các sự kiện của công ty.\n        ';A=self._process_data(self.raw_data,'OrganizationEvents')
		for(B,C)in _PRICE_INFO_MAP.items():
			if B in A.columns:A=A.rename(columns={B:C})
		A=A.drop(columns=[_D,_B,_C]);D=['public_date','issue_date','record_date','exright_date']
		for B in D:
			if B in A.columns:A[B]=pd.to_datetime(A[B],unit=_I).dt.strftime(_J)
		return A
	@agg_execution(_A)
	def reports(self):
		'\n        Truy xuất báo cáo phân tích về công ty.\n        \n        Returns:\n            pd.DataFrame: DataFrame chứa các báo cáo phân tích về công ty.\n        ';B='date';A=self._process_data(self.raw_data,'AnalysisReportFiles')
		if _C in A.columns:A=A.drop(columns=[_C])
		if B in A.columns and A[B].dtype in[int,float]:A[B]=pd.to_datetime(A[B],unit=_I).dt.strftime(_J)
		return A
	@agg_execution(_A)
	def trading_stats(self):
		'\n        Truy xuất thống kê giao dịch của công ty.\n        \n        Returns:\n            pd.DataFrame: DataFrame chứa thống kê giao dịch của công ty.\n        ';C,E=self._parse_price_info();A=pd.DataFrame(C);A.columns=[camel_to_snake(A)for A in A.columns];A=drop_cols_by_pattern(A,[_C])
		for(B,D)in _PRICE_INFO_MAP.items():
			if B in A.columns:A=A.rename(columns={B:D})
		A[_B]=self.symbol;A=reorder_cols(A,[_B],position=_F);return A
	@agg_execution(_A)
	def ratio_summary(self):'\n        Truy xuất tóm tắt các tỷ lệ tài chính của công ty.\n        \n        Returns:\n            pd.DataFrame: DataFrame chứa tóm tắt các tỷ lệ tài chính của công ty.\n        ';C,B=self._parse_price_info();B.columns=[camel_to_snake(A)for A in B.columns];A=drop_cols_by_pattern(B,[_C]);A[_B]=self.symbol;A=reorder_cols(A,cols=[_B],position=_F);return A
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('company','vci',Company)