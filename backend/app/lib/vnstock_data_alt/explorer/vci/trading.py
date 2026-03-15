_l='accumulated_value'
_k='accumulated_volume'
_j='organ_code'
_i='time_frame'
_h='listingInfo'
_g='coerce'
_f='content'
_e='foreign'
_d='toDate'
_c='fromDate'
_b='ONE_DAY'
_a='timeFrame'
_Z='close_price'
_Y='matchPrice'
_X='HSX'
_W='bidAsk'
_V='size'
_U='page'
_T='1D'
_S='stock'
_R='source'
_Q='ticker'
_P='session'
_O='symbols'
_N='volume'
_M='price'
_L='data'
_K='VCI'
_J=True
_I='trading_date'
_H='symbol'
_G='VCI.ext'
_F='HOSE'
_E='exchange'
_D='time'
_C=False
_B='listing'
_A=None
from typing import List,Optional,Union,Dict,Any
import json,time
from datetime import datetime
import pandas as pd,requests
from app.lib._vnstock_shared.compat import agg_execution
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib._vnstock_shared.core.utils.parser import get_asset_type,camel_to_snake,flatten_data
from app.lib._vnstock_shared.core.utils.transform import flatten_hierarchical_index
from app.lib.vnstock_alt.explorer.vci.const import _TRADING_URL
from app.lib.vnstock_data_alt.core.utils.client import ProxyConfig,send_request
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
from app.lib.vnstock_data_alt.core.utils.parser import filter_columns_by_language
from app.lib.vnstock_data_alt.core.utils.validation import validate_date
from app.lib.vnstock_data_alt.explorer.vci.const import _VCIQ_URL,_REPORT_RESOLUTION,_ODD_LOT_URL,_PUT_THROUGH_URL,_ODD_LOT_MAP,_PUT_THROUGH_MAP,_ODD_LOT_STANDARD_COLUMNS,_PUT_THROUGH_STANDARD_COLUMNS,_STOCK_BOARD_STANDARD_COLUMNS,_PRICE_BOARD_STANDARD_COLUMNS,_VCI_TO_SCHEMA_MAP
logger=get_logger(__name__)
class Trading:
	'\n    Truy xuất dữ liệu giao dịch của mã chứng khoán từ nguồn dữ liệu VCI.\n    '
	def __init__(A,symbol=_A,random_agent=_C,proxy_config=_A,show_log=_C):
		D=show_log;C=proxy_config;B=symbol;A.symbol=B.upper()if B else'';A.asset_type=get_asset_type(A.symbol)if A.symbol else _A;A.base_url=_VCIQ_URL;A.headers=get_headers(data_source=_K,random_agent=random_agent);A.proxy_config=C if C is not _A else ProxyConfig()
		if not D:logger.setLevel('CRITICAL')
		A.show_log=D
	def _process_dates(C,start,end):
		'\n        Validate and process start/end dates for API requests.\n        \n        Args:\n            start (str, optional): Start date in YYYY-mm-dd format\n            end (str, optional): End date in YYYY-mm-dd format\n            \n        Returns:\n            tuple: Processed dates in YYYYMMDD format or (None, None) if invalid\n        ';B=end;A=start
		if A and B:
			if not validate_date(A)or not validate_date(B):logger.error('Invalid date format. Please use the format YYYY-mm-dd.');return _A,_A
			else:return A.replace('-',''),B.replace('-','')
		return _A,_A
	def _fetch_data(A,endpoint,params):
		"\n        Core function to fetch data from VCI API endpoints.\n        \n        Args:\n            endpoint (str): API endpoint path (e.g., 'price-history-summary')\n            params (dict): Query parameters for the API request\n            \n        Returns:\n            dict: Raw JSON response from VCI API\n            \n        Raises:\n            requests.exceptions.RequestException: For HTTP-related errors\n            ValueError: For JSON parsing errors\n        ";B=endpoint;C=f"{A.base_url}/v1/company/{A.symbol}/{B}"
		try:
			D=send_request(C,headers=A.headers,method='GET',params=params,show_log=A.show_log,proxy_list=A.proxy_config.proxy_list,proxy_mode=A.proxy_config.proxy_mode,request_mode=A.proxy_config.request_mode)
			if A.show_log:logger.info(f"Successfully fetched data from {B} for {A.symbol}")
			return D
		except Exception as E:logger.error(f"Error fetching data from {B} for {A.symbol}: {E}");raise
	def _to_dataframe(D,data,data_path=_A):
		"\n        Convert API response data to pandas DataFrame with flexible data path extraction.\n        \n        Args:\n            data (dict): Raw JSON response from API\n            data_path (list, optional): Path to extract data from nested structure \n                                       (e.g., ['data'] or ['data', 'content'])\n                                       If None, defaults to ['data']\n            \n        Returns:\n            pd.DataFrame: Processed DataFrame with snake_case columns\n        ";B=data_path
		try:
			if B is _A:B=[_L]
			A=data
			for E in B:
				if isinstance(A,dict)and E in A:A=A[E]
				else:logger.warning(f"Data path {B} not found in response for {D.symbol}");return pd.DataFrame()
			if not A:logger.warning(f"No data found at path {B} for {D.symbol}");return pd.DataFrame()
			if isinstance(A,list):C=pd.DataFrame(A)
			elif isinstance(A,dict):C=pd.DataFrame([A])
			else:logger.error(f"Unexpected data type at path {B}: {type(A)}");return pd.DataFrame()
			if not C.empty:C.columns=[camel_to_snake(A)for A in C.columns]
			return C
		except(KeyError,TypeError,Exception)as F:logger.error(f"Unexpected error processing data for {D.symbol}: {F}");raise
	def _process_bid_ask_data(F,item,row):
		'\n        Extract and process bid/ask prices and volumes from item data.\n        \n        Args:\n            item (dict): Raw item data from API response\n            row (dict): Row dictionary to populate with bid/ask data\n        ';B=row
		try:
			C=item.get(_W,{});G=C.get('bidPrices',[])
			for(A,D)in enumerate(G,start=1):B[f"bidAsk_bid_{A}_price"]=D.get(_M);B[f"bidAsk_bid_{A}_volume"]=D.get(_N)
			H=C.get('askPrices',[])
			for(A,E)in enumerate(H,start=1):B[f"bidAsk_ask_{A}_price"]=E.get(_M);B[f"bidAsk_ask_{A}_volume"]=E.get(_N)
		except(KeyError,TypeError,AttributeError)as I:logger.debug(f"Error processing bid/ask data for {F.symbol}: {I}")
	def _normalize_exchange_code(B,df,column_name):
		'\n        Normalize exchange codes (HSX → HOSE) for consistency.\n        \n        Args:\n            df (pd.DataFrame): DataFrame to normalize\n            column_name (str): Column name to normalize\n        ';A=column_name
		if A in df.columns:df[A]=df[A].map(lambda x:_F if x==_X else x)
	def _flatten_price_board_columns(B,df,separator='_',drop_levels=_A):'\n        Flatten hierarchical columns in price board DataFrame.\n        \n        Args:\n            df (pd.DataFrame): DataFrame with MultiIndex columns\n            separator (str): Separator for flattened column names\n            drop_levels (int or list): Levels to drop during flattening\n            \n        Returns:\n            pd.DataFrame: DataFrame with flattened columns\n        ';A=flatten_hierarchical_index(df,separator=separator,drop_levels=drop_levels,handle_duplicates=_J);B._normalize_exchange_code(A,'listing_exchange');return A
	def _fetch_stock_board(C,symbols_list,show_log=_C,flatten_columns=_J,separator='_',drop_levels=_A):
		"\n        Internal method to fetch stock board (lô chẵn) data.\n        \n        Args:\n            symbols_list (List[str]): List of stock symbols to fetch\n            show_log (bool): Show logging information (default: False)\n            flatten_columns (bool): Flatten hierarchical columns (default: True)\n            separator (str): Separator for flattened column names (default: '_')\n            drop_levels (int or list): Levels to drop during flattening (default: None)\n            \n        Returns:\n            pd.DataFrame: Stock board data with normalized columns\n        ";L=show_log;K=symbols_list;J='message_type';I='received_time';H='code';D='match';B='bid_ask';M=f"{_TRADING_URL}price/symbols/getList";N=json.dumps({_O:K})
		if L:logger.info(f"Requested URL: {M} with query payload: {N}")
		try:O=send_request(M,headers=C.headers,method='POST',payload=json.loads(N),show_log=L,proxy_list=C.proxy_config.proxy_list,proxy_mode=C.proxy_config.proxy_mode,request_mode=C.proxy_config.request_mode)
		except Exception as F:logger.error(f"Failed to fetch price board data: {F}");raise
		if not O:raise ConnectionError('Tải dữ liệu không thành công hoặc không có dữ liệu trả về.')
		G=[]
		for E in O:
			try:Q={_B:E.get(_h,{}),_W:E.get(_W,{}),D:E.get(_Y,{})};P=flatten_data(Q);C._process_bid_ask_data(E,P);G.append(P)
			except Exception as F:logger.warning(f"Error processing item in price_board: {F}");continue
		if not G:logger.warning(f"No valid data rows found for symbols: {K}");return pd.DataFrame()
		A=pd.DataFrame(G);A.columns=pd.MultiIndex.from_tuples([tuple(camel_to_snake(A)for A in A.split('_',1))for A in A.columns]);R=[(B,H),(B,_H),(B,_P),(B,I),(B,J),(B,_D),(B,'bid_prices'),(B,'ask_prices'),(_B,H),(_B,'exercise_price'),(_B,'exercise_ratio'),(_B,'maturity_date'),(_B,'underlying_symbol'),(_B,'issuer_name'),(_B,I),(_B,J),(_B,'en_organ_name'),(_B,'en_organ_short_name'),(_B,'organ_short_name'),(_B,_Q),(D,H),(D,_H),(D,I),(D,J),(D,_D),(D,_P)];A=A.drop(columns=[B for B in R if B in A.columns]);A=A.rename(columns={'board':_E},level=1);C._normalize_exchange_code(A,(_B,_E))
		if flatten_columns:A=C._flatten_price_board_columns(A,separator=separator,drop_levels=drop_levels)
		A.attrs[_R]=_K;return A
	def _normalize_unified_data(K,df):
		'\n        Normalize VCI unified format data - standardize data types and units.\n        \n        Data Type Standards (semantic correctness):\n        - All prices (open, high, low, close, ceiling, floor, reference, average, bid/ask): float64\n          (Rationale: API provides with decimal precision, maintaining accuracy)\n        - All volumes (bid_vol, ask_vol, total_trades, foreign_volumes): int64\n          (Rationale: Share counts are integers)\n        - total_value: float64, scaled to VND from millions\n          (VCI provides in triệu đ, scale ×1,000,000 to match KBS in VND)\n        - Percentages (percent_change): float64\n        - Identifiers (symbol, exchange): str\n        - Timestamp (time): int64\n        \n        Unit Conversion:\n        - total_value: 771888.25 (triệu đ) → 771888.25 × 1,000,000 = 771,888,250,000 (VND)\n        \n        Args:\n            df (pd.DataFrame): VCI unified format DataFrame (after column renaming to flat structure)\n            \n        Returns:\n            pd.DataFrame: Normalized DataFrame with consistent data types\n        ';G='price_change';F='reference_price';E='float64';D='total_value';A=df
		if A is _A or A.empty:return A
		A=A.copy()
		if isinstance(A.columns,pd.MultiIndex):A.columns=A.columns.map(lambda x:'_'.join(str(A)for A in x)if isinstance(x,tuple)else x)
		if D in A.columns:
			try:A[D]=(A[D]*1000000).astype(E)
			except Exception as B:logger.warning(f"Could not scale total_value: {B}")
		try:
			H=[A for A in A.columns if isinstance(A,str)and _M in A.lower()]
			for C in H:
				if C in A.columns:A[C]=A[C].astype(E)
		except Exception as B:logger.warning(f"Could not normalize price columns: {B}")
		try:
			I=[A for A in A.columns if isinstance(A,str)and('vol'in A.lower()or _N in A.lower()or'trades'in A.lower())]
			for C in I:
				if C in A.columns and A[C].dtype!='object':A[C]=A[C].astype('int64')
		except Exception as B:logger.warning(f"Could not normalize volume columns: {B}")
		if F in A.columns and _Z in A.columns:
			try:A[G]=(A[_Z]-A[F]).astype(E);A['percent_change']=(A[G]/A[F]*100).round(2)
			except Exception as B:logger.warning(f"Could not calculate price_change/percent_change: {B}")
		try:J=int(time.time()*1000);A[_D]=J
		except Exception as B:logger.warning(f"Could not add time column: {B}")
		return A
	@agg_execution(_G)
	def price_board(self,symbols_list,board=_S,exchange=_F,show_log=_C,get_all=_C,get_unified=_C):
		"\n        Truy xuất bảng giá realtime cho danh sách mã chứng khoán.\n\n        Unified interface để lấy dữ liệu giá từ ba loại bảng giá:\n        - stock: Lô chẵn (giao dịch thông thường)\n        - odd_lot: Lô lẻ (giao dịch lô lẻ)\n        - put_through: Thỏa thuận (giao dịch thỏa thuận)\n\n        Args:\n            symbols_list (List[str]): Danh sách mã chứng khoán (VD: ['ACB', 'VNM', 'HPG']).\n            board (str): Loại bảng giá ('stock', 'odd_lot', 'put_through'). Mặc định 'stock'.\n            exchange (str): Sàn giao dịch ('HOSE', 'HNX', 'UPCOM'). Mặc định 'HOSE'.\n            show_log (bool): Hiển thị log debug.\n            get_all (bool): Nếu True, trả về tất cả các cột. Nếu False (mặc định), chỉ trả về các cột tiêu chuẩn.\n            get_unified (bool): Nếu True, transform VCI's prefixed columns sang unified schema (khớp với KBS).\n                               Nếu False (mặc định), trả về VCI's native format (backward compatible).\n\n        Returns:\n            pd.DataFrame: DataFrame chứa thông tin giá realtime.\n            - get_unified=False (default): VCI's native format với prefixed columns (listing_*, match_*, bid_ask_*)\n            - get_unified=True: Unified schema khớp với KBS (symbol, time, exchange, close_price, ...)\n\n        Examples:\n            >>> trading = Trading('ACB')\n            >>> df = trading.price_board(['ACB', 'VNM', 'HPG'])  # VCI native format\n            >>> df = trading.price_board(['ACB', 'VNM', 'HPG'], get_unified=True)  # Unified with KBS\n            >>> df = trading.price_board(['AAA', 'AAM'], board='odd_lot')  # Odd-lot board\n            >>> df = trading.price_board(['SCR'], board='put_through')  # Put-through board\n\n        Raises:\n            ValueError: Nếu symbols_list trống hoặc board không hợp lệ.\n        ";L='odd_lot';J=get_all;H=get_unified;G=exchange;E=show_log;D=board;C=self;B=symbols_list
		if not B:raise ValueError('symbols_list không được để trống.')
		K=[_S,L,'put_through']
		if D not in K:raise ValueError(f"board không hợp lệ. Các giá trị hợp lệ: {K}")
		B=[A.upper()for A in B]
		if D==_S:A=C._fetch_stock_board(B,show_log=E,flatten_columns=_C);I='lô chẵn';F=_STOCK_BOARD_STANDARD_COLUMNS
		elif D==L:A=C.odd_lot(symbols_list=B,exchange=G,show_log=E);I='lô lẻ';F=_ODD_LOT_STANDARD_COLUMNS
		else:
			A=C.put_through(exchange=G,show_log=E)
			if len(A)>0 and B:A=A[A[_H].isin(B)].reset_index(drop=_J)
			I='thỏa thuận';F=_PUT_THROUGH_STANDARD_COLUMNS
		if len(A)>0:
			if H and D==_S:
				from app.lib.vnstock_data_alt.explorer.vci.const import _VCI_TO_SCHEMA_MAP as M
				if isinstance(A.columns,pd.MultiIndex):A.columns=A.columns.map(lambda x:f"{x[0]}_{x[1]}"if isinstance(x,tuple)else x)
				A=A.rename(columns=M);A=C._normalize_unified_data(A);F=_PRICE_BOARD_STANDARD_COLUMNS
				if not J:N=[B for B in F if B in A.columns];A=A[N]
			else:0
			if _E in A.columns:A[_E]=A[_E].map(lambda x:_F if x==_X else x if pd.notna(x)else x)
			elif(_B,_E)in A.columns:A[_B,_E]=A[_B,_E].map(lambda x:_F if x==_X else x if pd.notna(x)else x)
		A.attrs[_O]=B;A.attrs['board']=D;A.attrs[_E]=G;A.attrs[_R]=_K;A.attrs['get_all']=J;A.attrs['get_unified']=H
		if E or C.show_log:O='unified'if H else'native';logger.info(f"Truy xuất thành công bảng giá {I} ({O}) cho {len(B)} mã chứng khoán.")
		return A
	@agg_execution(_G)
	def summary(self,resolution=_T,start=_A,end=_A,limit=100):
		"\n        Truy xuất thống kê giao dịch của mã chứng khoán được chọn.\n        \n        Args:\n            resolution (str): Time resolution for data (default: '1D')\n            start (str, optional): Start date in YYYY-mm-dd format\n            end (str, optional): End date in YYYY-mm-dd format\n            limit (int): Maximum number of records to return (default: 100)\n            \n        Returns:\n            pd.DataFrame: Trading summary data as DataFrame with snake_case columns\n        ";B=self;C,D=B._process_dates(start,end);E={_a:_REPORT_RESOLUTION.get(resolution,_b),_U:0,_V:limit}
		if C and D:E.update({_c:C,_d:D})
		F=B._fetch_data('price-history-summary',E);A=B._to_dataframe(F,[_L])
		if not A.empty and any(_e in A for A in A.columns):A.columns=A.columns.str.replace(_e,'fr',regex=_C)
		return A
	@agg_execution(_G)
	def price_history(self,resolution=_T,start=_A,end=_A,get_all=_C,limit=100):
		"\n        Retrieve price history data for the selected stock.\n        \n        Args:\n            resolution (str): Time resolution for data (default: '1D')\n            start (str, optional): Start date in YYYY-mm-dd format\n            end (str, optional): End date in YYYY-mm-dd format\n            get_all (bool, optional): Whether to get all records, including data for foreign trades, or not (default: False)\n            limit (int): Maximum number of records to return (default: 100)\n            \n        Returns:\n            pd.DataFrame: Price history data as DataFrame with snake_case columns\n        ";B=self;C,D=B._process_dates(start,end);E={_a:_REPORT_RESOLUTION.get(resolution,_b),_U:0,_V:limit}
		if C and D:E.update({_c:C,_d:D})
		F=B._fetch_data('price-history',E);A=B._to_dataframe(F,[_L,_f]);A.columns=A.columns.str.replace(_e,'fr',regex=_C);G=['id',_Q,'stock_type',_i];H={'open_price':'open',_Z:'close','highest_price':'high','lowest_price':'low','total_match_volume':'matched_volume','total_match_value':'matched_value','total_deal_volume':'deal_volume','total_deal_value':'deal_value'};A=A.drop(columns=[B for B in G if B in A.columns])
		if _I in A.columns:A[_I]=pd.to_datetime(A[_I])
		if get_all is _C:A=A.drop(columns=[A for A in A.columns if'fr_'in A])
		A=A.rename(columns={B:C for(B,C)in H.items()if B in A.columns});return A
	@agg_execution(_G)
	def foreign_trade(self,resolution=_T,start=_A,end=_A,limit=100):A=self.price_history(resolution=resolution,start=start,end=end,get_all=_J,limit=limit);A=A[[_I]+[A for A in A.columns if A.startswith('fr_')]];return A
	@agg_execution(_G)
	def prop_trade(self,resolution=_T,start=_A,end=_A,limit=100):
		"\n        Retrieve proprietary trading history for the selected stock.\n        \n        Args:\n            resolution (str): Time resolution for data (default: '1D')\n            start (str, optional): Start date in YYYY-mm-dd format\n            end (str, optional): End date in YYYY-mm-dd format\n            limit (int): Maximum number of records to return (default: 100)\n            \n        Returns:\n            pd.DataFrame: Proprietary trading data as DataFrame with snake_case columns\n        ";B=self;C,D=B._process_dates(start,end);E={_a:_REPORT_RESOLUTION.get(resolution,_b),_U:0,_V:limit}
		if C and D:E.update({_c:C,_d:D})
		F=B._fetch_data('proprietary-history',E);A=B._to_dataframe(F,[_L,_f]);A.columns=A.columns.str.replace('proprietary','prop',regex=_C);G=['id',_Q,_j,_i];A=A.drop(columns=[B for B in G if B in A.columns])
		if _I in A.columns:
			try:A[_I]=pd.to_datetime(A[_I])
			except Exception as H:logger.warning(f"Failed to convert trading_date to datetime: {H}")
		A=A.dropna(how='all');A=A.reset_index(drop=_J);return A
	@agg_execution(_G)
	def insider_deal(self,limit=100,lang='vi'):
		'\n        Retrieve insider transaction data for the selected stock.\n        \n        Args:\n            limit (int): Maximum number of records to return (default: 100)\n            \n        Returns:\n            pd.DataFrame: Insider transaction data as DataFrame with snake_case columns\n        ';C={_U:0,_V:limit};D=self._fetch_data('insider-transaction',C);A=self._to_dataframe(D,[_L,_f]);A=filter_columns_by_language(A,lang=lang);E=['id',_Q,_j,'display_date1','display_date2','event_code','action_type_code','icb_code_lv1'];A=A.drop(columns=[B for B in E if B in A.columns]);F=['start_date','end_date','public_date']
		for B in F:
			if B in A.columns:
				try:A[B]=pd.to_datetime(A[B],errors=_g)
				except Exception as G:logger.warning(f"Failed to convert {B} to datetime: {G}")
		A=A.dropna(how='all');A=A.reset_index(drop=_J);return A
	@agg_execution(_G)
	def odd_lot(self,symbols_list=_A,exchange=_F,show_log=_C):
		"\n        Truy xuất dữ liệu giao dịch lô lẻ (odd-lot) cho danh sách mã chứng khoán.\n\n        Args:\n            symbols_list (List[str], optional): Danh sách mã chứng khoán. Nếu None, truy xuất toàn bộ sàn.\n            exchange (str): Sàn giao dịch ('HOSE', 'HNX', 'UPCOM'). Mặc định 'HOSE'.\n            show_log (bool): Hiển thị log debug.\n\n        Returns:\n            pd.DataFrame: Dữ liệu giao dịch lô lẻ với các cột chuẩn hóa.\n\n        Examples:\n            >>> trading = Trading('ACB')\n            >>> df = trading.odd_lot(symbols_list=['ACB', 'VNM'])\n            >>> df = trading.odd_lot(exchange='HOSE')\n\n        Raises:\n            ValueError: Nếu exchange không hợp lệ.\n        ";O='lowest';N='highest';F=show_log;E=exchange;D=self;C=symbols_list;K=[_F,'HNX','UPCOM']
		if E not in K:raise ValueError(f"Exchange không hợp lệ. Các giá trị hợp lệ: {K}")
		L=_ODD_LOT_URL
		if C:C=[A.upper()for A in C];G={_O:C}
		else:G={_E:E}
		if F or D.show_log:logger.info(f"Requested URL: {L} with payload: {G}")
		try:H=send_request(L,headers=D.headers,method='POST',payload=json.loads(json.dumps(G)),show_log=F or D.show_log,proxy_list=D.proxy_config.proxy_list,proxy_mode=D.proxy_config.proxy_mode,request_mode=D.proxy_config.request_mode)
		except Exception as I:logger.error(f"Failed to fetch odd_lot data: {str(I)}");return pd.DataFrame()
		if not H or not isinstance(H,list):return pd.DataFrame()
		J=[]
		for M in H:
			try:
				B=M.get(_Y,{})
				if not B:continue
				P={_H:M.get(_h,{}).get(_H),_M:B.get(_Y),_N:B.get('matchVol'),N:B.get(N),O:B.get(O),'open':B.get('openPrice'),'avg_price':B.get('avgMatchPrice'),_k:B.get('accumulatedVolume'),_l:B.get('accumulatedValue'),_P:B.get(_P),_D:B.get(_D)};J.append(P)
			except Exception as I:logger.warning(f"Error processing odd_lot item: {I}");continue
		if not J:logger.warning(f"No valid odd_lot data found for exchange {E}");return pd.DataFrame()
		A=pd.DataFrame(J);Q=[B for B in _ODD_LOT_STANDARD_COLUMNS if B in A.columns];A=A[Q]
		if _D in A.columns:A[_D]=pd.to_datetime(A[_D],errors=_g)
		if C:A.attrs[_O]=C
		A.attrs[_E]=E;A.attrs[_R]=_K
		if F or D.show_log:
			if C:logger.info(f"Truy xuất thành công {len(A)} bản ghi giao dịch lô lẻ cho {len(C)} mã chứng khoán.")
			else:logger.info(f"Truy xuất thành công {len(A)} bản ghi giao dịch lô lẻ cho sàn {E}.")
		return A
	@agg_execution(_G)
	def put_through(self,exchange=_F,show_log=_C):
		"\n        Truy xuất dữ liệu giao dịch thỏa thuận (put-through) theo sàn.\n\n        Args:\n            exchange (str): Sàn giao dịch ('HOSE', 'HNX', 'UPCOM'). Mặc định 'HOSE'.\n            show_log (bool): Hiển thị log debug.\n\n        Returns:\n            pd.DataFrame: Dữ liệu giao dịch thỏa thuận với các cột chuẩn hóa.\n\n        Examples:\n            >>> trading = Trading('ACB')\n            >>> df = trading.put_through(exchange='HOSE')\n\n        Raises:\n            ValueError: Nếu exchange không hợp lệ.\n        ";E=show_log;D=exchange;C=self;I=[_F,'HNX','UPCOM']
		if D not in I:raise ValueError(f"Exchange không hợp lệ. Các giá trị hợp lệ: {I}")
		J=_PUT_THROUGH_URL;K={'group':D}
		if E or C.show_log:logger.info(f"Requested URL: {J} with params: {K}")
		try:F=send_request(J,headers=C.headers,method='GET',params=K,show_log=E or C.show_log,proxy_list=C.proxy_config.proxy_list,proxy_mode=C.proxy_config.proxy_mode,request_mode=C.proxy_config.request_mode)
		except Exception as G:logger.error(f"Failed to fetch put_through data: {str(G)}");return pd.DataFrame()
		if not F or not isinstance(F,list):return pd.DataFrame()
		H=[]
		for B in F:
			try:L={_H:B.get(_H),_M:B.get('ptMatchPrice'),_N:B.get('ptMatchVolume'),'change':B.get('ptChange'),'change_percent':B.get('ptChangePercent'),'match_value':B.get('ptMatchValue'),_k:B.get('ptAccumulatedVolume'),_l:B.get('ptAccumulatedValue'),_D:B.get(_D)};H.append(L)
			except Exception as G:logger.warning(f"Error processing put_through item: {G}");continue
		if not H:logger.warning(f"No valid put_through data found for exchange {D}");return pd.DataFrame()
		A=pd.DataFrame(H);M=[B for B in _PUT_THROUGH_STANDARD_COLUMNS if B in A.columns];A=A[M]
		if _D in A.columns:A[_D]=pd.to_datetime(A[_D],errors=_g)
		A.attrs[_E]=D;A.attrs[_R]=_K
		if E or C.show_log:logger.info(f"Truy xuất thành công {len(A)} bản ghi giao dịch thỏa thuận cho sàn {D}.")
		return A
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('trading','vci',Trading)