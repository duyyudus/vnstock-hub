_B='VNINDEX'
_A='VND.ext'
import requests,pandas as pd
from app.lib._vnstock_shared.compat import agg_execution
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib._vnstock_shared.core.utils.user_agent import get_headers
from app.lib.vnstock_data_alt.core.utils.parser import lookback_date
from app.lib.vnstock_data_alt.explorer.vnd.const import _INDEX_MAPPING
logger=get_logger(__name__)
class Market:
	"\n    Provides market insights, including P/E and P/B ratios over time.\n\n    Attributes:\n        index (str): Market index (e.g., 'VNINDEX', 'HNX').\n        base_url (str): Base URL for retrieving data.\n        headers (dict): HTTP headers for API requests.\n    "
	def __init__(A,index=_B,random_agent=False,show_log=False):
		A.index=A._index_validation(index);A.base_url='https://api-finfo.vndirect.com.vn/v4/ratios';A.headers=get_headers(data_source='VND',random_agent=random_agent)
		if not show_log:logger.setLevel('CRITICAL')
	def _index_validation(C,index):
		"\n        Validates the index input.\n\n        Parameters:\n            index (str): The index to validate. Valid indices are 'VNINDEX' and 'HNX'.\n\n        Returns:\n            str: The validated index.\n        ";B='VN30';A=index;A=A.upper()
		if A not in[_B,'HNXINDEX',B]:raise ValueError(f"Invalid index: {A}. Valid indices are: 'VNINDEX', 'HNX'")
		if A==B:return B
		else:return _INDEX_MAPPING[A]
	def _fetch_data(B,ratio_code,start_date):
		"\n        Fetches market ratio data from the API.\n\n        Parameters:\n            ratio_code (str): Ratio code ('PRICE_TO_BOOK' or 'PRICE_TO_EARNINGS').\n            start_date (str): Start date for fetching data (format: YYYY-MM-DD).\n\n        Returns:\n            pd.DataFrame: A DataFrame containing the report date and ratio values.\n        ";D='reportDate';C=ratio_code;G=f"{B.base_url}?q=ratioCode:{C}~code:{B.index}~reportDate:gte:{start_date}&sort=reportDate:desc&size=10000&fields=value,reportDate"
		try:
			logger.info(f"Fetching {C} data for index {B.index}...");E=requests.get(G,headers=B.headers);E.raise_for_status();F=E.json().get('data',[])
			if not F:logger.warning('No data returned from API.');return pd.DataFrame()
			A=pd.DataFrame(F);A[D]=pd.to_datetime(A[D]);A=A.rename(columns={'value':C.lower()});A=A.rename(columns={'price_to_earnings':'pe','price_to_book':'pb'});return A.set_index(D).sort_index()
		except requests.RequestException as H:logger.error(f"Failed to fetch data: {H}");return pd.DataFrame()
	@agg_execution(_A)
	def pe(self,duration='5Y'):"\n        Retrieves P/E (Price-to-Earnings) ratio data.\n\n        Parameters:\n            duration (str): Number of years to retrieve data back (e.g., '1Y', '5Y', '10Y').\n\n        Returns:\n            pd.DataFrame: A DataFrame containing the P/E ratio data.\n        ";A=lookback_date(duration);return self._fetch_data(ratio_code='PRICE_TO_EARNINGS',start_date=A)
	@agg_execution(_A)
	def pb(self,duration='5Y'):"\n        Retrieves P/B (Price-to-Book) ratio data.\n\n        Parameters:\n            duration (str): Number of years to retrieve data back (e.g., '1Y', '5Y', '10Y').\n\n        Returns:\n            pd.DataFrame: A DataFrame containing the P/B ratio data.\n        ";A=lookback_date(duration);return self._fetch_data(ratio_code='PRICE_TO_BOOK',start_date=A)
	@agg_execution(_A)
	def evaluation(self,duration='5Y'):
		"\n        Retrieves an overview of the market with both P/E and P/B ratios.\n\n        Parameters:\n            duration (str): Number of years to retrieve data back (e.g., '1Y', '5Y', '10Y').\n\n        Returns:\n            pd.DataFrame: A DataFrame containing P/E and P/B ratio data.\n        ";A=duration;E=lookback_date(A);B=self.pe(duration=A);C=self.pb(duration=A)
		if B.empty and C.empty:logger.warning('No data available for both P/E and P/B ratios.');return pd.DataFrame()
		D=pd.concat([B,C],axis=1);return D
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('market','vnd',Market)