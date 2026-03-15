import pandas as pd
from typing import Dict,Any
from app.lib.vnstock_data_alt.core.utils.fetcher import Fetcher
from app.lib.vnstock_data_alt.core.utils.user_agent import HEADERS_MAPPING_SOURCE
from app.lib._vnstock_shared.core.utils.interval import normalize_interval
from.const import BASE_URL,INTERVALS
class SPLFetcher(Fetcher):
	'\n    Specific fetcher for SPL data provider.\n    '
	def __init__(A):'\n        Initialize the SPLFetcher with SPL-specific settings.\n        ';super().__init__(base_url=BASE_URL,headers=HEADERS_MAPPING_SOURCE.get('SIMPLIZE',{}))
	def validate(D,params):
		'\n        Validate SPL-specific query parameters.\n\n        Parameters:\n            params (dict): Query parameters to validate.\n        \n        Raises:\n            ValueError: If required parameters are missing or invalid.\n        ';B=params
		if not B.get('ticker'):raise ValueError('Ticker is required for SPL requests.')
		A=B.get('interval','1D')
		if A not in INTERVALS:
			try:normalize_interval(A)
			except ValueError:C=', '.join(sorted(INTERVALS));raise ValueError(f"Invalid interval '{A}'. Supported: {C}")
	def to_dataframe(D,raw_data):"\n        Convert SPL-specific raw data to a Pandas DataFrame.\n\n        Parameters:\n            raw_data (list): Raw data from the SPL API response.\n        \n        Returns:\n            pd.DataFrame: A Pandas DataFrame with columns ['time', 'open', 'high', 'low', 'close', 'volume'].\n        ";B='time';C=[B,'open','high','low','close','volume'];A=pd.DataFrame(raw_data,columns=C);A[B]=pd.to_datetime(A[B],unit='s');return A