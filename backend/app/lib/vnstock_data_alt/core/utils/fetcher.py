import requests,pandas as pd
from typing import Dict,Any
from app.lib.vnstock_data_alt.core.const import ERROR_MESSAGES
class Fetcher:
	'\n    Common request fetcher for all data providers, ready to be extended for specific requirements.\n    '
	def __init__(A,base_url,headers,api_key=None):'\n        Initialize the Fetcher with a base URL and headers.\n\n        Parameters:\n            base_url (str): Base URL for the API.\n            headers (dict): Default headers for the API requests.\n            api_key (str, optional): API key for authentication. Default is None.\n        ';A.base_url=base_url;A.headers=headers;A.api_key=api_key
	def fetch(A,endpoint,params,extra_headers=None):
		'\n        Fetch raw data from a specific API endpoint.\n\n        Parameters:\n            endpoint (str): The API endpoint to fetch data from.\n            params (dict): Query parameters for the API request.\n            extra_headers (dict, optional): Additional headers specific to the request. Default is None.\n        \n        Returns:\n            dict: Raw JSON response data.\n\n        Raises:\n            RuntimeError: If the API request fails.\n            ValueError: If the response status code is not 200 or the data is empty.\n        ';C=extra_headers;F=f"{A.base_url}{endpoint}";D=A.headers.copy()
		if C:D.update(C)
		try:B=requests.get(F,headers=D,params=params);B.raise_for_status()
		except requests.RequestException as G:H=ERROR_MESSAGES['api_failure']+': '+str(G);raise RuntimeError(H)
		if B.status_code!=200:raise ValueError('No available data: Non-200 status code received.')
		E=B.json()
		if not A._has_data(E):raise ValueError('No available data: Empty response content.')
		return E
	def _has_data(B,response_data):
		'\n        Check if the response data contains any data.\n\n        Parameters:\n            response_data (Any): The data returned from the API response.\n        \n        Returns:\n            bool: True if data is present, False otherwise.\n        ';A=response_data
		if isinstance(A,(dict,list)):return bool(len(A))
		return False
	def validate(A,params):'\n        Placeholder for parameter validation. Extend in local data provider modules.\n\n        Parameters:\n            params (dict): Query parameters to validate.\n\n        Raises:\n            NotImplementedError: If not overridden in a subclass.\n        ';raise NotImplementedError("Override 'validate' method in the specific provider fetcher.")
	def to_dataframe(A,raw_data):'\n        Placeholder for converting raw data to a Pandas DataFrame. Extend in local data provider modules.\n\n        Parameters:\n            raw_data (Any): Raw data from the API response.\n        \n        Returns:\n            pd.DataFrame: A Pandas DataFrame representation of the raw data.\n\n        Raises:\n            NotImplementedError: If not overridden in a subclass.\n        ';raise NotImplementedError("Override 'to_dataframe' method in the specific provider fetcher.")