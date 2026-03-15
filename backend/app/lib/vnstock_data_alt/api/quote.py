'\nvnstock/api/quote.py\n\nUnified Quote adapter with dynamic method detection and parameter filtering.\n'
from typing import Any
from tenacity import retry,stop_after_attempt,wait_exponential
from app.lib.vnstock_data_alt.config import Config
from app.lib.vnstock_data_alt.base import BaseAdapter,dynamic_method
class Quote(BaseAdapter):
	_module_name='quote';'\n    Adapter for historical and intraday quote data.\n\n    Usage:\n        q = Quote(source="vci", symbol="VCI", random_agent=False, show_log=True)\n        df = q.history(start="2024-01-01", end="2024-04-18", interval="1D")\n        df2 = q.intraday(page_size=100)\n        depth = q.price_depth()\n    '
	def __init__(B,source='KBS',symbol='',random_agent=False,show_log=False):
		A=source
		if A.lower()not in['vci','kbs','vnd','mas']:raise ValueError("Lớp Quote chỉ nhận giá trị tham số source là 'VCI', 'KBS', 'VND' hoặc 'MAS'.")
		super().__init__(source=A,symbol=symbol,random_agent=random_agent,show_log=show_log)
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def history(self,*A,**B):'\n        Load historical OHLC data for the symbol.\n\n        Forwards only supported kwargs to provider.history().\n        '
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def intraday(self,*A,**B):'\n        Load intraday trade data for the symbol.\n        '
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def price_depth(self,*A,**B):'\n        Load price depth (order book) data for the symbol.\n        '