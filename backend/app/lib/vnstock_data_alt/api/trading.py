'\nvnstock/api/trading.py\n\nUnified Trading adapter with dynamic method detection and parameter filtering.\n'
from typing import Any
from tenacity import retry,stop_after_attempt,wait_exponential
from app.lib.vnstock_data_alt.config import Config
from app.lib.vnstock_data_alt.base import BaseAdapter,dynamic_method
class Trading(BaseAdapter):
	_module_name='trading';'\n    Adapter for trading data: trading_stats, side_stats, price_board.\n\n    Usage:\n        t = Trading(source="vci", symbol="VCI", random_agent=False, show_log=True)\n        df = t.trading_stats(start="2024-01-01", end="2024-12-31", limit=1000)\n        bids, asks = t.side_stats(dropna=True)\n        board = t.price_board(symbols_list=["VCI", "VCB"], **kwargs)\n    '
	def __init__(C,source='KBS',symbol='',random_agent=False,show_log=False):
		B=source;A=symbol
		if B.lower()not in['vci','kbs','vnd','cafef']:raise ValueError("Lớp Trading chỉ nhận giá trị tham số source là 'VCI', 'KBS', 'VND' hoặc 'CAFEF'.")
		if not A or A.strip()=='':A='VCI'
		super().__init__(source=B,symbol=A,random_agent=random_agent,show_log=show_log)
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def trading_stats(self,*A,**B):'\n        Retrieve trading statistics for the given symbol.\n        '
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def side_stats(self,*A,**B):'\n        Retrieve bid/ask side statistics for the given symbol.\n        '
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def price_board(self,*A,**B):'\n        Retrieve the price board (order book) for a list of symbols.\n        '
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def price_history(self,*A,**B):'\n        Retrieve the price history for a list of symbols.\n        '
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def foreign_trade(self,*A,**B):'\n        Retrieve foreign trade data for the given symbol.\n        '
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def prop_trade(self,*A,**B):'\n        Retrieve property trade data for the given symbol.\n        '
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def insider_deal(self,*A,**B):'\n        Retrieve insider deal data for the given symbol.\n        '
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def order_stats(self,*A,**B):'\n        Retrieve order statistics for the given symbol.\n        '
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def trade_history(self,*A,**B):'\n        Retrieve trade history for the given symbol.\n        '
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def matched_by_price(self,*A,**B):'\n        Retrieve trade data matched by price level for the given symbol.\n        '
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def odd_lot(self,*A,**B):'\n        Retrieve odd-lot (lô lẻ) trading data.\n        '
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def put_through(self,*A,**B):'\n        Retrieve put-through (thỏa thuận) trading data.\n        '