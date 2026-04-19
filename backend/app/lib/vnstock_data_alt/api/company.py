from typing import Any
from tenacity import retry,stop_after_attempt,wait_exponential
from app.lib.vnstock_data_alt.config import Config
from app.lib.vnstock_data_alt.base import BaseAdapter,dynamic_method
class Company(BaseAdapter):
	_module_name='company';'\n    Adapter for company-related data:\n      - overview\n      - shareholders\n      - officers\n      - subsidiaries\n      - ownership\n      - capital_history\n      - affiliate\n      - news\n      - events\n      - insider_trading\n\n    Usage:\n        c = Company(source="KBS", symbol="VCB", random_agent=False, show_log=True)\n        df_ov = c.overview()\n        df_sh = c.shareholders()\n        df_of = c.officers(filter_by="all")\n        df_sub = c.subsidiaries(filter_by="subsidiary")\n        df_owner = c.ownership()\n        df_capital = c.capital_history()\n        df_aff = c.affiliate()\n        df_news = c.news()\n        df_evt = c.events()\n        df_it = c.insider_trading()\n    '
	def __init__(B,source='KBS',symbol=None,random_agent=False,show_log=False):
		A=source
		if A.lower()not in['vci','kbs']:raise ValueError("Lớp Company chỉ nhận giá trị tham số source là 'VCI' hoặc 'KBS'.")
		super().__init__(source=A,symbol=symbol,random_agent=random_agent,show_log=show_log)
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def overview(self,*A,**B):'Retrieve company overview data.'
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def shareholders(self,*A,**B):'Retrieve company shareholders data.'
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def officers(self,*A,**B):"\n        Retrieve company officers data.\n        Supports kwargs like filter_by='working'|'resigned'|'all'.\n        "
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def subsidiaries(self,*A,**B):"\n        Retrieve company subsidiaries data.\n        Supports kwargs like filter_by='all'|'subsidiary'.\n        "
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def ownership(self,*A,**B):'Retrieve company ownership data.'
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def affiliate(self,*A,**B):'Retrieve company affiliate data.'
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def news(self,*A,**B):'Retrieve company news.'
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def events(self,*A,**B):'Retrieve company events.'
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def capital_history(self,*A,**B):'Retrieve company charter capital history.'
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def insider_trading(self,*A,**B):'Retrieve insider trading data for the given symbol.'
