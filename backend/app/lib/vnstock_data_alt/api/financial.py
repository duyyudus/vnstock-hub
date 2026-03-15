_A='quarter'
from typing import Any
from tenacity import retry,stop_after_attempt,wait_exponential
from app.lib.vnstock_data_alt.config import Config
from app.lib.vnstock_data_alt.base import BaseAdapter,dynamic_method
class Finance(BaseAdapter):
	_module_name='financial';'\n    Adapter for financial reports:\n      - balance_sheet\n      - income_statement\n      - cash_flow\n      - ratio\n\n    Usage:\n        f = Finance(source="vci", symbol="VCI", period="quarter", get_all=True, show_log=True)\n        df_bs = f.balance_sheet(period="annual", lang="en", dropna=True)\n        df_is = f.income_statement(lang="vi")\n        df_cf = f.cash_flow()\n        df_rat = f.ratio(flatten_columns=True, separator="_")\n    '
	def __init__(C,source,symbol,period=_A,get_all=True,show_log=False):
		B=period;A=source
		if A.lower()not in['vci','mas','kbs']:raise ValueError("Lớp Finance chỉ nhận giá trị tham số source là 'VCI' hoặc 'MAS'.")
		if B.lower()not in['year',_A]:raise ValueError("Lớp Finance chỉ nhận giá trị tham số period là 'year' hoặc 'quarter'.")
		super().__init__(source=A,symbol=symbol,period=B,get_all=get_all,show_log=show_log)
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def balance_sheet(self,*A,**B):'\n        Retrieve balance sheet data.\n        Forwards supported kwargs (e.g., period, lang, dropna, show_log).\n        '
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def income_statement(self,*A,**B):'\n        Retrieve income statement data.\n        '
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def cash_flow(self,*A,**B):'\n        Retrieve cash flow statement data.\n        '
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def note(self,*A,**B):
		"\n        Retrieve financial statement notes (thuyết minh báo cáo tài chính) if source is 'vci'.\n        "
		if self.source.lower()=='vci':return self._provider.note(*A,**B)
		raise NotImplementedError("'note' method is only implemented for source 'vci'.")
	@retry(stop=stop_after_attempt(Config.RETRIES),wait=wait_exponential(multiplier=Config.BACKOFF_MULTIPLIER,min=Config.BACKOFF_MIN,max=Config.BACKOFF_MAX))
	@dynamic_method
	def ratio(self,*A,**B):'\n        Retrieve financial ratios.\n        Supports provider kwargs like flatten_columns, separator, drop_levels, etc.\n        '