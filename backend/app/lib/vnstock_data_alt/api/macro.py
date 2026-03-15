_C=False
_B='month'
_A=None
from app.lib.vnstock_data_alt.base import BaseAdapter,dynamic_method
from typing import Optional,Union
class Macro(BaseAdapter):
	'\n    Adapter for macroeconomic data from multiple providers (e.g. MBK).\n    \n    Usage:\n        from app.lib.vnstock_data_alt.api.macro import Macro\n        m = Macro(source="mbk", random_agent=False, show_log=False)\n        df = m.gdp(start="2015-01", end="2025-04", period="quarter")\n    ';_module_name='macro'
	def __init__(B,source='mbk',random_agent=_C,show_log=_C):
		A=source
		if A!='mbk':raise ValueError('Lớp Macro không hỗ trợ thay đổi tham số source.')
		super().__init__(source=A,random_agent=random_agent,show_log=show_log)
	@dynamic_method
	def gdp(self,start=_A,end=_A,period='quarter',keep_label=_C,length=_A):'\n        Fetch GDP series.\n        '
	@dynamic_method
	def cpi(self,start=_A,end=_A,period=_B,length=_A):'\n        Fetch CPI series.\n        '
	@dynamic_method
	def industry_prod(self,start=_A,end=_A,period=_B,length=_A):'\n        Fetch Industrial Production series.\n        '
	@dynamic_method
	def import_export(self,start=_A,end=_A,period=_B,length=_A):'\n        Fetch Import-Export series.\n        '
	@dynamic_method
	def retail(self,start=_A,end=_A,period=_B,length=_A):'\n        Fetch Retail sales series.\n        '
	@dynamic_method
	def fdi(self,start=_A,end=_A,period=_B,length=_A):'\n        Fetch Foreign Direct Investment series.\n        '
	@dynamic_method
	def money_supply(self,start=_A,end=_A,period=_B,length=_A):'\n        Fetch Money Supply series.\n        '
	@dynamic_method
	def exchange_rate(self,start=_A,end=_A,period='day',length=_A):'\n        Fetch Exchange Rate series.\n        '
	@dynamic_method
	def interest_rate(self,start=_A,end=_A,period='day',format='pivot',length=_A):'\n        Fetch Interest Rate series.\n        '
	@dynamic_method
	def population_labor(self,start=_A,end=_A,period='year',length=_A):'\n        Fetch Population and Labor series.\n        '