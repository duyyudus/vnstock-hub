_A='VNINDEX'
from app.lib.vnstock_data_alt.base import BaseAdapter,dynamic_method
class TopStock(BaseAdapter):
	'\n    Adapter for VND TopStock “insight” APIs.  Only supports source="vnd".\n    ';_module_name='insight';SUPPORTED_SOURCES=['vnd']
	def __init__(C,source='vnd',**D):
		A=source;B=A.lower()
		if B not in C.SUPPORTED_SOURCES:raise ValueError("Lớp TopStock chỉ nhận giá trị tham số source là 'VND'. "+"Nhưng nhận được '"+A+"'.")
		super().__init__(source=B,**D)
	@dynamic_method
	def gainer(self,index=_A,limit=10):'\n        Top 10 gainers in the given index.\n        '
	@dynamic_method
	def loser(self,index=_A,limit=10):'\n        Top 10 losers in the given index.\n        '
	@dynamic_method
	def value(self,index=_A,limit=10):'\n        Top 10 by trading value in the given index.\n        '
	@dynamic_method
	def volume(self,index=_A,limit=10):'\n        Top 10 by abnormal volume in the given index.\n        '
	@dynamic_method
	def deal(self,index=_A,limit=10):'\n        Top 10 by block trade volume in the given index.\n        '
	@dynamic_method
	def foreign_buy(self,date=None,limit=10):'\n        Top 10 net foreign buys on the given date.\n        '
	@dynamic_method
	def foreign_sell(self,date=None,limit=10):'\n        Top 10 net foreign sells on the given date.\n        '