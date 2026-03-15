from app.lib.vnstock_data_alt.base import BaseAdapter,dynamic_method
class Market(BaseAdapter):
	_module_name='market';SUPPORTED_SOURCES=['vnd']
	def __init__(C,source='vnd',**D):
		A=source;B=A.lower()
		if B not in C.SUPPORTED_SOURCES:raise ValueError("Lớp Market chỉ nhận giá trị tham số source là 'VND'. "+"Nhưng nhận được '"+A+"'.")
		super().__init__(source=B,**D)
	@dynamic_method
	def pe(self,duration='5Y'):'\n        Retrieves P/E ratio data for the given duration.\n        '
	@dynamic_method
	def pb(self,duration='5Y'):'\n        Retrieves P/B ratio data for the given duration.\n        '
	@dynamic_method
	def evaluation(self,duration='5Y'):'\n        Retrieves combined P/E and P/B data for the given duration.\n        '