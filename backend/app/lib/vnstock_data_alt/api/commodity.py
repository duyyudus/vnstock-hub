_A=None
from app.lib.vnstock_data_alt.base import BaseAdapter,dynamic_method
from typing import Optional,Union
class CommodityPrice(BaseAdapter):
	'\n    Adapter for commodity prices from various SPL‐based providers.\n\n    Usage:\n        from app.lib.vnstock_data_alt.api.commodity import CommodityPrice\n        c = CommodityPrice(source="spl", start="2024-01-01", end="2024-04-01", show_log=False)\n        df = c.gold_vn()\n    ';_module_name='commodity'
	def __init__(B,source='spl',start=_A,end=_A,length=_A,show_log=False):
		A=source
		if A.lower()!='spl':raise ValueError("Lớp Commodity chỉ nhận giá trị tham số source là 'SPL'.")
		super().__init__(source=A,symbol=_A,start=start,end=end,length=length,show_log=show_log)
	@dynamic_method
	def gold_vn(self,start=_A,end=_A,length=_A):'Vietnam gold prices (buy & sell).'
	@dynamic_method
	def gold_global(self,start=_A,end=_A,length=_A):'Global gold price.'
	@dynamic_method
	def gas_vn(self,start=_A,end=_A,length=_A):'Vietnam gasoline & diesel prices.'
	@dynamic_method
	def oil_crude(self,start=_A,end=_A,length=_A):'Crude oil price.'
	@dynamic_method
	def gas_natural(self,start=_A,end=_A,length=_A):'Natural gas price.'
	@dynamic_method
	def coke(self,start=_A,end=_A,length=_A):'Coke price.'
	@dynamic_method
	def steel_d10(self,start=_A,end=_A,length=_A):'Vietnam rebar D10 price.'
	@dynamic_method
	def iron_ore(self,start=_A,end=_A,length=_A):'Iron ore price.'
	@dynamic_method
	def steel_hrc(self,start=_A,end=_A,length=_A):'Hot‑rolled coil (HRC) steel price.'
	@dynamic_method
	def fertilizer_ure(self,start=_A,end=_A,length=_A):'Urea fertilizer price.'
	@dynamic_method
	def soybean(self,start=_A,end=_A,length=_A):'Soybean price.'
	@dynamic_method
	def corn(self,start=_A,end=_A,length=_A):'Corn price.'
	@dynamic_method
	def sugar(self,start=_A,end=_A,length=_A):'Sugar price.'
	@dynamic_method
	def pork_north_vn(self,start=_A,end=_A,length=_A):'Northern Vietnam live hog price.'
	@dynamic_method
	def pork_china(self,start=_A,end=_A,length=_A):'China live hog price.'