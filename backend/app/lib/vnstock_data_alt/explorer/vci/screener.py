'Screener module.'
_O='VCI.ext'
_N='source'
_M='pageSize'
_L='VCI'
_K='3Month'
_J=False
_I='30Days'
_H='data'
_G='extraName'
_F='type'
_E='to'
_D='from'
_C='conditionOptions'
_B='name'
_A='value'
from typing import Dict,Optional,List,Any,Union
import json,pandas as pd
from app.lib._vnstock_shared.core.utils.parser import camel_to_snake
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib.vnstock_data_alt.core.utils.client import send_request
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers
from app.lib._vnstock_shared.compat import agg_execution
logger=get_logger(__name__)
# Deprecated/blocked VCI screener surface, observed 2026-05-04.
# These endpoints can return non-JSON error bodies; avoid backend production
# dependencies on them unless a live probe confirms access.
_SCREENER_PAGING_URL='https://iq.vietcap.com.vn/api/iq-insight-service/v1/screening/paging'
_SCREENER_CRITERIA_URL='https://iq.vietcap.com.vn/api/iq-insight-service/v1/screening/criteria'
_DEFAULT_SCREENER_PAYLOAD={'page':0,_M:50,'sortFields':[],'sortOrders':[],'filter':[{_B:'sectorLv1',_C:[{_F:_A,_A:'1000'},{_F:_A,_A:'3000'},{_F:_A,_A:'2000'},{_F:_A,_A:'0001'},{_F:_A,_A:'6000'},{_F:_A,_A:'4000'},{_F:_A,_A:'7000'},{_F:_A,_A:'5000'},{_F:_A,_A:'8600'},{_F:_A,_A:'8301'},{_F:_A,_A:'9000'},{_F:_A,_A:'8500'},{_F:_A,_A:'8700'}]},{_B:'exchange',_C:[{_F:_A,_A:'hsx'},{_F:_A,_A:'hnx'},{_F:_A,_A:'upcom'}]},{_B:'marketCap',_C:[{_D:0,_E:0x71afd498d0000}]},{_B:'marketPrice',_C:[{_D:0,_E:2000000}]},{_B:'dailyPriceChangePercent',_C:[{_D:-15,_E:15}]},{_B:'adtv',_G:_I,_C:[{_D:0,_E:2000000000000}]},{_B:'avgVolume',_G:_I,_C:[{_D:0,_E:200000000}]},{_B:'esVolumeVsAvgVolume',_G:_I,_C:[{_D:-900,_E:900}]},{_B:'stockStrength',_C:[{_D:0,_E:100}]},{_B:'rs',_G:_K,_C:[{_D:0,_E:100}]},{_B:'rsi',_C:[{_D:0,_E:100}]},{_B:'priceEma',_G:'ema20',_C:[{_D:-50,_E:50}]},{_B:'ema20Ema50',_C:[{_D:-50,_E:50}]},{_B:'ema50Ema200',_C:[{_D:-50,_E:50}]},{_B:'priceReturn',_G:_K,_C:[{_D:-100,_E:100}]},{_B:'outperformsIndex',_G:_K,_C:[{_D:-100,_E:100}]},{_B:'priceFluctuation',_G:_I,_C:[{_D:-100,_E:100}]},{_B:'macd',_C:[{_D:-200000,_E:200000}]},{_B:'histogram',_C:[{_D:-5000,_E:5000}]},{_B:'adx',_C:[{_D:0,_E:100}]},{_B:'stockTrend',_C:[{_F:_A,_A:'STRONG_UPTREND'}]},{_B:'aoTrend',_C:[{_F:_A,_A:'ABOVE_ZERO'}]},{_B:'ttmPe',_C:[{_D:0,_E:100}]},{_B:'ttmPb',_C:[{_D:0,_E:100}]},{_B:'ttmRoe',_C:[{_D:-50,_E:50}]},{_B:'npatmiGrowth',_G:'Yoy','extraName2':'Qm1',_C:[{_D:-100,_E:500}]},{_B:'revenueGrowth',_G:'Yoy',_C:[{_D:-100,_E:500}]},{_B:'netMargin',_C:[{_D:-100,_E:100}]},{_B:'grossMargin',_C:[{_D:-100,_E:100}]}]}
class Screener:
	'\n    Screener API cho VCI. Truy cập bộ lọc cổ phiếu với tất cả các tiêu chí.\n    '
	def __init__(A,random_agent=_J,show_log=_J):
		C='application/json';B=show_log;A.data_source=_L;A.headers=get_headers(data_source=A.data_source,random_agent=random_agent);A.headers.update({'Accept':C,'Accept-Language':'en-US,en;q=0.9,vi;q=0.8','Origin':'https://trading.vietcap.com.vn','Referer':'https://trading.vietcap.com.vn/','Sec-Fetch-Dest':'empty','Sec-Fetch-Mode':'cors','Sec-Fetch-Site':'same-site','Content-Type':C});A.show_log=B
		if not B:logger.setLevel('CRITICAL')
	@agg_execution(_O)
	def get_criteria(self,lang='vi',show_log=_J,to_df=True):
		'\n        Lấy cấu trúc mapping tên tiêu chí và định nghĩa readable của các fields từ VCI Screener.\n        ';H='category';C=send_request(url=_SCREENER_CRITERIA_URL,headers=self.headers,method='GET',show_log=show_log)
		if not C or _H not in C:raise ValueError('Không thể lấy dữ liệu criteria.')
		D=C[_H]
		if not to_df:return D
		E=[]
		for A in D:
			if not isinstance(A,dict):continue
			B=A.get(_B);I=A.get(H);K=A.get(_C,[]);J=camel_to_snake(B)if B else None
			if lang=='vi':F=A.get('viName',B)
			else:F=A.get('enName',B)
			E.append({H:I,'field_name':B,'column_name':J,'readable_name':F,'select_type':A.get('selectType')})
		G=pd.DataFrame(E);G.attrs[_N]=_L;return G
	@agg_execution(_O)
	def filter(self,show_log=_J,to_df=True,limit=2000):
		'\n        Lấy dữ liệu của tất cả cổ phiếu thỏa mãn cấu hình bộ lọc mặc định rộng nhất.\n        User sẽ dùng pandas dataframe để tự phân tích và filter thêm.\n        \n        Args:\n            show_log (bool, optional): Hiển thị log. Defaults to False.\n            to_df (bool, optional): Trả về DataFrame. Defaults to True.\n            limit (int, optional): Giới hạn số lượng bản ghi trả về. Defaults to 2000 (do thị trường VN có hơn 1600 mã).\n        ';I='content';D=limit;B=[];G=0;H=50;E=_DEFAULT_SCREENER_PAYLOAD.copy()
		while True:
			E['page']=G;E[_M]=H;C=send_request(url=_SCREENER_PAGING_URL,headers=self.headers,method='POST',payload=E,show_log=show_log)
			if not C or _H not in C or I not in C[_H]:break
			F=C[_H][I]
			if not F:break
			B.extend(F)
			if len(B)>=D or len(F)<H:break
			G+=1
		if not to_df:return B[:D]
		if not B:return pd.DataFrame()
		A=pd.DataFrame(B[:D]);A.columns=[camel_to_snake(A)for A in A.columns];A.attrs[_N]=_L
		if'id'in A.columns:A=A.drop(columns=['id'])
		return A
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
ProviderRegistry.register('screener','vci',Screener)
