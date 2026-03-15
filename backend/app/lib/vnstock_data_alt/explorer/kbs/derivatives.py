'\nPatchwork data source for KBS Covered Warrants.\n'
_N='application/json'
_M='application/json, text/plain, */*'
_L='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36'
_K='"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"'
_J='"macOS"'
_I='Content-Type'
_H='Accept'
_G='User-Agent'
_F='x-lang'
_E='sec-ch-ua-mobile'
_D='sec-ch-ua'
_C='Referer'
_B='sec-ch-ua-platform'
_A='coerce'
import pandas as pd,requests
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
class KBSDerivatives:
	'\n    KBS implementation for fetching derivatives data (Warrants, Futures).\n    '
	def __init__(A,symbol=None):'Standard detail provider initialization.';A.symbol=symbol
	def warrant_profile(N,symbol=None):
		'\n        Fetch the detailed profile and realtime statistics of a covered warrant.\n        \n        Args:\n            symbol (str): Optional warrant symbol. Defaults to instance symbol.\n        ';M='break_even_point';G='CWT';F='CP';E='EX';C='CPR';B='EP';H=symbol or N.symbol
		if not H:raise ValueError('Warrant symbol is required')
		O='https://kbbuddywts.kbsec.com.vn/iis-server/investment/stock/iss';P={_B:_J,_C:'https://kbbuddywts.kbsec.com.vn/',_D:_K,_E:'?0',_F:'vi',_G:_L,_H:_M,'DNT':'1',_I:_N};Q={'code':H};I=requests.post(O,headers=P,json=Q);I.raise_for_status();J=I.json()
		if not J:return pd.DataFrame()
		A=pd.DataFrame(J);from app.lib.vnstock_data_alt.explorer.kbs.const import _EXCHANGE_CODE_MAP as R
		if E in A.columns:A[E]=A[E].map(lambda x:R.get(x,x)if pd.notna(x)else x)
		S=[B,C,'RE','CL','FL',F,'AP','B1','B2','B3','S1','S2','S3','OP']
		for D in S:
			if D in A.columns:A[D]=pd.to_numeric(A[D],errors=_A)/1000
		if G in A.columns:T={'C':'Call','P':'Put'};A[G]=A[G].map(lambda x:T.get(str(x).upper(),x))
		if B in A.columns and'ER'in A.columns and F in A.columns:
			try:
				K=A['ER'].astype(str).str.split(':');L=pd.to_numeric(K.str[0],errors=_A)/pd.to_numeric(K.str[1],errors=_A);A[M]=A[B]+A[F]*L
				if C in A.columns:A['break_even_point_diff']=A[M]-A[C];A['intrinsic_value']=((A[C]-A[B])/L).clip(lower=0)
			except Exception:pass
		return A
	def future_profile(H,symbol=None):
		'\n        Fetch the detailed profile and realtime statistics of an index future.\n        \n        Args:\n            symbol (str): Optional futures symbol. Defaults to instance symbol.\n        ';G='data';E='time';D='exchange';from app.lib.vnstock_data_alt.explorer.kbs.const import _DERIVATIVE_MAP as I;from app.lib.vnstock_data_alt.core.utils.parser import safe_convert_derivative_symbol as J;C=symbol or H.symbol
		if not C:raise ValueError('Future symbol is required')
		C=J(C);K='https://kbbuddywts.kbsec.com.vn/iis-server/investment/derivative/iss';L={_B:_J,_C:'https://kbbuddywts.kbsec.com.vn/DER',_D:_K,_E:'?0',_F:'vi',_G:_L,_H:_M,'DNT':'1',_I:_N};M={'code':C};F=requests.post(K,headers=L,json=M);F.raise_for_status();B=F.json()
		if not B:return pd.DataFrame()
		if isinstance(B,dict)and G in B:B=B[G]
		if not isinstance(B,list):return pd.DataFrame()
		A=pd.DataFrame(B);A=A.rename(columns=I);from app.lib.vnstock_data_alt.explorer.kbs.const import _EXCHANGE_CODE_MAP as N
		if D in A.columns:A[D]=A[D].map(lambda x:N.get(x,x)if pd.notna(x)else x)
		if E in A.columns:A[E]=pd.to_datetime(A[E],unit='ms',errors=_A)
		O=['symbol','full_name','underlying_symbol',D,'first_trading_date','last_trading_date','reference_price','ceiling_price','floor_price','open_price','high_price','low_price','close_price','open_interest','basis','foreign_buy_volume','foreign_sell_volume'];P=[B for B in O if B in A.columns];A=A[P];return A
ProviderRegistry.register('derivatives','kbs',KBSDerivatives)