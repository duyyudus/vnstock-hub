_I='coerce'
_H=True
_G=None
_F=False
_E='MAS'
_D='VCI'
_C='volume'
_B='match_type'
_A='time'
import pytz,pandas as pd
from bs4 import BeautifulSoup
from typing import Dict,Any,List,Optional,Union
from datetime import datetime,timedelta,time
from app.lib._vnstock_shared.core.utils.parser import localize_timestamp
from app.lib._vnstock_shared.core.utils.transform import get_trading_date
from app.lib._vnstock_shared.core.utils.interval import normalize_interval
from app.lib._vnstock_shared.core.types import TimeFrame
def generate_period(df):
	D='report_period';C='lengthReport';A='yearReport';E=[A,C,D]
	for B in E:
		if B not in df.columns:raise ValueError(f"Thiếu cột {B} trong df")
	df.loc[:,'period']=df.apply(lambda x:str(x[A])if x[D]=='year'else str(x[A])+'-Q'+str(x[C]),axis=1);return df
def remove_pattern_columns(df,patterns):"\n    Remove columns from a DataFrame that contains any of the given patterns in their names\n    \n    Parameters\n    ----------\n    df : pd.DataFrame\n        Input DataFrame\n    patterns : List[str]\n        List of patterns to remove\n    \n    Returns\n    -------\n    pd.DataFrame\n        DataFrame with columns that match the given patterns removed\n    \n    Examples\n    --------\n    >>> df = pd.DataFrame({'FooBar': [1, 2], 'BarFoo': [3, 4]})\n    >>> remove_pattern_columns(df, ['foo'])\n       BarFoo\n    0       3\n    1       4\n    ";A=[A for A in df.columns if any(B in A.lower()for B in patterns)];df.drop(columns=A,inplace=_H);return df
def clean_numeric_string(s):
	"\n    Loại bỏ dấu phân nhóm (',' hoặc NBSP), chuẩn hoá dấu thập phân về '.'\n    ";A=','
	if not isinstance(s,str):return s
	s=s.replace('\xa0','').replace(A,'')
	if s.count('.')==0 and s.count(A)==1:s=s.replace(A,'.')
	return s.strip()
def process_match_types(df,asset_type,source):
	"\n    Process match_type labels with special handling for stock ATO/ATC transactions.\n    \n    For VCI:\n        - Replace 'b' with 'Buy' and 's' with 'Sell'.\n        - Missing values are represented by the string 'unknown'.\n    For MAS:\n        - Replace 'BUY' with 'Buy' and 'SELL' with 'Sell'.\n        - Fill in NaN values with 'unknown'.\n    For TCBS:\n        - Replace 'BU' with 'Buy' and 'SD' with 'Sell'.\n        - Missing values are assumed to be an empty string.\n    For KBS:\n        - Replace 'B' with 'Buy' and 'S' with 'Sell'.\n        - Missing values are assumed to be an empty string.\n    \n    Once basic replacements are done, if the asset type is 'stock' and missing\n    match_type values are present, the code will mark the ATO (at the earliest\n    morning transaction in the 9:13–9:17 window) and ATC (at the latest afternoon\n    transaction in the 14:43–14:47 window) within each trading day.\n    \n    Parameters:\n        df (DataFrame): The input DataFrame with a 'time' and 'match_type' column.\n        asset_type (str): The asset type (for example, 'stock').\n        source (str): The data source (e.g., 'VCI', 'MAS', 'TCBS', or 'KBS').\n        \n    Returns:\n        DataFrame: The modified DataFrame with updated match_type values.\n    ";H='unknown';E='Sell';D='Buy';C='date';B=source;A=df
	if B==_D:A[_B]=A[_B].replace({'b':D,'s':E})
	elif B==_E:A[_B]=A[_B].replace({'BUY':D,'SELL':E});A[_B]=A[_B].fillna(H)
	elif B=='TCBS':A[_B]=A[_B].replace({'BU':D,'SD':E})
	elif B=='KBS':A[_B]=A[_B].replace({'B':D,'S':E})
	F=H if B in[_D,_E]else''
	if asset_type=='stock'and(A[_B].eq(F).any()or A[_B].eq('').any()):
		A=A.sort_values(_A);A[C]=A[_A].dt.date
		def G(day_df):
			B=day_df;A=B[B[_B]==F]
			if A.empty:return B
			C=A[(A[_A].dt.hour==9)&A[_A].dt.minute.between(13,17)]
			if not C.empty:E=C[_A].idxmin();B.loc[E,_B]='ATO'
			D=A[(A[_A].dt.hour==14)&A[_A].dt.minute.between(43,47)]
			if not D.empty:G=D[_A].idxmax();B.loc[G,_B]='ATC'
			return B
		try:A=A.groupby(C,group_keys=_F).apply(G,include_groups=_F)
		except TypeError:A=A.groupby(C,group_keys=_F).apply(G)
		if C in A.columns:A.drop(columns=[C],inplace=_H)
	return A
def ohlc_to_df(data,column_map,dtype_map,asset_type,symbol,source,interval='1D',floating=2,resample_map=_G):
	'\n    Convert OHLC data from any source to standardized DataFrame format.\n    \n    Supports all 14 intervals: 1m, 5m, 15m, 30m, 1H, 4h, 1D, 1W, 1M\n    ';Q='Asia/Ho_Chi_Minh';P='UTC';M=interval;L=asset_type;J=source;I=column_map;H=data;G=resample_map;F='low';E='high';D='close';C='open'
	if not H:raise ValueError('Input data is empty or not provided.')
	try:K=normalize_interval(M)
	except ValueError as R:raise ValueError(f"Invalid interval '{M}': {str(R)}")
	if J=='TCBS':A=pd.DataFrame(H);A.rename(columns=I,inplace=_H)
	else:S={A:I[A]for A in I.keys()if A in H};A=pd.DataFrame(H)[S.keys()].rename(columns=I)
	T=[_A,C,E,F,D,_C];N=[B for B in T if B not in A.columns]
	if N:U=A.columns.tolist();raise ValueError(f"Missing required columns: {N}. Available columns: {U}")
	A=A[[_A,C,E,F,D,_C]]
	if _A in A.columns:
		if J==_D:A[_A]=pd.to_datetime(A[_A].astype(int),unit='s').dt.tz_localize(P);A[_A]=A[_A].dt.tz_convert(Q)
		elif J==_E:A[_A]=pd.to_datetime(A[_A].astype(float),unit='s').dt.tz_localize(P).dt.tz_convert(Q)
		else:A[_A]=pd.to_datetime(A[_A],errors=_I)
	if L not in['index','derivative']:A[[C,E,F,D]]=A[[C,E,F,D]].div(1000)
	A[[C,E,F,D]]=A[[C,E,F,D]].round(floating)
	if G is _G:G=_get_default_resample_map()
	V=[TimeFrame.MINUTE_1,TimeFrame.HOUR_1,TimeFrame.DAY_1]
	if K not in V and G:
		if K.value in G:A=A.set_index(_A).resample(G[K.value]).agg({C:'first',E:'max',F:'min',D:'last',_C:'sum'}).dropna(subset=[C,D]).reset_index()
	for(B,O)in dtype_map.items():
		if B in A.columns:
			if O=='datetime64[ns]'and hasattr(A[B],'dt')and A[B].dt.tz is not _G:
				A[B]=A[B].dt.tz_localize(_G)
				if K in[TimeFrame.DAY_1,TimeFrame.DAILY]:A[B]=A[B].dt.date
			A[B]=A[B].astype(O)
	A.name=symbol;A.category=L;A.source=J;return A
def _get_default_resample_map():'\n    Get default resample frequency mapping for all 14 intervals.\n    \n    Returns:\n        dict: Mapping of TimeFrame value to pandas resample frequency\n    ';return{'1m':'1min','5m':'5min','15m':'15min','30m':'30min','1H':'1H','4h':'4H','1D':'1D','1W':'1W','1M':'1MS'}
def intraday_to_df(data,column_map,dtype_map,symbol,asset_type,source):
	'\n    Convert intraday trading data to standardized DataFrame format,\n    với:\n      - Tiền xử lý chuỗi số cho price/volume\n      - Map scale dựa trên source (không hardcode /1000)\n      - Kiểm soát NaN, rounding volume an toàn\n      - Xử lý time, match_type như trước\n    ';P='symbol';J=symbol;G=asset_type;F='price';D=column_map;C=source
	if not data:E=pd.DataFrame(columns=list(D.values()));E.attrs[P]=J;E.category=G;E.source=C;return E
	A=pd.DataFrame(data);H=[B for B in D if B in A.columns]
	if not H:Q=list(D);R=A.columns.tolist();raise ValueError(f"Expected columns {Q} not found, got {R}")
	A=A[H].rename(columns={A:D[A]for A in H})
	for B in(F,_C):
		if B in A.columns:
			A[B]=A[B].map(clean_numeric_string);A[B]=pd.to_numeric(A[B],errors=_I);K=A[B].isna().sum()
			if K:I='[Warning] '+str(K)+" giá trị ở '"+B+"' không parse được, chuyển thành NaN";print(I)
	S={_D:1000,_E:1000};T=S.get(C,1)
	if F in A.columns:A[F]=A[F]/T
	if _C in A.columns:
		L=A[_C].fillna(0);M=L%1!=0
		if M.any():U=int(M.sum());I=f"[Info] {U} giá trị volume có decimal, sẽ làm tròn";print(I)
		A[_C]=L.round().astype(int)
	if _A in A.columns:
		V=get_trading_date()
		if C==_D:A[_A]=localize_timestamp(A[_A].astype(int),unit='s')
		elif C==_E:A[_A]=localize_timestamp(A[_A].astype(int),unit='ms');A[_A]=A[_A].dt.floor('s')
		else:
			N=str(A[_A].iloc[0])if not A.empty else''
			if':'in N and len(N)<=8:A[_A]=A[_A].apply(lambda x:datetime.combine(V,datetime.strptime(x,'%H:%M:%S').time())if isinstance(x,str)and':'in x else pd.NaT);A[_A]=localize_timestamp(A[_A],return_string=_F)
			else:
				A[_A]=pd.to_datetime(A[_A],format='%Y-%m-%d %H:%M:%S',errors=_I)
				if A[_A].dt.tz is _G:A[_A]=localize_timestamp(A[_A],return_string=_F)
	if _B in A.columns:A=process_match_types(A,G,C)
	if _A in A.columns:A=A.sort_values(_A)
	A=A.reset_index(drop=_H);O={B:C for(B,C)in dtype_map.items()if B in A.columns and B!=_A}
	if O:A=A.astype(O)
	A.attrs[P]=J;A.category=G;A.source=C;return A