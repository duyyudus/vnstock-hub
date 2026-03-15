_A=None
import re,urllib.parse,pandas as pd,unicodedata
from datetime import datetime,timedelta,date
from dateutil.relativedelta import relativedelta
import math
def get_asset_type(symbol):
	"\n    Xác định loại tài sản dựa trên mã chứng khoán được cung cấp.\n\n    Tham số: \n        - symbol (str): Mã chứng khoán hoặc mã chỉ số.\n    \n    Trả về:\n        - 'index' nếu mã chứng khoán là mã chỉ số.\n        - 'stock' nếu mã chứng khoán là mã cổ phiếu.\n        - 'derivative' nếu mã chứng khoán là mã hợp đồng tương lai hoặc quyền chọn.\n        - 'bond' nếu mã chứng khoán là mã trái phiếu (chính phủ hoặc doanh nghiệp).\n        - 'coveredWarr' nếu mã chứng khoán là mã chứng quyền.\n    ";symbol=symbol.upper()
	if symbol in['VNINDEX','HNXINDEX','UPCOMINDEX','VN30','VN100','HNX30','VNSML','VNMID','VNALL','VNREAL','VNMAT','VNIT','VNHEAL','VNFINSELECT','VNFIN','VNENE','VNDIAMOND','VNCONS','VNCOND']:return'index'
	elif len(symbol)==3:return'stock'
	elif len(symbol)in[7,9]:
		fm_pattern=re.compile('^VN30F\\d{1,2}M$');ym_pattern=re.compile('^VN30F\\d{4}$');gov_bond_pattern=re.compile('^GB\\d{2}F\\d{4}$');comp_bond_pattern=re.compile('^(?!VN30F)[A-Z]{3}\\d{6}$')
		if gov_bond_pattern.match(symbol)or comp_bond_pattern.match(symbol):return'bond'
		elif fm_pattern.match(symbol)or ym_pattern.match(symbol):return'derivative'
		else:raise ValueError('Invalid derivative or bond symbol. Symbol must be in format of VN30F1M, VN30F2024, GB10F2024, or for company bonds, e.g., BAB122032')
	elif len(symbol)==8:return'coveredWarr'
	else:raise ValueError('Invalid symbol. Your symbol format is not recognized!')
def encode_url(string,safe=''):'\n    Encode a string to url format.\n    ';return urllib.parse.quote(string,safe=safe)
def days_between(start,end,format='%m/%d/%Y'):"\n    Calculate the number of days between two given datestrings.\n\n    Parameters:\n        start (str): The start date string.\n        end (str): The end date string.\n        format (str): The format of the date strings. Default is '%m/%d/%Y'.\n        \n    Returns:\n        int: The number of days between the two dates.\n    ";start=pd.to_datetime(start,format=format);end=pd.to_datetime(end,format=format);days=(end-start).days;return days
def lookback_date(period):
	"\n    Calculates the start date based on a lookback period.\n\n    Parameters:\n        period (str): Lookback period (e.g., '1D', '5M', '10Y').\n                      D = days, M = months, Y = years.\n\n    Returns:\n        str: Calculated start date in YYYY-MM-DD format.\n\n    Raises:\n        ValueError: If the period format is invalid.\n    "
	try:
		unit=period[-1].upper();value=int(period[:-1])
		if unit=='D':start_date=datetime.now()-timedelta(days=value)
		elif unit=='M':start_date=datetime.now()-timedelta(days=value*30)
		elif unit=='Y':start_date=datetime.now()-timedelta(days=value*365)
		else:raise ValueError("Invalid period format. Use 'D', 'M', or 'Y' for days, months, or years.")
		return start_date.strftime('%Y-%m-%d')
	except Exception as e:raise ValueError(f"Error parsing period: {e}")
def filter_columns_by_language(df,lang):
	'\n    Filter DataFrame columns based on language preference.\n    \n    Args:\n        df (pd.DataFrame): Input DataFrame with multilingual columns\n        lang (str): Language preference ("vi", "en", or "both")\n        \n    Returns:\n        pd.DataFrame: DataFrame with filtered columns based on language preference\n    ';B='_en';A='_vi'
	if lang.lower()=='both':return df
	all_columns=df.columns.tolist();vi_columns=[col for col in all_columns if col.endswith(A)];en_columns=[col for col in all_columns if col.endswith(B)];base_columns=[col for col in all_columns if not(col.endswith(A)or col.endswith(B))]
	if lang.lower()=='vi':columns_to_keep=base_columns+vi_columns;filtered_df=df[columns_to_keep].copy();rename_dict={col:col.replace(A,'')for col in vi_columns};filtered_df=filtered_df.rename(columns=rename_dict)
	elif lang.lower()=='en':columns_to_keep=base_columns+en_columns;filtered_df=df[columns_to_keep].copy();rename_dict={col:col.replace(B,'')for col in en_columns};filtered_df=filtered_df.rename(columns=rename_dict)
	else:logger.warning(f"Invalid language parameter '{lang}'. Use 'vi', 'en', or 'both'. Returning all columns.");return df
	return filtered_df
def vn_to_snake_case(text):
	'\n    Convert Vietnamese text with diacritics to unaccented, lowercase, snake_case string.\n    ';B='_';A='a';text=unicodedata.normalize('NFD',text);text=''.join([c for c in text if unicodedata.category(c)!='Mn']);VN_CHAR_MAP={'đ':'d','Đ':'d','ô':'o','Ô':'o','ư':'u','Ư':'u','ê':'e','Ê':'e','ă':A,'Ă':A,'â':A,'Â':A}
	for(src,tgt)in VN_CHAR_MAP.items():text=text.replace(src,tgt)
	text=text.lower();text=re.sub('[^a-z0-9]+',B,text);text=text.strip(B);text=re.sub('_+',B,text);return text
def get_derivative_maturity_date(symbol_suffix,reference_date=_A):
	"\n    Calculate the maturity date for a derivative symbol suffix.\n    \n    Args:\n        symbol_suffix: Suffix like 'F2506' (explicit) or 'F1M', 'F1Q' (relative).\n        reference_date: Reference date for relative calculation (default: today).\n        \n    Returns:\n        date: The maturity date (3rd Thursday of the month).\n    "
	if reference_date is _A:reference_date=date.today()
	maturity_month=reference_date.month;maturity_year=reference_date.year
	if len(symbol_suffix)==5 and symbol_suffix.startswith('F')and symbol_suffix[1:].isdigit():yy=int(symbol_suffix[1:3]);mm=int(symbol_suffix[3:5]);maturity_year=2000+yy;maturity_month=mm
	elif symbol_suffix.startswith('F')and len(symbol_suffix)==3:
		def get_expiry(y,m):d=date(y,m,1);days_to_thursday=(3-d.weekday()+7)%7;return d+timedelta(days=days_to_thursday+14)
		current_expiry=get_expiry(reference_date.year,reference_date.month);base_date=reference_date
		if reference_date>current_expiry:base_date=reference_date+relativedelta(months=1)
		if symbol_suffix=='F1M':target_date=base_date
		elif symbol_suffix=='F2M':target_date=base_date+relativedelta(months=1)
		elif symbol_suffix=='F1Q':
			current_month=base_date.month;quarter_months=[3,6,9,12];target_month=next((m for m in quarter_months if m>=current_month),_A)
			if target_month is _A:target_date=date(base_date.year+1,3,1)
			else:target_date=date(base_date.year,target_month,1)
		elif symbol_suffix=='F2Q':
			current_month=base_date.month;quarter_months=[3,6,9,12];target_month=next((m for m in quarter_months if m>=current_month),_A)
			if target_month is _A:target_date=date(base_date.year+1,6,1)
			else:
				idx=quarter_months.index(target_month)
				if idx+1<len(quarter_months):target_date=date(base_date.year,quarter_months[idx+1],1)
				else:target_date=date(base_date.year+1,3,1)
		else:return reference_date
		if'target_date'in locals():maturity_year=target_date.year;maturity_month=target_date.month
	d=date(maturity_year,maturity_month,1);days_to_thursday=(3-d.weekday()+7)%7;maturity_date=d+timedelta(days=days_to_thursday+14);return maturity_date
def convert_derivative_symbol(symbol,reference_date=_A):
	'\n    Convert old derivative symbol (e.g. VN30F2506, GB05F2506) to new KRX format (e.g. 41I1F7000).\n    Effective from May 2025.\n    \n    Format: 41 + Underlying + Year + Month + 000\n    Length: 9 chars.\n    Reference: https://owa.hnx.vn/ftp//PORTALNEW/HEADER_IMAGES/20250428/21.04.2025_Tai%20lieu%20huong%20dan%20quy%20uoc%20ma%20giao%20dich%20moi%20ckps_final.pdf\n    ';symbol=symbol.upper();underlying_map={'VN30':'I1','VN100':'I2','GB05':'B5','GB10':'BA'};underlying_code=_A;suffix=_A
	for(prefix,code)in underlying_map.items():
		if symbol.startswith(prefix):underlying_code=code;suffix=symbol[len(prefix):];break
	if not underlying_code:raise ValueError('Unknown underlying asset for symbol {}. Supported prefixes: {}'.format(symbol,', '.join(underlying_map.keys())))
	if reference_date is _A:0
	mat_date=get_derivative_maturity_date(suffix,reference_date);mat_year=mat_date.year;mat_month=mat_date.month;year_cycle_index=(mat_year-2010)%30;alphabet='ABCDEFGHJKLMNPQRSTVW'
	if 0<=year_cycle_index<=9:year_code=str(year_cycle_index)
	else:year_code=alphabet[year_cycle_index-10]
	if 1<=mat_month<=9:month_code=str(mat_month)
	else:month_codes={10:'A',11:'B',12:'C'};month_code=month_codes[mat_month]
	krx_symbol=f"41{underlying_code}{year_code}{month_code}000";return krx_symbol
def safe_convert_derivative_symbol(symbol,reference_date=_A):
	"\n    Safely convert derivative symbol to KRX format. If it's already KRX or an error occurs,\n    returns the original symbol.\n    "
	try:
		if len(symbol)==9 and symbol.startswith('41'):return symbol
		return convert_derivative_symbol(symbol,reference_date)
	except Exception:return symbol