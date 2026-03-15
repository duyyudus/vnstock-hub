'\nAPI client utilities for vnstock data sources.\n\nModule này cung cấp các tiện ích để gửi request tới các nguồn dữ liệu của vnstock, hỗ trợ nhiều chế độ gửi (trực tiếp, qua proxy) và nhiều chế độ chọn proxy (try, rotate, random, single).\n\nCác hàm chính:\n- send_request: interface trung tâm cho tất cả các mode gửi request\n- send_request_direct: gửi request trực tiếp\n- send_proxy_request: gửi request qua proxy thông thường\n'
_D='direct'
_C='GET'
_B=False
_A=None
import requests,json,random
from typing import Dict,Any,Optional,Union,List
from enum import Enum
from pydantic import BaseModel
from app.lib._vnstock_shared.core.utils.logger import get_logger
logger=get_logger(__name__)
class ProxyConfig(BaseModel):
	'\n    Cấu hình proxy cho các request API.\n    Sử dụng cho các class/module cần truyền proxy.\n    \n    Attributes:\n        proxy_list: Danh sách proxy URL (định dạng string)\n        proxy_objects: Danh sách Proxy objects với metadata\n        proxy_mode: Chế độ chọn proxy (TRY, ROTATE, RANDOM, SINGLE)\n        request_mode: Chế độ gửi request (DIRECT, PROXY)\n        auto_fetch: Tự động lấy proxy từ API\n        validate_proxies: Kiểm tra tính hợp lệ của proxy\n        prefer_speed: Ưu tiên proxy có tốc độ tốt nhất\n    ';proxy_list:Optional[List[str]]=_A;proxy_objects:Optional[List[Any]]=_A;proxy_mode:'ProxyMode'='try';request_mode:'RequestMode'=_D;auto_fetch:bool=_B;validate_proxies:bool=_B;prefer_speed:bool=_B
	class Config:arbitrary_types_allowed=True
logger=get_logger(__name__)
class ProxyMode(Enum):'\n    Các chế độ sử dụng proxy khi gửi request:\n    - TRY: Thử lần lượt từng proxy cho đến khi thành công\n    - ROTATE: Luân phiên proxy sau mỗi lần gọi\n    - RANDOM: Chọn ngẫu nhiên proxy cho mỗi lần gọi\n    - SINGLE: Luôn dùng proxy đầu tiên\n    ';TRY='try';ROTATE='rotate';RANDOM='random';SINGLE='single'
class RequestMode(Enum):'\n    Các chế độ gửi request:\n    - DIRECT: Gửi trực tiếp không qua proxy\n    - PROXY: Gửi qua proxy thông thường\n    ';DIRECT=_D;PROXY='proxy'
_current_proxy_index=0
def build_proxy_dict(proxy_url):'\n    Chuyển đổi proxy URL thành dict format cho requests.\n    Args:\n        proxy_url (str): URL của proxy\n    Returns:\n        Dict[str, str]: Dict cấu hình proxy cho requests\n    ';A=proxy_url;return{'http':A,'https':A}
def get_proxy_by_mode(proxy_list,mode):
	'\n    Lấy proxy từ danh sách proxy theo chế độ đã chọn.\n    Args:\n        proxy_list (List[str]): Danh sách proxy URL\n        mode (ProxyMode): Chế độ chọn proxy\n    Returns:\n        str: Proxy URL được chọn\n    ';B=mode;A=proxy_list;global _current_proxy_index
	if not A:raise ValueError('Proxy list is empty')
	if B==ProxyMode.SINGLE:return A[0]
	elif B==ProxyMode.RANDOM:return random.choice(A)
	elif B==ProxyMode.ROTATE:C=A[_current_proxy_index%len(A)];_current_proxy_index+=1;return C
	else:return A[0]
def send_request(url,headers,method=_C,params=_A,payload=_A,show_log=_B,timeout=30,proxy_list=_A,proxy_mode=ProxyMode.TRY,request_mode=RequestMode.DIRECT,auto_fetch=_B,validate_proxies=_B,prefer_speed=_B):
	'\n    Interface trung tâm cho tất cả các mode gửi request.\n    Tùy theo request_mode và proxy_mode sẽ chọn cách gửi request phù hợp.\n    \n    Args:\n        url (str): Địa chỉ endpoint\n        headers (Dict[str, str]): Header cho request\n        method (str): "GET" hoặc "POST". Mặc định "GET"\n        params (Optional[Dict]): Tham số query cho GET\n        payload (Optional[Union[Dict, str]]): Dữ liệu gửi đi (POST)\n        show_log (bool): Bật log chi tiết\n        timeout (int): Timeout (giây)\n        proxy_list (Optional[List[str]]): Danh sách proxy URLs (cho PROXY mode)\n        proxy_mode (Union[ProxyMode, str]): Chế độ sử dụng proxy\n        request_mode (Union[RequestMode, str]): Chế độ gửi request\n        auto_fetch (bool): Tự động lấy proxy từ API\n        validate_proxies (bool): Kiểm tra tính hợp lệ của proxy\n        prefer_speed (bool): Ưu tiên proxy có tốc độ tốt nhất\n    \n    Returns:\n        Dict[str, Any]: Dữ liệu JSON trả về\n    \n    Raises:\n        ConnectionError: Nếu tất cả proxy đều thất bại hoặc request lỗi\n    ';X='://';W='HTTP';T=prefer_speed;S=validate_proxies;R=auto_fetch;L=headers;K=':';J=timeout;I=method;H=url;G=payload;F=params;E=request_mode;C=proxy_mode;B=show_log;A=proxy_list
	if R or S or T:
		from.proxy_manager import ProxyManager as Y;M=Y(timeout=J)
		if R:
			if B:logger.info('Auto-fetching proxies from proxyscrape API...')
			try:
				A=[str(A)for A in M.fetch_proxies(limit=10)];E=RequestMode.PROXY
				if B:logger.info(f"Fetched {len(A)} proxies")
			except Exception as D:logger.warning(f"Failed to auto-fetch proxies: {D}")
		if S and A:
			if B:logger.info('Validating proxies...')
			try:
				from.proxy import Proxy;N=[Proxy(protocol=W,ip=A.split(X)[-1].split(K)[0],port=int(A.split(K)[-1]))for A in A];Z=M.test_proxies(N);A=[str(A)for A in Z]
				if B:logger.info(f"{len(A)} proxies are valid")
			except Exception as D:logger.warning(f"Failed to validate proxies: {D}")
		if T and A:
			if B:logger.info('Selecting fastest proxy...')
			try:
				from.proxy import Proxy;N=[Proxy(protocol=W,ip=A.split(X)[-1].split(K)[0],port=int(A.split(K)[-1]))for A in A];O=M.get_best_proxy(N)
				if O:
					A=[str(O)];C=ProxyMode.SINGLE
					if B:logger.info(f"Using fastest proxy: {O}")
			except Exception as D:logger.warning(f"Failed to select fastest proxy: {D}")
	if isinstance(C,str):
		try:C=ProxyMode(C)
		except ValueError:raise ValueError(f"Invalid proxy_mode: {C}")
	if isinstance(E,str):
		try:E=RequestMode(E)
		except ValueError:raise ValueError(f"Invalid request_mode: {E}")
	if B:
		logger.info(f"{I.upper()} request to {H} (mode: {E.value})")
		if F:logger.info(f"Params: {F}")
		if G:logger.info(f"Payload: {G}")
	if E==RequestMode.PROXY:
		if not A:raise ValueError('proxy_list is required for PROXY mode')
		if C==ProxyMode.TRY:
			U=_A
			for P in A:
				try:
					if B:logger.info(f"Trying proxy: {P}")
					Q=build_proxy_dict(P);return send_request_direct(H,L,I,F,G,J,Q)
				except ConnectionError as D:
					U=D
					if B:logger.warning(f"Proxy {P} failed: {D}")
					continue
			raise ConnectionError(f"All proxies failed. Last error: {U}")
		else:
			V=get_proxy_by_mode(A,C);Q=build_proxy_dict(V)
			if B:logger.info(f"Using proxy ({C.value} mode): {V}")
			return send_request_direct(H,L,I,F,G,J,Q)
	else:
		if B:logger.info('Sending direct request (no proxy)')
		return send_request_direct(H,L,I,F,G,J,proxies=_A)
def send_request_direct(url,headers,method=_C,params=_A,payload=_A,timeout=30,proxies=_A):
	'\n    Gửi request trực tiếp tới endpoint, không qua proxy đặc biệt.\n    Args:\n        url (str): Endpoint URL\n        headers (Dict[str, str]): Header cho request\n        method (str): "GET" hoặc "POST"\n        params (Optional[Dict]): Tham số query cho GET\n        payload (Optional[Union[Dict, str]]): Dữ liệu gửi đi (POST)\n        timeout (int): Timeout (giây)\n        proxies (Optional[Dict[str, str]]): Dict proxy nếu có\n    Returns:\n        Dict[str, Any]: Dữ liệu JSON trả về\n    Raises:\n        ConnectionError: Nếu request thất bại hoặc trả về mã lỗi\n    ';F=proxies;E=timeout;D=headers;A=payload
	try:
		if method.upper()==_C:B=requests.get(url,headers=D,params=params,timeout=E,proxies=F)
		else:
			if A is not _A:
				if isinstance(A,dict):C=json.dumps(A)
				elif isinstance(A,str):C=A
				else:raise ValueError('Payload must be either a dictionary or a raw string.')
			else:C=_A
			B=requests.post(url,headers=D,data=C,timeout=E,proxies=F)
		if B.status_code!=200:raise ConnectionError(f"Failed to fetch data: {B.status_code} - {B.reason}")
		return B.json()
	except requests.exceptions.RequestException as H:G=f"API request failed: {str(H)}";logger.error(G);raise ConnectionError(G)
def reset_proxy_rotation():'\n    Reset proxy rotation index về 0.\n    Dùng khi muốn bắt đầu lại vòng quay proxy ở chế độ ROTATE.\n    ';global _current_proxy_index;_current_proxy_index=0
def send_direct_request(url,headers,**A):'\n    Gửi request trực tiếp không qua proxy.\n    Args:\n        url (str): Endpoint URL\n        headers (Dict[str, str]): Header cho request\n        **kwargs: Các tham số bổ sung cho send_request\n    Returns:\n        Dict[str, Any]: Dữ liệu JSON trả về\n    ';return send_request(url,headers,request_mode=RequestMode.DIRECT,**A)
def send_proxy_request(url,headers,proxy_list,**A):'\n    Gửi request qua proxy thông thường.\n    Args:\n        url (str): Endpoint URL\n        headers (Dict[str, str]): Header cho request\n        proxy_list (List[str]): Danh sách proxy URL\n        **kwargs: Các tham số bổ sung cho send_request\n    Returns:\n        Dict[str, Any]: Dữ liệu JSON trả về\n    ';return send_request(url,headers,proxy_list=proxy_list,request_mode=RequestMode.PROXY,**A)