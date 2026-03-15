'\nProxy manager for vnstock_data.\n\nProvides functionality to fetch, test, and manage proxies with speed tracking\nand intelligent selection strategies.\n'
_C='https://httpbin.org/ip'
_B='http'
_A=None
import requests
from typing import List,Optional,Dict,Any
from datetime import datetime
from enum import Enum
from app.lib._vnstock_shared.core.utils.logger import get_logger
from.proxy import Proxy
logger=get_logger(__name__)
class ProxyManager:
	'\n    Manages proxy fetching, testing, and selection.\n    \n    Features:\n    - Fetch proxies from proxyscrape API\n    - Test proxy connectivity and speed\n    - Track proxy metadata (speed, last_checked, country)\n    - Select proxies based on different strategies\n    - Maintain proxy cache\n    ';PROXYSCRAPE_API='https://api.proxyscrape.com/v2/'
	def __init__(A,timeout=5):'\n        Initialize ProxyManager.\n        \n        Args:\n            timeout: Request timeout in seconds (default: 5)\n        ';A.timeout=timeout;A.proxy_cache={};A.last_fetch=_A
	def fetch_proxies(A,limit=10,protocol=_B,ssl='all',anonymity='all',country=_A):
		'\n        Fetch proxies from proxyscrape API.\n        \n        Args:\n            limit: Number of proxies to fetch (default: 10)\n            protocol: Protocol type - http, socks4, socks5 (default: http)\n            ssl: SSL support - yes, no, all (default: all)\n            anonymity: Anonymity level - elite, anonymous, transparent, all (default: all)\n            country: Country code filter (optional)\n        \n        Returns:\n            List[Proxy]: List of fetched proxies\n        \n        Raises:\n            requests.RequestException: If API request fails\n        ';L='proxies';K='country';G=country;F=protocol;E=limit
		try:
			H={'request':'getproxies','format':'json','protocol':F,'ssl':ssl,'anonymity':anonymity,'limit':E,'simplified':'true'}
			if G:H[K]=G
			logger.info(f"Fetching {E} proxies from proxyscrape API...");I=requests.get(A.PROXYSCRAPE_API,params=H,timeout=A.timeout);I.raise_for_status();J=I.json();B=[]
			if J.get(L):
				for C in J[L]:D=Proxy(protocol=F.upper(),ip=C.get('ip'),port=int(C.get('port',0)),country=C.get(K),last_checked=datetime.now());B.append(D);A.proxy_cache[str(D)]=D
			A.last_fetch=datetime.now();logger.info(f"Successfully fetched {len(B)} proxies");return B
		except requests.RequestException as M:logger.error(f"Failed to fetch proxies: {M}");raise
	def test_proxies(F,proxies,test_url=_C,timeout=_A):
		'\n        Test proxy connectivity and measure speed.\n        \n        Args:\n            proxies: List of proxies to test\n            test_url: URL to test proxy against (default: httpbin.org)\n            timeout: Request timeout in seconds (default: self.timeout)\n        \n        Returns:\n            List[Proxy]: List of valid proxies with speed measurements\n        ';D=proxies;B=timeout
		if B is _A:B=F.timeout
		C=[]
		for A in D:
			try:
				G={_B:str(A),'https':str(A)};H=datetime.now();I=requests.get(test_url,proxies=G,timeout=B);J=datetime.now()
				if I.status_code==200:E=(J-H).total_seconds()*1000;A.speed=E;A.last_checked=datetime.now();C.append(A);logger.debug(f"Proxy {A} is valid (speed: {E:.2f}ms)")
			except(requests.RequestException,Exception)as K:logger.debug(f"Proxy {A} failed: {K}");continue
		logger.info(f"Tested {len(D)} proxies, {len(C)} are valid");return C
	def get_best_proxy(D,proxies):
		'\n        Get proxy with best speed.\n        \n        Args:\n            proxies: List of proxies to choose from\n        \n        Returns:\n            Optional[Proxy]: Proxy with best speed, or None if list is empty\n        ';A=proxies
		if not A:return
		C=[A for A in A if A.speed is not _A]
		if not C:return A[0]
		B=min(C,key=lambda p:p.speed);logger.debug(f"Best proxy selected: {B} (speed: {B.speed:.2f}ms)");return B
	def select_proxy(C,proxies,mode='best'):
		'\n        Select proxy based on mode.\n        \n        Args:\n            proxies: List of proxies to choose from\n            mode: Selection mode - best, random, first (default: best)\n        \n        Returns:\n            Optional[Proxy]: Selected proxy, or None if list is empty\n        ';B=mode;A=proxies
		if not A:return
		if B=='best':return C.get_best_proxy(A)
		elif B=='random':import random as D;return D.choice(A)
		elif B=='first':return A[0]
		else:logger.warning(f"Unknown selection mode: {B}, using 'first'");return A[0]
	def validate_proxy(C,proxy):
		'\n        Validate if proxy is still working.\n        \n        Args:\n            proxy: Proxy to validate\n        \n        Returns:\n            bool: True if proxy is valid, False otherwise\n        ';B=False;A=proxy
		try:
			D={_B:str(A),'https':str(A)};E=requests.get(_C,proxies=D,timeout=C.timeout)
			if E.status_code==200:A.last_checked=datetime.now();return True
			return B
		except Exception as F:logger.debug(f"Proxy validation failed: {F}");return B
	def get_cached_proxies(A):'\n        Get all cached proxies.\n        \n        Returns:\n            List[Proxy]: List of cached proxies\n        ';return list(A.proxy_cache.values())
	def clear_cache(A):'Clear proxy cache.';A.proxy_cache.clear();logger.info('Proxy cache cleared')
	def get_cache_stats(A):'\n        Get cache statistics.\n        \n        Returns:\n            Dict: Cache statistics including size, last fetch time, etc.\n        ';B=[A for A in A.proxy_cache.values()if A.is_valid()];return{'total_cached':len(A.proxy_cache),'valid_proxies':len(B),'last_fetch':A.last_fetch.isoformat()if A.last_fetch else _A,'cache_size':len(A.proxy_cache)}