'\nProxy data structures and utilities for vnstock_data.\n'
_D='last_checked'
_C='country'
_B='protocol'
_A=None
from dataclasses import dataclass,field
from typing import Optional
from datetime import datetime
@dataclass
class Proxy:
	'Proxy information with metadata.';protocol:str;ip:str;port:int;country:Optional[str]=_A;speed:Optional[float]=_A;last_checked:Optional[datetime]=_A
	def __str__(A):'String representation of proxy.';return f"{A.protocol}://{A.ip}:{A.port}"
	def __repr__(A):'Detailed representation of proxy.';return f"Proxy(protocol={A.protocol}, ip={A.ip}, port={A.port}, country={A.country}, speed={A.speed}, last_checked={A.last_checked})"
	def is_valid(A):
		"\n        Check if proxy is still valid.\n        \n        A proxy is considered valid if it was checked within the last 24 hours.\n        If never checked, it's considered valid.\n        \n        Returns:\n            bool: True if proxy is valid, False otherwise\n        "
		if A.last_checked is _A:return True
		B=datetime.now()-A.last_checked;return B.days<1
	def to_dict(A):'Convert proxy to dictionary.';return{_B:A.protocol,'ip':A.ip,'port':A.port,_C:A.country,'speed':A.speed,_D:A.last_checked.isoformat()if A.last_checked else _A}
	@classmethod
	def from_dict(C,data):
		'Create proxy from dictionary.';A=data;B=A.get(_D)
		if B and isinstance(B,str):B=datetime.fromisoformat(B)
		return C(protocol=A[_B],ip=A['ip'],port=A['port'],country=A.get(_C),speed=A.get('speed'),last_checked=B)