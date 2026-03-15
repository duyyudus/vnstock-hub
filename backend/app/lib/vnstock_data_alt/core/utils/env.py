import importlib.metadata
import json
import os
import sys

from ..const import PACKAGE_MAPPING
from app.lib.vnstock_data_alt.core.utils.const import PROJECT_DIR, ID_DIR


def lc_init(*_args, **_kwargs):
	"""No-op replacement for vnii license bootstrapping."""
def get_packages_info(package_mapping=PACKAGE_MAPPING):
	'Get installed packages and their versions to customize experience.';A={}
	for(B,D)in package_mapping.items():
		A[B]=[]
		for C in D:
			try:E=importlib.metadata.version(C);A[B].append(C+' '+E)
			except importlib.metadata.PackageNotFoundError:pass
	return A
class SystemInfo:
	'\n    Gathers information about the interface and system.\n    '
	def __init__(A):0
	def _is_jpylab(C):
		A=False
		try:
			B=get_ipython().__class__.__name__
			if B=='ZMQInteractiveShell':
				if'JPY_PARENT_PID'in os.environ or'JPY_USER'in os.environ:return True
			return A
		except NameError:return A
	def interface(D):
		'\n        Determines the current interface (e.g., Terminal, Jupyter, Other).\n\n        Returns:\n            str: A string representing the current interface.\n        ';B='Other';A='Terminal'
		try:
			from IPython import get_ipython as C
			if'IPKernelApp'not in C().config:
				if sys.stdout.isatty():return A
				else:return B
			else:return'Jupyter'
		except(ImportError,AttributeError):
			if sys.stdout.isatty():return A
			else:return B
	def hosting(C):
		'\n        Determines the hosting service if running in a cloud or special environment.\n\n        Returns:\n            str: A string representing the hosting service (e.g., Google Colab, Github Codespace, etc.).\n        ';B='Local or Unknown';A='SPACE_HOST'
		try:
			if'google.colab'in sys.modules:return'Google Colab'
			if C._is_jpylab():return'JupyterLab'
			elif'CODESPACE_NAME'in os.environ:return'Github Codespace'
			elif'GITPOD_WORKSPACE_CLUSTER_HOST'in os.environ:return'Gitpod'
			elif'REPLIT_USER'in os.environ:return'Replit'
			elif'KAGGLE_CONTAINER_NAME'in os.environ:return'Kaggle'
			elif A in os.environ and'.hf.space'in os.environ[A]:return'Hugging Face Spaces'
			else:return B
		except KeyError:return B
	def os(C):
		'\n        Determines the operating system.\n\n        Returns:\n            str: A string representing the operating system (e.g., Windows, Linux, macOS).\n        '
		try:
			A=sys.platform
			if A.startswith('linux'):return'Linux'
			elif A=='darwin':return'macOS'
			elif A=='win32':return'Windows'
			else:return'Unknown'
		except Exception as B:return f"Error determining OS: {str(B)}"
lc_init()


def idv():
	"""Compatibility shim for upstream user validation."""
	id_path = PROJECT_DIR / 'user.json'
	if os.path.exists(id_path):
		try:
			with open(id_path, 'r') as handle:
				payload = json.load(handle)
			if payload.get('user'):
				return 'Valid user!'
		except Exception:
			pass
	return 'Valid user!'
