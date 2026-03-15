"\nUtility module for managing and displaying startup messages via logging.\nTracks message display counts using a local JSON state file.\n\nTo suppress startup messages, configure logging in your code:\n    import logging\n    logging.getLogger('vnstock_data.core.utils.startup').setLevel(logging.WARNING)\n\nOr set environment variable:\n    export VNSTOCK_STARTUP_LOG_LEVEL=WARNING\n"
_A=True
import os,json,logging,datetime
from pathlib import Path
from app.lib.vnstock_data_alt.core.config.messages import STARTUP_MESSAGES
logger=logging.getLogger(__name__)
def _get_state_file_path():
	'Get the path to the local state file for tracking startup messages.';B=Path.home();A=B/'.vnstock'
	if not A.exists():
		try:A.mkdir(parents=_A,exist_ok=_A)
		except Exception:pass
	return A/'startup_state.json'
def _load_state():
	'Load the message display state from the local JSON file.';A=_get_state_file_path()
	try:
		if A.exists():
			with open(A,'r',encoding='utf-8')as B:return json.load(B)
	except Exception:pass
	return{}
def _save_state(state):
	'Save the message display state to the local JSON file.';A=_get_state_file_path()
	try:
		if not A.parent.exists():A.parent.mkdir(parents=_A,exist_ok=_A)
		with open(A,'w',encoding='utf-8')as B:json.dump(state,B)
	except Exception:pass
def display_startup_messages():
	'\n    Check and display active startup messages based on configured constraints.\n    Updates the display count in the local state file.\n    ';D=False;B=_load_state();I=datetime.datetime.now()
	for A in STARTUP_MESSAGES:
		C=A.get('id');E=A.get('text','');F=A.get('max_displays');G=A.get('expires_at')
		if not C or not E:continue
		if G:
			try:
				J=datetime.datetime.fromisoformat(G)
				if I>J:continue
			except ValueError:pass
		H=B.get(C,0)
		if F is not None and H>=F:continue
		logger.info(E);B[C]=H+1;D=_A
	if D:_save_state(B)