'\nCore standardization logic for UI Reference Layer.\n'
from typing import Any,Dict
_SCHEMA_REGISTRY={}
_STANDARD_COLS_REGISTRY={}
_ENUM_REGISTRY={}
def register_schemas(schema_map,standard_cols,enum_map=None):
	'Register schema maps, standard columns, and optional enum value maps.';A=enum_map;_SCHEMA_REGISTRY.update(schema_map);_STANDARD_COLS_REGISTRY.update(standard_cols)
	if A:_ENUM_REGISTRY.update(A)
def standardize_columns(df,domain_method,source,strict=False):
	"\n    Standardize dataframe columns based on registered native schema mappings.\n    \n    Args:\n        df: Pandas DataFrame from provider.\n        domain_method: Key representing the domain method (e.g., 'company.profile').\n        source: Provider source (e.g., 'vci').\n        strict: If True, filters out any columns not part of the standard schema.\n    ";C=domain_method;A=df;import pandas as F
	if not isinstance(A,F.DataFrame):return A
	if A.empty:return A
	D=_SCHEMA_REGISTRY.get(C,{}).get(source)
	if D:G={B:C for(B,C)in D.items()if B in A.columns};A=A.rename(columns=G)
	if strict:
		E=_STANDARD_COLS_REGISTRY.get(C,[])
		if E:H=[B for B in E if B in A.columns];A=A[H]
	for(B,I)in _ENUM_REGISTRY.items():
		if B in A.columns:A[B]=A[B].replace(I)
	return A