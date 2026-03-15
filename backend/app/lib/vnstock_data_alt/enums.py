'\nEnumerations for vnstock_data library.\n\nProvides standardized enum types for common parameters and options.\n'
_B='INVESTMENT'
_A='SECTOR'
from enum import Enum
class IndexGroup(str,Enum):
	"\n    Enumeration for standardized market index groups.\n    \n    Provides short names for easy reference with automatic mapping to full names.\n    Inherits from str for ease of use with string-based APIs.\n    \n    Examples:\n        >>> from app.lib.vnstock_data_alt.enums import IndexGroup\n        >>> IndexGroup.HOSE\n        <IndexGroup.HOSE: 'HOSE'>\n        >>> IndexGroup.HOSE.value\n        'HOSE'\n        >>> IndexGroup.HOSE.full_name\n        'HOSE Indices'\n    ";HOSE='HOSE';SECTOR=_A;INVESTMENT=_B;VNX='VNX'
	@property
	def full_name(self):'Get the full name of the index group.';A={'HOSE':'HOSE Indices',_A:'Sector Indices',_B:'Investment Indices','VNX':'VNX Indices'};return A[self.value]
	def __str__(A):'Return the short name as string representation.';return A.value