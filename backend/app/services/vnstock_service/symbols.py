"""Symbol group helpers for vnstock listings."""

# Valid groups supported by symbols_by_group
VALID_GROUPS = {
    'HOSE', 'VN30', 'VNMidCap', 'VNSmallCap', 'VNAllShare', 'VN100',
    'ETF', 'HNX', 'HNX30', 'HNXCon', 'HNXFin', 'HNXLCap', 'HNXMSCap',
    'HNXMan', 'UPCOM', 'FU_INDEX', 'FU_BOND', 'BOND', 'CW'
}


def get_group_code_for_index(index_symbol: str) -> str:
    """
    Map index symbol from all_indices() to group code expected by symbols_by_group().
    """
    mapping = {
        'VN30': 'VN30',
        'VN100': 'VN100',
        'VNMID': 'VNMidCap',
        'VNSML': 'VNSmallCap',
        'VNALL': 'VNAllShare',
        'HNX30': 'HNX30',
        # Add more mappings as needed based on valid groups
    }
    return mapping.get(index_symbol, index_symbol)
