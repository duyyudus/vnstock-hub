"""Symbol group helpers for vnstock listings."""

VCI_VALID_GROUPS = {
    'HOSE', 'VN30', 'VNMidCap', 'VNSmallCap', 'VNAllShare', 'VN100',
    'ETF', 'HNX', 'HNX30', 'HNXCon', 'HNXFin', 'HNXLCap', 'HNXMSCap',
    'HNXMan', 'UPCOM', 'FU_INDEX', 'FU_BOND', 'BOND', 'CW'
}

KBS_VALID_GROUPS = {
    'HOSE', 'HNX', 'UPCOM', 'VN30', 'VN100', 'VNMidCap', 'VNSmallCap',
    'VNSI', 'VNX50', 'VNXALL', 'VNALL', 'HNX30', 'ETF', 'CW', 'BOND',
    'FU_INDEX'
}

# Backward-compatible default for existing VCI callers.
VALID_GROUPS = VCI_VALID_GROUPS


def get_valid_groups_for_source(source: str = "VCI") -> set[str]:
    normalized_source = str(source or "").strip().upper()
    if normalized_source == "KBS":
        return KBS_VALID_GROUPS
    return VCI_VALID_GROUPS


def get_group_code_for_index(index_symbol: str, source: str = "VCI") -> str:
    """
    Map index symbol from all_indices() to group code expected by symbols_by_group().
    """
    normalized = str(index_symbol or "").strip().upper().replace("-", "").replace("_", "")
    normalized_source = str(source or "").strip().upper()

    mapping = {
        'VNINDEX': 'HOSE',
        'HNXINDEX': 'HNX',
        'UPCOMINDEX': 'UPCOM',
        'VNXINDEX': 'VNAllShare',
        'VN30': 'VN30',
        'VN100': 'VN100',
        'VNMID': 'VNMidCap',
        'VNSML': 'VNSmallCap',
        'VNALL': 'VNALL' if normalized_source == 'KBS' else 'VNAllShare',
        'HNX30': 'HNX30',
        # Add more mappings as needed based on valid groups
    }
    if normalized in mapping:
        return mapping[normalized]
    return str(index_symbol or "").strip()
