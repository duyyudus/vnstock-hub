from app.lib._vnstock_shared.common import indices  # Standardized market constants
from app.lib._vnstock_shared.common.data import (
    StockComponents, MSNComponents, Quote, Listing, Trading,
    Company, Finance, Fund
)

# Lazy initialization to avoid circular import deadlock
_initialized = False

def _ensure_initialized():
    """Ensure common module is initialized (called on first use)."""
    global _initialized
    if _initialized:
        return

    _initialized = True
