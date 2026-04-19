"""
Company Reference Domain.
"""

from app.lib.vnstock_data_alt.ui._base import BaseDetail
from app.lib.vnstock_data_alt.ui._registry import REFERENCE_SOURCES


class CompanyReference(BaseDetail):
    """
    Company Reference Data (Layer 1).
    Wraps functionality for retrieving company-specific static data.
    """

    def __init__(self, symbol):
        super().__init__(symbol=symbol, domain_name="company", layer_sources=REFERENCE_SOURCES)

    def info(self):
        """Get company info/overview."""
        return self._dispatch("info")

    def shareholders(self):
        """Get company shareholders."""
        return self._dispatch("shareholders")

    def officers(self, filter_by="working"):
        """
        Get company officers.

        Args:
            filter_by (str): 'working', 'resigned', or 'all'. Default 'working'.
        """
        return self._dispatch("officers", filter_by=filter_by)

    def subsidiaries(self, filter_by="all"):
        """
        Get company subsidiaries.

        Args:
            filter_by (str): 'all', 'subsidiary', or 'affiliate'. Default 'all'.
        """
        return self._dispatch("subsidiaries", filter_by=filter_by)

    def ownership(self):
        """Get company ownership composition."""
        return self._dispatch("ownership")

    def capital_history(self):
        """Get company charter capital history."""
        return self._dispatch("capital_history")

    def news(self):
        """Get company news."""
        return self._dispatch("news")

    def events(self):
        """Get company events."""
        return self._dispatch("events")

    def insider_trading(self):
        """Get insider trading data."""
        return self._dispatch("insider_trading")

    def margin_ratio(self):
        """Get margin lending ratio for the company across brokers."""
        return self._dispatch("margin_ratio")
