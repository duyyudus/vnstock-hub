from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IndexValue:
    """Index value data class."""
    symbol: str
    name: str
    value: float  # Current/latest close price
    change: float  # Price change from open (percentage)
    change_value: float  # Absolute change from open


@dataclass
class StockInfo:
    """Stock information data class."""
    ticker: str
    price: float
    market_cap: float  # In billion VND (tỷ đồng)
    company_name: str = ""
    exchange: str = ""
    charter_capital: float = 0.0  # In billion VND
    pe_ratio: float | None = None
    accumulated_value: float | None = None  # In billion VND
    price_change_24h: float | None = None  # Percentage
    price_change_1w: float | None = None  # Percentage
    price_change_1m: float | None = None  # Percentage
    price_change_1y: float | None = None  # Percentage
