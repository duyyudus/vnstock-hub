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
    foreign_buy_value: float | None = None  # In billion VND
    foreign_sell_value: float | None = None  # In billion VND
    current_room: int | None = None
    total_room: int | None = None
    price_change_24h: float | None = None  # Percentage
    price_change_1w: float | None = None  # Percentage
    price_change_1m: float | None = None  # Percentage
    price_change_6m: float | None = None  # Percentage
    price_change_1y: float | None = None  # Percentage
    price_change_2y: float | None = None  # Percentage
    price_change_3y: float | None = None  # Percentage
    atl_price: float | None = None  # VND
    atl_date: str | None = None
    atl_diff_pct: float | None = None  # Percentage
    ath_price: float | None = None  # VND
    ath_date: str | None = None
    ath_diff_pct: float | None = None  # Percentage
    industry: str = ""  # ICB Level 2 industry classification


@dataclass
class IndexContributionRow:
    """Per-stock contribution to an index or basket current-session move."""
    ticker: str
    company_name: str
    price: float
    prior_price: float
    session_return: float
    outstanding_shares: float
    free_float_ratio: float
    capping_factor: float
    effective_weight: float
    percent_contribution: float
    point_contribution: float | None
    missing_outstanding_shares: bool = False
    missing_free_float: bool = False
    used_market_cap_shares_fallback: bool = False


@dataclass
class IndexContributionTotals:
    """Aggregate contribution summary."""
    positive_percent: float
    negative_percent: float
    net_percent: float
    positive_points: float | None
    negative_points: float | None
    net_points: float | None
    excluded_count: int
    missing_outstanding_shares_count: int
    missing_free_float_count: int


@dataclass
class IndexContribution:
    """Index contribution result."""
    symbol: str
    name: str
    value: float | None
    change: float | None
    change_value: float | None
    rows: list[IndexContributionRow]
    totals: IndexContributionTotals
