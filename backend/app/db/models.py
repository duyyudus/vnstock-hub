from sqlalchemy import Column, String, Integer, Float, BigInteger, Date, DateTime, UniqueConstraint, Index, JSON, Boolean, ForeignKey
from datetime import datetime
from app.db.database import Base

class StockCompany(Base):
    """Model to store company full names for stock symbols."""
    __tablename__ = "stock_companies"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), unique=True, index=True, nullable=False)
    company_name = Column(String(255), nullable=False)
    exchange = Column(String(20), nullable=True)
    charter_capital = Column(Float, nullable=True)
    pe_ratio = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class StockIndex(Base):
    """Model to store stock market indices."""
    __tablename__ = "stock_indices"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(String(500), nullable=True)
    group = Column(String(100), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class StockDailyPrice(Base):
    """Historical daily prices for stocks (OHLCV data)."""
    __tablename__ = "stock_daily_prices"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_symbol_date'),
        Index('ix_stock_daily_prices_symbol_date', 'symbol', 'date'),
    )


class StockHistoryBackfillState(Base):
    """State tracking for incremental historical backfill per stock symbol."""
    __tablename__ = "stock_history_backfill_states"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, index=True, nullable=False)
    target_start_date = Column(Date, nullable=True)
    oldest_date = Column(Date, nullable=True)
    latest_date = Column(Date, nullable=True)
    last_seen_at = Column(DateTime, nullable=True)
    last_attempt_at = Column(DateTime, nullable=True)
    next_attempt_at = Column(DateTime, nullable=True)
    weekly_sync_last_attempt_at = Column(DateTime, nullable=True)
    weekly_sync_last_attempt_start_date = Column(Date, nullable=True)
    no_progress_attempts = Column(Integer, nullable=False, default=0)
    is_exhausted = Column(Boolean, nullable=False, default=False)
    exhausted_until = Column(DateTime, nullable=True)
    last_error = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('ix_stock_history_backfill_next_attempt', 'next_attempt_at'),
        Index('ix_stock_history_backfill_last_seen', 'last_seen_at'),
        Index('ix_stock_history_backfill_exhausted', 'is_exhausted', 'exhausted_until'),
    )


class FundNav(Base):
    """Historical NAV data for mutual funds."""
    __tablename__ = "fund_navs"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(30), nullable=False)  # Fund symbol/short_name
    date = Column(Date, nullable=False)
    nav = Column(Float, nullable=False)  # NAV per unit
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_fund_symbol_date'),
        Index('ix_fund_navs_symbol_date', 'symbol', 'date'),
    )


class FundDetailCache(Base):
    """Cached fund details (top holdings, industry holdings, asset holdings)."""
    __tablename__ = "fund_detail_cache"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(30), nullable=False)
    detail_type = Column(String(30), nullable=False)  # top_holding | industry_holding | asset_holding
    data = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('symbol', 'detail_type', name='uq_fund_detail_symbol_type'),
        Index('ix_fund_detail_symbol_type', 'symbol', 'detail_type'),
    )


class FundListing(Base):
    """Cached fund listing (open-end funds)."""
    __tablename__ = "fund_listings"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(30), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=True)
    fund_type = Column(String(50), nullable=True)
    fund_owner = Column(String(255), nullable=True)
    data = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('ix_fund_listings_symbol_type', 'symbol', 'fund_type'),
    )


class User(Base):
    """Application user account."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)


class BookmarkGroup(Base):
    """User-defined group for favorite stocks."""
    __tablename__ = "bookmark_groups"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(120), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_bookmark_group_user_name"),
    )


class BookmarkStock(Base):
    """Stock membership inside a bookmark group."""
    __tablename__ = "bookmark_stocks"

    id = Column(Integer, primary_key=True, index=True)
    group_id = Column(Integer, ForeignKey("bookmark_groups.id", ondelete="CASCADE"), nullable=False, index=True)
    ticker = Column(String(10), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("group_id", "ticker", name="uq_bookmark_group_ticker"),
        Index("ix_bookmark_group_ticker", "group_id", "ticker"),
    )


class PortfolioPosition(Base):
    """User portfolio position."""
    __tablename__ = "portfolio_positions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    ticker = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    average_cost = Column(Float, nullable=True)
    purchase_date = Column(Date, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "ticker", name="uq_portfolio_position_user_ticker"),
    )
