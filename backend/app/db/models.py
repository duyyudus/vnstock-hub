from sqlalchemy import Column, String, Integer, Float, BigInteger, Date, DateTime, UniqueConstraint, Index, JSON
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
