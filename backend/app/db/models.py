from sqlalchemy import Column, String, Integer, Float, BigInteger, Date, DateTime, UniqueConstraint, Index
from datetime import datetime
from app.db.database import Base

class StockCompany(Base):
    """Model to store company full names for stock symbols."""
    __tablename__ = "stock_companies"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), unique=True, index=True, nullable=False)
    company_name = Column(String(255), nullable=False)
    charter_capital = Column(Float, nullable=True)
    pe_ratio = Column(Float, nullable=True)
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

