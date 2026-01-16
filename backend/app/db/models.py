from sqlalchemy import Column, String, Integer
from app.db.database import Base

class StockCompany(Base):
    """Model to store company full names for stock symbols."""
    __tablename__ = "stock_companies"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), unique=True, index=True, nullable=False)
    company_name = Column(String(255), nullable=False)
