"""
Service for interacting with vnstock library to fetch Vietnam stock market data.
"""
from typing import List
from dataclasses import dataclass
import asyncio
import pandas as pd
from sqlalchemy import select
from app.db.database import async_session
from app.db.models import StockCompany


@dataclass
class StockInfo:
    """Stock information data class."""
    ticker: str
    price: float
    market_cap: float  # In billion VND (tỷ đồng)
    company_name: str = ""


class VnstockService:
    """Service class for vnstock operations."""
    
    def __init__(self):
        pass
    
    async def get_vn100_stocks(self) -> List[StockInfo]:
        """
        Fetch VN-100 stocks (top 100 by market cap) with price and market cap data.
        
        Returns:
            List of StockInfo objects include company_name
        """
        loop = asyncio.get_event_loop()
        stocks = await loop.run_in_executor(None, self._fetch_vn100_data)
        
        # Add company names from cache or fetch if missing
        return await self._enrich_stocks_with_company_names(stocks)

    async def _enrich_stocks_with_company_names(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """Add company names to stock info objects, using DB cache."""
        if not stocks:
            return []

        tickers = [s.ticker for s in stocks]
        
        async with async_session() as session:
            # Try to get from DB
            stmt = select(StockCompany).where(StockCompany.symbol.in_(tickers))
            result = await session.execute(stmt)
            cached_companies = {c.symbol: c.company_name for c in result.scalars().all()}
            
            missing_tickers = [t for t in tickers if t not in cached_companies]
            
            if missing_tickers:
                # Fetch missing from vnstock
                loop = asyncio.get_event_loop()
                all_symbols_df = await loop.run_in_executor(None, self._fetch_all_symbols)
                
                new_companies = []
                for _, row in all_symbols_df.iterrows():
                    symbol = row['symbol']
                    name = row['organ_name']
                    if symbol in missing_tickers:
                        cached_companies[symbol] = name
                    
                    # Also proactively cache everything if not in DB
                    # (Simplified: just cache the missing ones for now to avoid DB bloat if needed, 
                    # but usually it's better to cache everything we fetched anyway)
                    if symbol in missing_tickers:
                        new_companies.append(StockCompany(symbol=symbol, company_name=name))
                
                if new_companies:
                    session.add_all(new_companies)
                    await session.commit()
            
            for stock in stocks:
                stock.company_name = cached_companies.get(stock.ticker, "")
                
        return stocks

    def _fetch_all_symbols(self) -> pd.DataFrame:
        """Fetch all symbols from vnstock."""
        from vnstock import Listing
        listing = Listing()
        return listing.all_symbols()
    
    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten multi-level column names."""
        if df.columns.nlevels > 1:
            new_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    # Join non-NaN parts with underscore
                    new_cols.append('_'.join([str(c) for c in col if pd.notna(c)]))
                else:
                    new_cols.append(str(col))
            df.columns = new_cols
        return df
    
    def _fetch_vn100_data(self) -> List[StockInfo]:
        """
        Synchronous method to fetch VN-100 data using vnstock.
        Called in thread pool executor to avoid blocking.
        """
        from vnstock import Listing, Trading
        
        try:
            # Get all stock symbols
            listing = Listing()
            all_symbols_df = listing.all_symbols()
            all_symbols = all_symbols_df['symbol'].tolist()
            
            # Get price board for stocks in batches
            trading = Trading(source='VCI')
            stocks_data = []
            batch_size = 100
            
            # Process symbols to find HSX (HOSE) stocks with market cap
            for i in range(0, min(len(all_symbols), 600), batch_size):
                batch = all_symbols[i:i + batch_size]
                try:
                    price_board = trading.price_board(batch)
                    
                    if price_board is not None and not price_board.empty:
                        # Flatten multi-level column names
                        price_board = self._flatten_columns(price_board)
                        
                        # Filter for HSX exchange only (VN-100 is HOSE/HSX-based)
                        if 'listing_exchange' in price_board.columns:
                            price_board = price_board[price_board['listing_exchange'] == 'HSX']
                        
                        for _, row in price_board.iterrows():
                            ticker = row.get('listing_symbol', '')
                            if not ticker:
                                continue
                            
                            # Get price (in VND)
                            price = 0
                            if 'match_match_price' in row.index:
                                try:
                                    price = float(row['match_match_price'])
                                except (ValueError, TypeError):
                                    pass
                            
                            # Calculate market cap: price * listed_share
                            # Returns value in VND
                            market_cap = 0
                            if 'listing_listed_share' in row.index:
                                try:
                                    listed_shares = float(row['listing_listed_share'])
                                    # Market cap in billion VND = (price * shares) / 1e9
                                    market_cap = (price * listed_shares) / 1e9
                                except (ValueError, TypeError):
                                    pass
                            
                            if ticker and price > 0:
                                stocks_data.append(StockInfo(
                                    ticker=str(ticker),
                                    price=price,
                                    market_cap=round(market_cap, 2)
                                ))
                                
                except Exception as e:
                    print(f"Error fetching batch {i}: {e}")
                    continue
            
            # Sort by market cap descending and take top 100
            stocks_data.sort(key=lambda x: x.market_cap, reverse=True)
            return stocks_data[:100]
            
        except Exception as e:
            print(f"Error fetching VN100 data: {e}")
            import traceback
            traceback.print_exc()
            return []


# Singleton instance
vnstock_service = VnstockService()
