"""
Service for interacting with vnstock library to fetch Vietnam stock market data.
"""
from typing import List
from dataclasses import dataclass
import asyncio
import pandas as pd


@dataclass
class StockInfo:
    """Stock information data class."""
    ticker: str
    price: float
    market_cap: float  # In billion VND (tỷ đồng)


class VnstockService:
    """Service class for vnstock operations."""
    
    def __init__(self):
        pass
    
    async def get_vn100_stocks(self) -> List[StockInfo]:
        """
        Fetch VN-100 stocks (top 100 by market cap) with price and market cap data.
        
        Returns:
            List of StockInfo objects with ticker, price, and market_cap
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._fetch_vn100_data)
        return result
    
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
