"""
Service for interacting with vnstock library to fetch Vietnam stock market data.
"""
from typing import List, Dict, Callable, TypeVar, Any
from dataclasses import dataclass
import asyncio
from datetime import datetime, date, timedelta
import time
import pandas as pd
from sqlalchemy import select, and_
from app.db.database import async_session
from app.db.models import StockCompany, StockDailyPrice, StockIndex
from app.core.config import settings

T = TypeVar('T')


def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 25.0,  # VCI rate limit is typically 21 seconds
    backoff_multiplier: float = 2.0,
    rate_limit_keywords: List[str] = None
) -> T:
    """
    Execute a function with retry logic and exponential backoff for rate limits.
    
    Args:
        func: Callable to execute
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds after rate limit hit
        backoff_multiplier: Multiplier for delay on each subsequent retry
        rate_limit_keywords: Keywords in error message that indicate rate limit
        
    Returns:
        Result of the function call
        
    Raises:
        Last exception if all retries failed
    """
    if rate_limit_keywords is None:
        rate_limit_keywords = ["Rate limit", "rate limit", "429", "quá nhiều request"]
    
    last_exception = None
    delay = initial_delay
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except (SystemExit, Exception) as e:
            last_exception = e
            error_msg = str(e)
            
            # Check if it's a rate limit error
            is_rate_limit = any(keyword in error_msg for keyword in rate_limit_keywords)
            
            if is_rate_limit and attempt < max_retries:
                print(f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
                delay *= backoff_multiplier
            else:
                # Not a rate limit error or out of retries
                raise
    
    raise last_exception


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
    charter_capital: float = 0.0  # In billion VND
    pe_ratio: float | None = None
    accumulated_value: float | None = None  # In billion VND
    price_change_24h: float | None = None  # Percentage
    price_change_1w: float | None = None  # Percentage
    price_change_1m: float | None = None  # Percentage
    price_change_1y: float | None = None  # Percentage


class VnstockService:
    """Service class for vnstock operations."""
    
    # Valid groups supported by symbols_by_group
    VALID_GROUPS = {
        'HOSE', 'VN30', 'VNMidCap', 'VNSmallCap', 'VNAllShare', 'VN100', 
        'ETF', 'HNX', 'HNX30', 'HNXCon', 'HNXFin', 'HNXLCap', 'HNXMSCap', 
        'HNXMan', 'UPCOM', 'FU_INDEX', 'FU_BOND', 'BOND', 'CW'
    }

    def __init__(self):
        self._enriching_tickers = set()
        # Fund caches: {key: (data, timestamp)}
        self._fund_listing_cache = {}
        self._fund_nav_cache = {}
        self._fund_top_holding_cache = {}
        self._fund_industry_holding_cache = {}
        self._fund_asset_holding_cache = {}
        
        # Cache TTLs in seconds
        self._FUND_LISTING_TTL = 3600  # 1 hour
        self._FUND_DETAILS_TTL = 1800  # 30 minutes

    def _is_cache_valid(self, cache: Dict, key: str, ttl: int) -> bool:
        """Check if cache entry exists and is not expired."""
        if key in cache:
            data, timestamp = cache[key]
            if time.time() - timestamp < ttl:
                return True
        return False
    
    async def sync_indices(self) -> None:
        """
        Fetch all indices from vnstock and update the database.
        """
        loop = asyncio.get_event_loop()
        indices_df = await loop.run_in_executor(None, self._fetch_all_indices_from_lib)
        
        # Define manual indices (Exchanges/Groups that are not in all_indices but are supported)
        manual_indices = [
            {'symbol': 'HOSE', 'name': 'HOSE Exchange', 'group': 'Exchange'},
            {'symbol': 'HNX', 'name': 'HNX Exchange', 'group': 'Exchange'},
            {'symbol': 'UPCOM', 'name': 'UPCOM Exchange', 'group': 'Exchange'},
        ]

        async with async_session() as session:
            count = 0
            
            # Process manual indices
            for item in manual_indices:
                try:
                    symbol = item['symbol']
                    # Ensure supported (it should be, since we manually added it)
                    if symbol not in self.VALID_GROUPS:
                        continue
                        
                    # Upsert
                    stmt = select(StockIndex).where(StockIndex.symbol == symbol)
                    result = await session.execute(stmt)
                    existing = result.scalar_one_or_none()
                    
                    if existing:
                        existing.name = item['name']
                        existing.group = item['group']
                        existing.updated_at = datetime.utcnow()
                    else:
                        new_index = StockIndex(
                            symbol=symbol,
                            name=item['name'],
                            group=item['group']
                        )
                        session.add(new_index)
                    count += 1
                except Exception as e:
                    print(f"Error processing manual index {item['symbol']}: {e}")

            # Process fetched indices
            if indices_df is not None and not indices_df.empty:
                for _, row in indices_df.iterrows():
                    try:
                        symbol = row.get('symbol')
                        if not symbol:
                            continue

                        # Check if supported
                        group_code = self._get_group_code_for_index(symbol)
                        if group_code not in self.VALID_GROUPS:
                            continue
                            
                        # Prepare data
                        name = row.get('name', '')
                        if 'full_name' in row:
                            name = row['full_name']
                        elif not name and 'index_name' in row:
                            name = row['index_name']
                        
                        if not name:
                            name = symbol

                        group = row.get('group', None)
                        description = row.get('description', None)
                        
                        # Upsert logic
                        stmt = select(StockIndex).where(StockIndex.symbol == symbol)
                        result = await session.execute(stmt)
                        existing = result.scalar_one_or_none()
                        
                        if existing:
                            existing.name = name
                            existing.group = group
                            existing.description = description
                            existing.updated_at = datetime.utcnow()
                        else:
                            new_index = StockIndex(
                                symbol=symbol,
                                name=name,
                                group=group,
                                description=description
                            )
                            session.add(new_index)
                        count += 1
                    except Exception as e:
                        print(f"Error processing index {row.get('symbol')}: {e}")
                    
            await session.commit()
            print(f"Synced {count} supported indices to database.")

    def _fetch_all_indices_from_lib(self) -> pd.DataFrame:
        """Fetch all indices from vnstock library synchronously."""
        from vnstock import Listing
        return Listing().all_indices()

    async def get_indices(self) -> List[StockIndex]:
        """
        Get all available indices from the database.
        """
        async with async_session() as session:
            stmt = select(StockIndex).order_by(StockIndex.symbol)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    # Index name mapping for display
    INDEX_NAMES = {
        'VNINDEX': 'VN-Index',
        'HNXINDEX': 'HNX-Index',
        'UPCOMINDEX': 'UPCOM-Index',
        'VN30': 'VN30',
        'HNX30': 'HNX30',
    }

    async def get_index_values(self, symbols: List[str] = None) -> List[IndexValue]:
        """
        Fetch latest values for major market indices.
        
        Args:
            symbols: List of index symbols. Defaults to main indices if not specified.
            
        Returns:
            List of IndexValue objects with current price and change data.
        """
        if symbols is None:
            symbols = ['VNINDEX', 'HNXINDEX', 'UPCOMINDEX', 'VN30', 'HNX30']
        
        loop = asyncio.get_event_loop()
        results = []
        
        for symbol in symbols:
            try:
                index_data = await loop.run_in_executor(None, self._fetch_index_value_sync, symbol)
                if index_data:
                    results.append(index_data)
            except Exception as e:
                print(f"Error fetching index value for {symbol}: {e}")
        
        return results

    def _fetch_index_value_sync(self, symbol: str) -> IndexValue | None:
        """
        Fetch latest value for a single index synchronously.
        """
        from vnstock import Vnstock
        
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            vs = Vnstock(symbol=symbol, source='VCI')
            stock = vs.stock()
            df = stock.quote.history(start=week_ago, end=today)
            
            if df is not None and not df.empty:
                last_row = df.iloc[-1]
                open_price = float(last_row['open'])
                close_price = float(last_row['close'])
                change_value = close_price - open_price
                change_pct = (change_value / open_price) * 100 if open_price > 0 else 0
                
                return IndexValue(
                    symbol=symbol,
                    name=self.INDEX_NAMES.get(symbol, symbol),
                    value=round(close_price, 2),
                    change=round(change_pct, 2),
                    change_value=round(change_value, 2)
                )
        except Exception as e:
            print(f"Error fetching index value for {symbol}: {e}")
        
        return None


    async def get_index_stocks(self, index_symbol: str, limit: int = 100) -> List[StockInfo]:
        """
        Fetch stocks for a specific index with price and market cap data.
        
        Args:
            index_symbol: The symbol of the index (e.g., "VN100", "VN30", "VNDIAMOND")
            limit: Maximum number of stocks to return
            
        Returns:
            List of StockInfo objects
        """
        loop = asyncio.get_event_loop()
        stocks = await loop.run_in_executor(None, self._fetch_index_data, index_symbol, limit)
        
        # Launch background task for enrichment
        asyncio.create_task(self._enrich_stocks_with_metadata(stocks))
        
        # Apply current cache to the response immediately
        return await self._apply_cache_to_stocks(stocks)

    async def get_industry_list(self) -> List[Dict[str, str]]:
        """
        Fetch all ICB level 2 industries.
        """
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, self._fetch_industries_sync)
        if df is not None and not df.empty:
            # Filter for level 2 industries
            l2_df = df[df['level'] == 2]
            return l2_df[['icb_name', 'en_icb_name', 'icb_code']].to_dict('records')
        return []

    def _fetch_industries_sync(self) -> pd.DataFrame:
        """Fetch industries synchronously."""
        from vnstock import Listing
        return Listing().industries_icb()

    async def get_industry_stocks(self, industry_name: str, limit: int = 100) -> List[StockInfo]:
        """
        Fetch stocks for a specific ICB industry.
        """
        loop = asyncio.get_event_loop()
        stocks = await loop.run_in_executor(None, self._fetch_industry_data, industry_name, limit)
        
        # Launch background task for enrichment
        asyncio.create_task(self._enrich_stocks_with_metadata(stocks))
        
        # Apply current cache to the response immediately
        return await self._apply_cache_to_stocks(stocks)

    async def get_vn100_stocks(self) -> List[StockInfo]:
        """
        Fetch VN-100 stocks (top 100 by market cap) with price and market cap data.
        
        Returns:
            List of StockInfo objects include company_name
        """
        return await self.get_index_stocks("VN100", settings.vn100_limit)

    async def get_vn30_stocks(self) -> List[StockInfo]:
        """
        Fetch VN-30 stocks (top 30 by market cap and liquidity) with price and market cap data.
        
        Returns:
            List of StockInfo objects include company_name
        """
        return await self.get_index_stocks("VN30", settings.vn30_limit)

    async def _apply_cache_to_stocks(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """Apply currently cached data to stocks without fetching new data."""
        if not stocks:
            return []
            
        tickers = [s.ticker for s in stocks]
        async with async_session() as session:
            stmt = select(StockCompany).where(StockCompany.symbol.in_(tickers))
            result = await session.execute(stmt)
            cached_data = {c.symbol: c for c in result.scalars().all()}
            
            for stock in stocks:
                if stock.ticker in cached_data:
                    company = cached_data[stock.ticker]
                    stock.company_name = company.company_name
                    if stock.charter_capital == 0 and company.charter_capital:
                        stock.charter_capital = company.charter_capital
                    if stock.pe_ratio is None and company.pe_ratio:
                        stock.pe_ratio = company.pe_ratio
        return stocks

    async def _enrich_stocks_with_metadata(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """Add company names and financial metadata to stock info objects, using DB cache."""
        if not stocks:
            return []

        tickers = [s.ticker for s in stocks 
                   if s.ticker not in self._enriching_tickers]
        if not tickers:
            return stocks

        now = datetime.utcnow()
        stale_threshold = now - timedelta(days=7) 
        error_stale_threshold = now - timedelta(hours=1) # Retry missing PE after 1 hour

        async with async_session() as session:
            # Try to get from DB
            stmt = select(StockCompany).where(StockCompany.symbol.in_(tickers))
            result = await session.execute(stmt)
            cached_data = {c.symbol: c for c in result.scalars().all()}
            
            # Identify what's missing or stale
            tickers_needing_name = [t for t in tickers if t not in cached_data]
            tickers_needing_finance = [
                t for t in tickers 
                if (t not in cached_data or 
                    cached_data[t].updated_at is None or 
                    (cached_data[t].pe_ratio is None and cached_data[t].updated_at < error_stale_threshold) or
                    cached_data[t].updated_at < stale_threshold)
            ]
            
            # Fetch missing names if needed
            if tickers_needing_name:
                loop = asyncio.get_event_loop()
                all_symbols_df = await loop.run_in_executor(None, self._fetch_all_symbols)
                
                for _, row in all_symbols_df.iterrows():
                    symbol = row['symbol']
                    name = row['organ_name']
                    if symbol in tickers_needing_name:
                        if symbol not in cached_data:
                            new_company = StockCompany(symbol=symbol, company_name=name)
                            session.add(new_company)
                            cached_data[symbol] = new_company
                        else:
                            cached_data[symbol].company_name = name
            
            # Fetch missing/stale financial data if needed
            if tickers_needing_finance:
                # Limit batch size to avoid long hangs in one request
                batch_limit = 50
                tickers_to_fetch = [t for t in tickers_needing_finance if t not in self._enriching_tickers][:batch_limit]
                
                if not tickers_to_fetch:
                    return stocks

                # Mark as enriching to avoid multiple tasks for same symbols
                self._enriching_tickers.update(tickers_to_fetch)
                
                try:
                    print(f"Fetching financial metadata for {len(tickers_to_fetch)}/{len(tickers_needing_finance)} stocks in background...")
                    loop = asyncio.get_event_loop()
                    
                    # Fetch one by one and commit incrementally
                    for symbol in tickers_to_fetch:
                        try:
                            # Add a small delay between symbols
                            await asyncio.sleep(1.0)
                            data = await loop.run_in_executor(None, self._fetch_stock_finance_sync, symbol)
                            
                            if data and symbol in cached_data:
                                cached_data[symbol].pe_ratio = data.get('pe_ratio')
                                cached_data[symbol].updated_at = now
                                await session.commit()
                            elif symbol in cached_data:
                                # Still update to avoid retrying immediately, but mark as updated now
                                cached_data[symbol].updated_at = now
                                await session.commit()
                        except Exception as e:
                            print(f"Error enriching {symbol}: {e}")
                            if "Rate limit" in str(e) or "429" in str(e) or "quá nhiều" in str(e):
                                break
                finally:
                    # Clean up
                    for t in tickers_to_fetch:
                        self._enriching_tickers.discard(t)
            
            await session.commit()
            
            for stock in stocks:
                if stock.ticker in cached_data:
                    company = cached_data[stock.ticker]
                    stock.company_name = company.company_name
                    # Use cached value if real-time value is missing
                    if stock.charter_capital == 0 and company.charter_capital:
                        stock.charter_capital = company.charter_capital
                    if stock.pe_ratio is None and company.pe_ratio:
                        stock.pe_ratio = company.pe_ratio
                
        return stocks

    def _fetch_stock_finance_sync(self, symbol: str) -> Dict | None:
        """
        Fetch financial metadata for a single symbol synchronously.
        Uses retry mechanism with exponential backoff for rate limits.
        """
        from vnstock import Vnstock
        
        def fetch_finance():
            # We only need the latest ratio
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            ratio = s.finance.ratio(period='quarter', lang='vi')
            if ratio is not None and not ratio.empty:
                # Try multiple possible column names for P/E
                pe = None
                if ('Chỉ tiêu định giá', 'P/E') in ratio.columns:
                    pe = ratio.iloc[0].get(('Chỉ tiêu định giá', 'P/E'))
                else:
                    # Look for any column containing 'P/E'
                    for col in ratio.columns:
                        col_str = str(col)
                        if 'P/E' in col_str:
                            pe = ratio.iloc[0].get(col)
                            break
                
                return {
                    'pe_ratio': float(pe) if pd.notna(pe) else None
                }
            return None
        
        try:
            return retry_with_backoff(fetch_finance, max_retries=3, initial_delay=25.0)
        except Exception as e:
            print(f"Error fetching financial metadata for {symbol}: {e}")
            raise e

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
    
    def _get_group_code_for_index(self, index_symbol: str) -> str:
        """
        Map index symbol from all_indices() to group code expected by symbols_by_group().
        """
        mapping = {
            'VN30': 'VN30',
            'VN100': 'VN100',
            'VNMID': 'VNMidCap',
            'VNSML': 'VNSmallCap',
            'VNALL': 'VNAllShare',
            'HNX30': 'HNX30',
            # Add more mappings as needed based on valid groups:
            # ['HOSE', 'VN30', 'VNMidCap', 'VNSmallCap', 'VNAllShare', 'VN100', 
            #  'ETF', 'HNX', 'HNX30', 'HNXCon', 'HNXFin', 'HNXLCap', 'HNXMSCap', 
            #  'HNXMan', 'UPCOM', 'FU_INDEX', 'FU_BOND', 'BOND', 'CW']
        }
        return mapping.get(index_symbol, index_symbol)

    def _fetch_index_data(self, index_name: str, limit: int) -> List[StockInfo]:
        """
        Synchronous method to fetch index data (VN100, VN30, etc.) using vnstock.
        Called in thread pool executor to avoid blocking.
        """
        from vnstock import Listing
        
        try:
            # Map index name to group code
            group_code = self._get_group_code_for_index(index_name)
            
            # Get stock symbols for the specified group
            listing = Listing()
            try:
                symbols_df = listing.symbols_by_group(group_code)
            except ValueError as e:
                print(f"Warning: Group '{group_code}' (mapped from '{index_name}') not supported by symbols_by_group: {e}")
                return []
            except Exception as e:
                print(f"Error fetching symbols for group '{group_code}': {e}")
                return []

            if symbols_df is None or symbols_df.empty:
                print(f"Warning: Could not fetch symbols for {group_code} group.")
                return []
            else:
                # The series returned by symbols_by_group contains the symbols
                symbols = symbols_df.tolist()
            
            return self._fetch_symbols_data(symbols, limit)
            
        except Exception as e:
            print(f"Error fetching {index_name} data: {e}")
            return []

    def _fetch_industry_data(self, industry_name: str, limit: int) -> List[StockInfo]:
        """
        Synchronous method to fetch industry data using vnstock.
        """
        from vnstock import Listing
        try:
            listing = Listing()
            # Get all symbols with industry info
            df = listing.symbols_by_industries()
            if df is not None and not df.empty:
                # Filter by icb_name2 (Level 2) or icb_name3/4 if needed
                # We'll match against any of them for flexibility
                cols_to_check = ['icb_name2', 'icb_name3', 'icb_name4']
                mask = pd.Series([False] * len(df))
                for col in cols_to_check:
                    if col in df.columns:
                        mask |= (df[col] == industry_name)
                
                filtered_df = df[mask]
                symbols = filtered_df['symbol'].tolist()
                
                return self._fetch_symbols_data(symbols, limit)
            return []
        except Exception as e:
            print(f"Error fetching industry {industry_name} data: {e}")
            return []

    def _fetch_symbols_data(self, symbols: List[str], limit: int) -> List[StockInfo]:
        """
        Generic method to fetch price and market cap data for a list of symbols.
        """
        from vnstock import Trading
        try:
            # Get price board for stocks in batches
            trading = Trading(source='VCI')
            stocks_data = []
            batch_size = 50 # Smaller batch for more reliable fetching
            
            # Process symbols
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                try:
                    price_board = trading.price_board(batch)
                    
                    # Add delay between batches
                    time.sleep(1.0) # Slightly longer delay for safety
                    
                    if price_board is not None and not price_board.empty:
                        # Flatten multi-level column names
                        price_board = self._flatten_columns(price_board)
                        
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
                            listed_shares = 0
                            if 'listing_listed_share' in row.index:
                                try:
                                    listed_shares = float(row['listing_listed_share'])
                                    # Market cap in billion VND = (price * shares) / 1e9
                                    market_cap = (price * listed_shares) / 1e9
                                except (ValueError, TypeError):
                                    pass
                            
                            # Get charter capital (in billion VND)
                            # Use listing_charter_capital if available, else estimate from listed shares
                            charter_capital = 0.0
                            if 'listing_charter_capital' in row.index:
                                try:
                                    charter_capital = float(row['listing_charter_capital']) / 1e9
                                except (ValueError, TypeError):
                                    pass
                            
                            # Get P/E ratio
                            pe_ratio = None
                            if 'financial_pe' in row.index:
                                try:
                                    pe_val = row['financial_pe']
                                    if pd.notna(pe_val) and pe_val != 0:
                                        pe_ratio = float(pe_val)
                                except (ValueError, TypeError):
                                    pass

                            # Get accumulated trading value (in billion VND)
                            # API returns value in Million VND, divide by 1000 to get Billion VND
                            accumulated_value = None
                            if 'match_accumulated_value' in row.index:
                                try:
                                    acc_val = row['match_accumulated_value']
                                    if pd.notna(acc_val):
                                        accumulated_value = float(acc_val) / 1e3
                                except (ValueError, TypeError):
                                    pass

                            # Get 24h price change percentage
                            price_change_24h = None
                            if 'match_price_change_ratio' in row.index:
                                try:
                                    change_val = row['match_price_change_ratio']
                                    if pd.notna(change_val):
                                        # Convert to percentage (value is already ratio)
                                        price_change_24h = float(change_val) * 100
                                except (ValueError, TypeError):
                                    pass
                            
                            # Fallback: calculate from reference price if match_price_change_ratio is missing or zero
                            if (price_change_24h is None or price_change_24h == 0) and 'listing_ref_price' in row.index:
                                try:
                                    ref_price = float(row['listing_ref_price'])
                                    if ref_price > 0 and price > 0:
                                        price_change_24h = ((price - ref_price) / ref_price) * 100
                                except (ValueError, TypeError):
                                    pass
                            
                            # Fallback for charter capital: shares * 10,000 (par value) / 1e9
                            if charter_capital == 0 and listed_shares > 0:
                                charter_capital = (listed_shares * 10000) / 1e9

                            if ticker and price > 0:
                                stocks_data.append(StockInfo(
                                    ticker=str(ticker),
                                    price=price,
                                    market_cap=round(market_cap, 2),
                                    charter_capital=round(charter_capital, 2),
                                    pe_ratio=round(pe_ratio, 2) if pe_ratio is not None else None,
                                    accumulated_value=round(accumulated_value, 2) if accumulated_value is not None else None,
                                    price_change_24h=round(price_change_24h, 2) if price_change_24h is not None else None
                                ))
                                
                except Exception as e:
                    print(f"Error fetching batch {i}: {e}")
                    continue
            
            # Sort by market cap descending and take the requested limit
            stocks_data.sort(key=lambda x: x.market_cap, reverse=True)
            top_stocks = stocks_data[:limit]
            
            # Fetch historical price changes for top stocks
            top_stocks = self._enrich_with_price_changes(top_stocks)
            
            return top_stocks
            
        except Exception as e:
            print(f"Error fetching symbols data: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _enrich_with_price_changes(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """
        Enrich stock data with historical price changes (1w, 1m, 1y).
        """
        return self._enrich_with_price_changes_sync(stocks)
    
    def _enrich_with_price_changes_sync(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """
        Synchronous fallback for price change enrichment.
        Queries DB cache directly, fetches missing from API.
        """
        from vnstock import Vnstock
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        
        # Calculate target dates
        today = datetime.now().date()
        target_dates = {
            '1w': today - timedelta(days=7),
            '1m': today - timedelta(days=30),
            '1y': today - timedelta(days=365),
        }
        
        # Use sync connection for DB lookup  
        sync_url = settings.database_url.replace('+asyncpg', '')
        engine = create_engine(sync_url)
        
        symbols = [s.ticker[:3] for s in stocks]
        
        with Session(engine) as session:
            # Get cached prices for all symbols at target dates (with some tolerance)
            cached_prices = self._get_cached_prices_sync(session, symbols, target_dates)
            
            # Find symbols missing cache data
            symbols_needing_fetch = set()
            for symbol in symbols:
                for period in ['1w', '1m', '1y']:
                    if (symbol, period) not in cached_prices:
                        symbols_needing_fetch.add(symbol)
            
            # Fetch missing data from API and save to DB
            if symbols_needing_fetch:
                # Limit how many symbols we fetch history for in one request
                # To avoid hitting API limits and long timeouts
                max_history_fetch = 100
                symbols_to_fetch = list(symbols_needing_fetch)[:max_history_fetch]
                print(f"Fetching historical data for {len(symbols_to_fetch)}/{len(symbols_needing_fetch)} symbols (capped at {max_history_fetch})...")
                self._fetch_and_cache_history_sync(session, symbols_to_fetch)
                # Re-query cached prices after fetch
                cached_prices = self._get_cached_prices_sync(session, symbols, target_dates)
        
        engine.dispose()
        
        # Calculate price changes from cached data
        for stock in stocks:
            symbol = stock.ticker[:3]
            # Convert current price to same units as history (1,000 VND)
            current_price_unit = stock.price / 1000
            
            # 1 week change
            if (symbol, '1w') in cached_prices and cached_prices[(symbol, '1w')] > 0:
                week_price = cached_prices[(symbol, '1w')]
                stock.price_change_1w = round(((current_price_unit - week_price) / week_price) * 100, 2)
            
            # 1 month change
            if (symbol, '1m') in cached_prices and cached_prices[(symbol, '1m')] > 0:
                month_price = cached_prices[(symbol, '1m')]
                stock.price_change_1m = round(((current_price_unit - month_price) / month_price) * 100, 2)
            
            # 1 year change
            if (symbol, '1y') in cached_prices and cached_prices[(symbol, '1y')] > 0:
                year_price = cached_prices[(symbol, '1y')]
                stock.price_change_1y = round(((current_price_unit - year_price) / year_price) * 100, 2)
        
        return stocks
    
    def _get_cached_prices_sync(self, session, symbols: List[str], target_dates: Dict[str, date]) -> Dict[tuple, float]:
        """
        Get cached prices for given symbols at target dates.
        Returns dict of (symbol, period) -> close_price
        """
        result = {}
        
        for period, target_date in target_dates.items():
            # Look for prices within 7 days of target (to handle weekends/holidays)
            min_date = target_date - timedelta(days=7)
            max_date = target_date + timedelta(days=1)
            
            stmt = select(StockDailyPrice).where(
                and_(
                    StockDailyPrice.symbol.in_(symbols),
                    StockDailyPrice.date >= min_date,
                    StockDailyPrice.date <= max_date
                )
            ).order_by(StockDailyPrice.date.desc())
            
            rows = session.execute(stmt).scalars().all()
            
            # Group by symbol and take the closest to target date
            symbol_prices = {}
            for row in rows:
                if row.symbol not in symbol_prices:
                    symbol_prices[row.symbol] = row.close
            
            for symbol, close in symbol_prices.items():
                result[(symbol, period)] = close
        
        return result
    
    def _fetch_and_cache_history_sync(self, session, symbols: List[str]) -> None:
        """
        Fetch historical data for given symbols from vnstock API and cache to DB.
        Uses retry mechanism with exponential backoff to handle rate limits.
        """
        from vnstock import Vnstock
        
        today = datetime.now()
        one_year_ago = today - timedelta(days=400)  # Fetch extra buffer
        
        for symbol in symbols:
            try:
                # Use retry mechanism for API call
                def fetch_history():
                    s = Vnstock().stock(symbol=symbol, source='VCI')
                    return s.quote.history(
                        start=one_year_ago.strftime('%Y-%m-%d'),
                        end=today.strftime('%Y-%m-%d'),
                        interval='1D'
                    )
                
                hist = retry_with_backoff(fetch_history, max_retries=3, initial_delay=25.0)
                
                # Add delay between symbols to proactively avoid rate limits
                time.sleep(1.0)
                
                if hist is not None and not hist.empty:
                    # Insert each day's data
                    for _, row in hist.iterrows():
                        try:
                            price_date = pd.to_datetime(row['time']).date()
                            
                            # Check if already exists
                            existing = session.execute(
                                select(StockDailyPrice).where(
                                    and_(
                                        StockDailyPrice.symbol == symbol,
                                        StockDailyPrice.date == price_date
                                    )
                                )
                            ).scalar_one_or_none()
                            
                            if not existing:
                                price_record = StockDailyPrice(
                                    symbol=symbol,
                                    date=price_date,
                                    open=float(row.get('open', 0)) if pd.notna(row.get('open')) else None,
                                    high=float(row.get('high', 0)) if pd.notna(row.get('high')) else None,
                                    low=float(row.get('low', 0)) if pd.notna(row.get('low')) else None,
                                    close=float(row['close']),
                                    volume=int(row.get('volume', 0)) if pd.notna(row.get('volume')) else None
                                )
                                session.add(price_record)
                        except Exception as e:
                            print(f"Error inserting price for {symbol} on {row.get('time')}: {e}")
                            continue
                    
                    session.commit()
                    print(f"Cached {len(hist)} price records for {symbol}")
                    
            except Exception as e:
                print(f"Error fetching history for {symbol}: {e}")
                continue
    


    async def get_income_statement(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        """
        Fetch income statement data for a given stock symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'VIC', 'VNM')
            period: 'quarter' or 'year'
            lang: 'en' or 'vi'
            
        Returns:
            List of income statement records with period data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_income_statement_sync, symbol, period, lang)
    
    def _fetch_income_statement_sync(self, symbol: str, period: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch income statement synchronously."""
        from vnstock import Vnstock
        
        try:
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            df = s.finance.income_statement(period=period, lang=lang)
            
            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                # Convert to list of dicts, handling NaN values
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except Exception as e:
            print(f"Error fetching income statement for {symbol}: {e}")
            return []
    
    async def get_balance_sheet(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        """
        Fetch balance sheet data for a given stock symbol.
        
        Args:
            symbol: Stock ticker symbol
            period: 'quarter' or 'year'
            lang: 'en' or 'vi'
            
        Returns:
            List of balance sheet records with period data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_balance_sheet_sync, symbol, period, lang)
    
    def _fetch_balance_sheet_sync(self, symbol: str, period: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch balance sheet synchronously."""
        from vnstock import Vnstock
        
        try:
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            df = s.finance.balance_sheet(period=period, lang=lang)
            
            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except Exception as e:
            print(f"Error fetching balance sheet for {symbol}: {e}")
            return []
    
    async def get_cash_flow(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        """
        Fetch cash flow statement data for a given stock symbol.
        
        Args:
            symbol: Stock ticker symbol
            period: 'quarter' or 'year'
            lang: 'en' or 'vi'
            
        Returns:
            List of cash flow records with period data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_cash_flow_sync, symbol, period, lang)
    
    def _fetch_cash_flow_sync(self, symbol: str, period: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch cash flow synchronously."""
        from vnstock import Vnstock
        
        try:
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            df = s.finance.cash_flow(period=period, lang=lang)
            
            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except Exception as e:
            print(f"Error fetching cash flow for {symbol}: {e}")
            return []
    
    async def get_financial_ratios(self, symbol: str, period: str = 'quarter', lang: str = 'en') -> List[Dict[str, Any]]:
        """
        Fetch financial ratios for a given stock symbol.
        Includes P/E, P/B, P/S, ROE, ROA, etc.
        
        Args:
            symbol: Stock ticker symbol
            period: 'quarter' or 'year'
            lang: 'en' or 'vi'
            
        Returns:
            List of financial ratio records with period data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_financial_ratios_sync, symbol, period, lang)
    
    def _fetch_financial_ratios_sync(self, symbol: str, period: str, lang: str) -> List[Dict[str, Any]]:
        """Fetch financial ratios synchronously."""
        from vnstock import Vnstock
        
        try:
            s = Vnstock().stock(symbol=symbol[:3], source='VCI')
            df = s.finance.ratio(period=period, lang=lang)
            
            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except Exception as e:
            print(f"Error fetching financial ratios for {symbol}: {e}")
            return []



    async def get_company_overview(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch company overview for a given stock symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_company_overview_sync, symbol)

    def _fetch_company_overview_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch company overview synchronously."""
        from vnstock import Company
        try:
            c = Company(symbol=symbol[:3], source='VCI')
            df = c.overview()
            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except Exception as e:
            print(f"Error fetching company overview for {symbol}: {e}")
            return []

    async def get_shareholders(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch shareholders for a given stock symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_shareholders_sync, symbol)

    def _fetch_shareholders_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch shareholders synchronously."""
        from vnstock import Company
        try:
            c = Company(symbol=symbol[:3], source='VCI')
            df = c.shareholders()
            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except Exception as e:
            print(f"Error fetching shareholders for {symbol}: {e}")
            return []

    async def get_officers(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch officers for a given stock symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_officers_sync, symbol)

    def _fetch_officers_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch officers synchronously."""
        from vnstock import Company
        try:
            c = Company(symbol=symbol[:3], source='VCI')
            df = c.officers()
            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except Exception as e:
            print(f"Error fetching officers for {symbol}: {e}")
            return []

    async def get_subsidiaries(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch subsidiaries for a given stock symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_subsidiaries_sync, symbol)

    def _fetch_subsidiaries_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch subsidiaries synchronously."""
        from vnstock import Company
        try:
            c = Company(symbol=symbol[:3], source='VCI')
            df = c.subsidiaries()
            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []
        except Exception as e:
            print(f"Error fetching subsidiaries for {symbol}: {e}")
            return []

    async def get_volume_history(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Fetch volume history for a given stock symbol.

        Args:
            symbol: Stock ticker symbol
            days: Number of days to fetch (default: 30)

        Returns:
            Dict with symbol, company_name, and data (list of VolumeDataPoint dicts)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_volume_history_sync, symbol, days)

    def _fetch_volume_history_sync(self, symbol: str, days: int) -> Dict[str, Any]:
        """Fetch volume history synchronously."""
        from vnstock import Vnstock
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        symbol_clean = symbol[:3]
        company_name = symbol_clean

        # Get company name from database
        sync_url = settings.database_url.replace('+asyncpg', '')
        engine = create_engine(sync_url)

        try:
            with Session(engine) as session:
                # Get company name
                stmt = select(StockCompany).where(StockCompany.symbol == symbol_clean)
                company = session.execute(stmt).scalar_one_or_none()
                if company:
                    company_name = company.company_name

                # Calculate date range
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=days + 10)  # Extra buffer for weekends/holidays

                # Query cached data
                stmt = select(StockDailyPrice).where(
                    and_(
                        StockDailyPrice.symbol == symbol_clean,
                        StockDailyPrice.date >= start_date,
                        StockDailyPrice.date <= end_date
                    )
                ).order_by(StockDailyPrice.date.desc())

                cached_records = session.execute(stmt).scalars().all()

                # If we have enough cached data, use it
                if len(cached_records) >= days:
                    data = []
                    for record in sorted(cached_records[:days], key=lambda x: x.date):
                        value = None
                        if record.volume and record.close:
                            # Calculate value in billion VND: (volume * close_price_in_1000_VND) / 1e6
                            value = (record.volume * record.close) / 1e6

                        data.append({
                            'date': record.date.strftime('%Y-%m-%d'),
                            'volume': record.volume if record.volume else 0,
                            'value': round(value, 2) if value else None
                        })

                    engine.dispose()
                    return {
                        'symbol': symbol_clean,
                        'company_name': company_name,
                        'data': data
                    }

                # Otherwise, fetch from API and cache
                try:
                    s = Vnstock().stock(symbol=symbol_clean, source='VCI')
                    hist = s.quote.history(
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        interval='1D'
                    )

                    if hist is not None and not hist.empty:
                        # Cache the data
                        for _, row in hist.iterrows():
                            try:
                                price_date = pd.to_datetime(row['time']).date()

                                # Check if already exists
                                existing = session.execute(
                                    select(StockDailyPrice).where(
                                        and_(
                                            StockDailyPrice.symbol == symbol_clean,
                                            StockDailyPrice.date == price_date
                                        )
                                    )
                                ).scalar_one_or_none()

                                if not existing:
                                    price_record = StockDailyPrice(
                                        symbol=symbol_clean,
                                        date=price_date,
                                        open=float(row.get('open', 0)) if pd.notna(row.get('open')) else None,
                                        high=float(row.get('high', 0)) if pd.notna(row.get('high')) else None,
                                        low=float(row.get('low', 0)) if pd.notna(row.get('low')) else None,
                                        close=float(row['close']),
                                        volume=int(row.get('volume', 0)) if pd.notna(row.get('volume')) else None
                                    )
                                    session.add(price_record)
                            except Exception as e:
                                print(f"Error caching price for {symbol_clean} on {row.get('time')}: {e}")
                                continue

                        session.commit()

                        # Convert to response format
                        data = []
                        hist_sorted = hist.sort_values('time', ascending=True).tail(days)

                        for _, row in hist_sorted.iterrows():
                            volume = int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0
                            close = float(row['close']) if pd.notna(row['close']) else 0
                            value = None
                            if volume and close:
                                # Calculate value in billion VND
                                value = (volume * close) / 1e6

                            data.append({
                                'date': pd.to_datetime(row['time']).strftime('%Y-%m-%d'),
                                'volume': volume,
                                'value': round(value, 2) if value else None
                            })

                        engine.dispose()
                        return {
                            'symbol': symbol_clean,
                            'company_name': company_name,
                            'data': data
                        }

                except Exception as e:
                    print(f"Error fetching volume history for {symbol_clean}: {e}")

                engine.dispose()
                return {
                    'symbol': symbol_clean,
                    'company_name': company_name,
                    'data': []
                }

        except Exception as e:
            print(f"Error in volume history fetch: {e}")
            engine.dispose()
            return {
                'symbol': symbol_clean,
                'company_name': company_name,
                'data': []
            }
    async def get_fund_listing(self, fund_type: str = "") -> List[Dict[str, Any]]:
        """
        Fetch all available funds.
        """
        cache_key = fund_type or "all"
        if self._is_cache_valid(self._fund_listing_cache, cache_key, self._FUND_LISTING_TTL):
            return self._fund_listing_cache[cache_key][0]
            
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, self._fetch_fund_listing_sync, fund_type)
        
        # Update cache
        if data:
            self._fund_listing_cache[cache_key] = (data, time.time())
        return data

    def _fetch_fund_listing_sync(self, fund_type: str) -> List[Dict[str, Any]]:
        """Fetch fund listing synchronously."""
        from vnstock import Fund

        def fetch_listing():
            fund = Fund()
            df = fund.listing()
            if df is not None and not df.empty:
                # Filter by fund_type if provided
                if fund_type:
                    df = df[df['fund_type'] == fund_type]

                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    # Normalize field names for frontend
                    self._normalize_fund_field(record, 'symbol', ['short_name', 'fund_code'])
                    self._normalize_fund_field(record, 'name', ['name', 'short_name'])
                    self._normalize_fund_field(record, 'fund_owner', ['fund_owner_name', 'management_company'])
                    
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []

        try:
            return retry_with_backoff(fetch_listing, max_retries=2, initial_delay=30.0)
        except Exception as e:
            print(f"Error fetching fund listing: {e}")
            return []

    def _normalize_fund_field(self, record: dict, target_key: str, source_keys: List[str]) -> None:
        """
        Map vnstock field name to expected frontend field name.

        Args:
            record: Dictionary record to modify
            target_key: Desired field name for frontend
            source_keys: List of possible source field names from vnstock (in priority order)
        """
        if target_key in record:
            return
        for key in source_keys:
            if key in record:
                record[target_key] = record[key]
                break

    async def get_fund_nav_report(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch NAV (Net Asset Value) history for a specific fund.
        """
        if self._is_cache_valid(self._fund_nav_cache, symbol, self._FUND_DETAILS_TTL):
            return self._fund_nav_cache[symbol][0]

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, self._fetch_fund_nav_report_sync, symbol)
        
        if data:
            self._fund_nav_cache[symbol] = (data, time.time())
        return data

    def _fetch_fund_nav_report_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch fund NAV report synchronously."""
        from vnstock import Fund

        def fetch_nav():
            fund = Fund()
            df = fund.details.nav_report(symbol=symbol)
            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    # Normalize field names for frontend
                    self._normalize_fund_field(record, 'nav', ['nav_per_unit', 'nav'])
                    # Clean NaN values
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []

        try:
            return retry_with_backoff(fetch_nav, max_retries=2, initial_delay=30.0)
        except Exception as e:
            print(f"Error fetching NAV report for {symbol}: {e}")
            return []

    async def get_fund_top_holding(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch top stock holdings for a specific fund.
        """
        if self._is_cache_valid(self._fund_top_holding_cache, symbol, self._FUND_DETAILS_TTL):
            return self._fund_top_holding_cache[symbol][0]

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, self._fetch_fund_top_holding_sync, symbol)
        
        if data:
            self._fund_top_holding_cache[symbol] = (data, time.time())
        return data

    def _fetch_fund_top_holding_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch fund top holdings synchronously."""
        from vnstock import Fund

        def fetch_top_holding():
            fund = Fund()
            df = fund.details.top_holding(symbol=symbol)
            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    # Normalize field names for frontend
                    self._normalize_fund_field(record, 'ticker', ['stock_code', 'ticker', 'symbol'])
                    self._normalize_fund_field(record, 'allocation', ['net_asset_percent', 'allocation', 'weight', 'percentage'])
                    # Clean NaN values
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []

        try:
            return retry_with_backoff(fetch_top_holding, max_retries=2, initial_delay=30.0)
        except Exception as e:
            print(f"Error fetching top holdings for {symbol}: {e}")
            return []

    async def get_fund_industry_holding(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch industry allocation for a specific fund.
        """
        if self._is_cache_valid(self._fund_industry_holding_cache, symbol, self._FUND_DETAILS_TTL):
            return self._fund_industry_holding_cache[symbol][0]

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, self._fetch_fund_industry_holding_sync, symbol)
        
        if data:
            self._fund_industry_holding_cache[symbol] = (data, time.time())
        return data

    def _fetch_fund_industry_holding_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch fund industry holdings synchronously."""
        from vnstock import Fund

        def fetch_industry():
            fund = Fund()
            df = fund.details.industry_holding(symbol=symbol)
            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    # Normalize field names for frontend
                    self._normalize_fund_field(record, 'allocation', ['net_asset_percent', 'allocation', 'weight', 'percentage'])
                    # Clean NaN values
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []

        try:
            return retry_with_backoff(fetch_industry, max_retries=2, initial_delay=30.0)
        except Exception as e:
            print(f"Error fetching industry holdings for {symbol}: {e}")
            return []

    async def get_fund_asset_holding(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch asset type allocation for a specific fund.
        """
        if self._is_cache_valid(self._fund_asset_holding_cache, symbol, self._FUND_DETAILS_TTL):
            return self._fund_asset_holding_cache[symbol][0]

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, self._fetch_fund_asset_holding_sync, symbol)
        
        if data:
            self._fund_asset_holding_cache[symbol] = (data, time.time())
        return data

    def _fetch_fund_asset_holding_sync(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch fund asset holdings synchronously."""
        from vnstock import Fund

        def fetch_asset():
            fund = Fund()
            df = fund.details.asset_holding(symbol=symbol)
            if df is not None and not df.empty:
                df = self._flatten_columns(df)
                records = df.to_dict('records')
                for record in records:
                    # Normalize field names for frontend
                    self._normalize_fund_field(record, 'allocation', ['asset_percent', 'allocation', 'weight', 'percentage'])
                    # Clean NaN values
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                return records
            return []

        try:
            return retry_with_backoff(fetch_asset, max_retries=2, initial_delay=30.0)
        except Exception as e:
            print(f"Error fetching asset holdings for {symbol}: {e}")
            return []


# Singleton instance
vnstock_service = VnstockService()
