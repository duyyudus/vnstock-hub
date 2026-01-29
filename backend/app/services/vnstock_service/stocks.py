from __future__ import annotations

from typing import List, Dict
import asyncio
import pandas as pd

from .core import (
    frontend_executor,
    logger,
    bg_logger,
    api_circuit_breaker,
    CircuitOpenError,
    _record_rate_limit,
    _is_rate_limit_error,
    _flatten_columns,
)
from .models import StockInfo
from .symbols import get_group_code_for_index
from .stock_metadata import StockMetadataService
from .history import HistoryService


class StocksService:
    """Stocks and listings related operations."""

    def __init__(self, metadata: StockMetadataService, history: HistoryService):
        self._metadata = metadata
        self._history = history

    async def get_index_stocks(self, index_symbol: str, limit: int = 100) -> List[StockInfo]:
        """
        Fetch stocks for a specific index with price and market cap data.
        """
        loop = asyncio.get_event_loop()
        stocks = await loop.run_in_executor(frontend_executor, self._fetch_index_data, index_symbol, limit)

        # Launch background task for enrichment
        asyncio.create_task(self._metadata.enrich_stocks_with_metadata(stocks))

        # Apply current cache to the response immediately
        return await self._metadata.apply_cache_to_stocks(stocks)

    async def get_industry_list(self) -> List[Dict[str, str]]:
        """
        Fetch all ICB level 2 industries.
        """
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(frontend_executor, self._fetch_industries_sync)
        if df is not None and not df.empty:
            # Filter for level 2 industries
            l2_df = df[df['level'] == 2]
            return l2_df[['icb_name', 'en_icb_name', 'icb_code']].to_dict('records')
        return []

    async def get_industry_stocks(self, industry_name: str, limit: int = 100) -> List[StockInfo]:
        """
        Fetch stocks for a specific ICB industry.
        """
        loop = asyncio.get_event_loop()
        stocks = await loop.run_in_executor(frontend_executor, self._fetch_industry_data, industry_name, limit)

        # Launch background task for enrichment
        asyncio.create_task(self._metadata.enrich_stocks_with_metadata(stocks))

        # Apply current cache to the response immediately
        return await self._metadata.apply_cache_to_stocks(stocks)

    def _fetch_industries_sync(self) -> pd.DataFrame:
        """Fetch industries synchronously."""
        from vnstock import Listing

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError("Circuit breaker open - cannot fetch industries")

        try:
            result = Listing(source='VCI').industries_icb()
            api_circuit_breaker.record_success()
            return result
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching industries: {e}")
            raise

    def _fetch_index_data(self, index_name: str, limit: int) -> List[StockInfo]:
        """
        Synchronous method to fetch index data (VN100, VN30, etc.) using vnstock.
        Called in thread pool executor to avoid blocking.
        """
        from vnstock import Listing

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch index {index_name}")

        try:
            # Map index name to group code
            group_code = get_group_code_for_index(index_name)

            # Get stock symbols for the specified group
            listing = Listing(source='VCI')
            try:
                symbols_df = listing.symbols_by_group(group_code)
                api_circuit_breaker.record_success()
            except ValueError as e:
                logger.warning(f"Group '{group_code}' (mapped from '{index_name}') not supported by symbols_by_group: {e}")
                return []
            except (SystemExit, Exception) as e:
                if _is_rate_limit_error(e):
                    _record_rate_limit(reset_seconds=30.0)
                    raise CircuitOpenError(f"Rate limited fetching symbols for {group_code}: {e}")
                logger.warning(f"Error fetching symbols for group '{group_code}': {e}")
                return []

            if symbols_df is None or symbols_df.empty:
                logger.warning(f"Could not fetch symbols for {group_code} group")
                return []
            else:
                # The series returned by symbols_by_group contains the symbols
                symbols = symbols_df.tolist()

            return self._fetch_symbols_data(symbols, limit)

        except CircuitOpenError:
            raise  # Re-raise circuit breaker errors
        except Exception as e:
            logger.warning(f"Error fetching {index_name} data: {e}")
            return []

    def _fetch_industry_data(self, industry_name: str, limit: int) -> List[StockInfo]:
        """
        Synchronous method to fetch industry data using vnstock.
        """
        from vnstock import Listing

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch industry {industry_name}")

        try:
            listing = Listing(source='VCI')
            # Get all symbols with industry info
            df = listing.symbols_by_industries()
            api_circuit_breaker.record_success()

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
        except CircuitOpenError:
            raise  # Re-raise circuit breaker errors
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching industry {industry_name}: {e}")
            logger.warning(f"Error fetching industry {industry_name} data: {e}")
            return []

    def _fetch_symbols_data(self, symbols: List[str], limit: int) -> List[StockInfo]:
        """
        Generic method to fetch price and market cap data for a list of symbols.
        """
        from vnstock import Trading

        # Check circuit breaker before making API calls
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError("Circuit breaker open - API rate limited")

        try:
            # Get price board for stocks in batches
            trading = Trading(source='VCI')
            stocks_data = []
            batch_size = 50  # Smaller batch for more reliable fetching

            # Process symbols
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                try:
                    price_board = trading.price_board(batch)
                    api_circuit_breaker.record_success()

                    if price_board is not None and not price_board.empty:
                        # Flatten multi-level column names
                        price_board = _flatten_columns(price_board)

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
                                # Map exchange codes to full names if needed
                                exchange = row.get('listing_exchange', '')
                                if exchange == 'HSX':
                                    exchange = 'HOSE'

                                # Get company name if available
                                company_name = row.get('listing_organ_name', '')

                                stocks_data.append(StockInfo(
                                    ticker=str(ticker),
                                    price=price,
                                    company_name=company_name,
                                    exchange=exchange,
                                    market_cap=round(market_cap, 2),
                                    charter_capital=round(charter_capital, 2),
                                    pe_ratio=round(pe_ratio, 2) if pe_ratio is not None else None,
                                    accumulated_value=round(accumulated_value, 2) if accumulated_value is not None else None,
                                    price_change_24h=round(price_change_24h, 2) if price_change_24h is not None else None
                                ))

                except (SystemExit, Exception) as e:
                    if _is_rate_limit_error(e):
                        _record_rate_limit(reset_seconds=60.0)
                        raise CircuitOpenError(f"Rate limited while fetching price board batch {i}")
                    logger.warning(f"Error fetching batch {i}: {e}")
                    continue

            # Sort by market cap descending and take the requested limit
            stocks_data.sort(key=lambda x: x.market_cap, reverse=True)
            top_stocks = stocks_data[:limit]

            # Fetch historical price changes for top stocks
            top_stocks = self._history.enrich_with_price_changes_sync(top_stocks)

            return top_stocks

        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            # Check if this is a rate limit error and record circuit breaker failure
            error_name = type(e).__name__
            if error_name in {"RateLimitExceeded", "RateLimitError"} or "rate limit" in str(e).lower():
                _record_rate_limit(reset_seconds=60.0)
                raise CircuitOpenError(f"Rate limited while fetching symbols data: {e}")
            logger.warning(f"Error fetching symbols data: {e}")
            import traceback
            bg_logger.error(f"Stack trace for symbols data error:\n{traceback.format_exc()}")
            return []
