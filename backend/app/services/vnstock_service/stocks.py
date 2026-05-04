from __future__ import annotations

from datetime import date
from typing import Any, Awaitable, Callable, List, Dict, Optional
import asyncio
import math
import pandas as pd
import time

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
from .models import IndexContribution, IndexContributionRow, IndexContributionTotals, IndexValue, StockInfo
from .symbols import get_group_code_for_index
from .stock_metadata import StockMetadataService
from .history import HistoryService


class StocksService:
    """Stocks and listings related operations."""

    # Industry cache: symbol -> ICB level 2 industry name
    _industry_cache: Dict[str, str] = {}
    _industry_cache_timestamp: float = 0
    _industry_cache_ttl: float = 6 * 3600  # 6 hours in seconds

    def __init__(self, metadata: StockMetadataService, history: HistoryService):
        self._metadata = metadata
        self._history = history
        self._industry_list_cache: List[Dict[str, str]] = []
        self._industry_list_cache_timestamp: float = 0.0
        self._industry_list_failure_timestamp: float = 0.0
        self._industry_list_failure_ttl: float = 5 * 60
        self._industry_mapping_failure_timestamp: float = 0.0
        self._industry_mapping_failure_ttl: float = 5 * 60

    async def get_index_contribution(
        self,
        index_symbol: str,
        index_value: IndexValue | None,
        company_overview_fetcher: Callable[[str], Awaitable[List[Dict[str, Any]]]],
    ) -> IndexContribution:
        """
        Estimate current-session index contribution using HOSE-style adjusted caps.

        HOSE index calculation uses price * outstanding shares * rounded free-float
        * capping factor, divided by a divisor. The official divisor and capping
        factors are not exposed by the current data sources, so this computes
        effective weights from available company overview data and reconciles
        point contribution to the official index move when available.
        """
        stocks = await self.get_index_stocks(index_symbol, limit=1000)
        overview_by_ticker = await self._fetch_contribution_overviews(stocks, company_overview_fetcher)
        rows, excluded_count = self.build_index_contribution_rows(
            stocks=stocks,
            overview_by_ticker=overview_by_ticker,
            index_change_value=index_value.change_value if index_value else None,
            apply_single_stock_cap=self._should_apply_single_stock_cap(index_symbol),
        )

        totals = self._build_index_contribution_totals(rows, excluded_count)
        return IndexContribution(
            symbol=index_symbol.upper(),
            name=index_value.name if index_value else index_symbol.upper(),
            value=index_value.value if index_value else None,
            change=index_value.change if index_value else None,
            change_value=index_value.change_value if index_value else None,
            rows=rows,
            totals=totals,
        )

    async def _fetch_contribution_overviews(
        self,
        stocks: List[StockInfo],
        company_overview_fetcher: Callable[[str], Awaitable[List[Dict[str, Any]]]],
    ) -> Dict[str, Dict[str, Any]]:
        async def fetch_one(stock: StockInfo) -> tuple[str, Dict[str, Any]]:
            ticker = stock.ticker.upper()
            try:
                records = await company_overview_fetcher(ticker)
            except Exception as e:
                logger.warning(f"Could not fetch company overview for {ticker}: {e}")
                return ticker, {}
            first_record = records[0] if records else {}
            return ticker, first_record if isinstance(first_record, dict) else {}

        pairs = await asyncio.gather(*(fetch_one(stock) for stock in stocks))
        return dict(pairs)

    @classmethod
    def build_index_contribution_rows(
        cls,
        stocks: List[StockInfo],
        overview_by_ticker: Dict[str, Dict[str, Any]],
        index_change_value: float | None,
        apply_single_stock_cap: bool = True,
    ) -> tuple[List[IndexContributionRow], int]:
        candidates: List[Dict[str, Any]] = []
        excluded_count = 0

        for stock in stocks:
            ticker = stock.ticker.upper()
            session_return = cls._safe_float(stock.price_change_24h)
            price = cls._safe_float(stock.price)
            if session_return is None or price is None or price <= 0 or session_return <= -100:
                excluded_count += 1
                continue

            prior_price = price / (1 + (session_return / 100))
            if not math.isfinite(prior_price) or prior_price <= 0:
                excluded_count += 1
                continue

            overview = overview_by_ticker.get(ticker, {})
            outstanding_shares, missing_outstanding, used_market_cap_fallback = cls._resolve_outstanding_shares(
                stock,
                overview,
            )
            if outstanding_shares is None or outstanding_shares <= 0:
                excluded_count += 1
                continue

            free_float_ratio, missing_free_float = cls._resolve_free_float_ratio(overview)
            rounded_free_float = cls._round_hose_free_float(free_float_ratio)
            adjusted_cap = prior_price * outstanding_shares * rounded_free_float
            if not math.isfinite(adjusted_cap) or adjusted_cap <= 0:
                excluded_count += 1
                continue

            candidates.append({
                "stock": stock,
                "ticker": ticker,
                "price": price,
                "prior_price": prior_price,
                "session_return": session_return,
                "outstanding_shares": outstanding_shares,
                "free_float_ratio": rounded_free_float,
                "missing_outstanding": missing_outstanding,
                "missing_free_float": missing_free_float,
                "used_market_cap_fallback": used_market_cap_fallback,
                "adjusted_cap": adjusted_cap,
            })

        if not candidates:
            return [], excluded_count

        caps = [0.10 if apply_single_stock_cap else 1.0 for _ in candidates]
        weights = cls._apply_weight_caps([item["adjusted_cap"] for item in candidates], caps)
        raw_total = sum(item["adjusted_cap"] for item in candidates)

        provisional_rows: List[IndexContributionRow] = []
        for item, effective_weight in zip(candidates, weights):
            raw_weight = item["adjusted_cap"] / raw_total if raw_total > 0 else 0
            capping_factor = effective_weight / raw_weight if raw_weight > 0 else 1
            percent_contribution = effective_weight * item["session_return"]
            stock = item["stock"]

            provisional_rows.append(IndexContributionRow(
                ticker=item["ticker"],
                company_name=stock.company_name or item["ticker"],
                price=round(item["price"], 4),
                prior_price=round(item["prior_price"], 4),
                session_return=round(item["session_return"], 4),
                outstanding_shares=round(item["outstanding_shares"], 4),
                free_float_ratio=round(item["free_float_ratio"], 6),
                capping_factor=round(capping_factor, 6),
                effective_weight=round(effective_weight, 8),
                percent_contribution=percent_contribution,
                point_contribution=None,
                missing_outstanding_shares=item["missing_outstanding"],
                missing_free_float=item["missing_free_float"],
                used_market_cap_shares_fallback=item["used_market_cap_fallback"],
            ))

        net_percent = sum(row.percent_contribution for row in provisional_rows)
        rows: List[IndexContributionRow] = []
        for row in provisional_rows:
            point_contribution = None
            can_scale_to_points = (
                index_change_value is not None
                and abs(net_percent) > 1e-9
                and (
                    abs(index_change_value) <= 1e-9
                    or (net_percent > 0 and index_change_value > 0)
                    or (net_percent < 0 and index_change_value < 0)
                )
            )
            if can_scale_to_points:
                point_contribution = (row.percent_contribution / net_percent) * index_change_value
            rows.append(IndexContributionRow(
                **{
                    **row.__dict__,
                    "percent_contribution": round(row.percent_contribution, 6),
                    "point_contribution": round(point_contribution, 6) if point_contribution is not None else None,
                }
            ))

        rows.sort(key=lambda row: (abs(row.point_contribution if row.point_contribution is not None else row.percent_contribution), row.ticker), reverse=True)
        return rows, excluded_count

    @staticmethod
    def _build_index_contribution_totals(
        rows: List[IndexContributionRow],
        excluded_count: int,
    ) -> IndexContributionTotals:
        positive_percent = sum(row.percent_contribution for row in rows if row.percent_contribution > 0)
        negative_percent = sum(row.percent_contribution for row in rows if row.percent_contribution < 0)
        point_values = [row.point_contribution for row in rows if row.point_contribution is not None]
        positive_points = sum(value for value in point_values if value > 0) if point_values else None
        negative_points = sum(value for value in point_values if value < 0) if point_values else None
        return IndexContributionTotals(
            positive_percent=round(positive_percent, 6),
            negative_percent=round(negative_percent, 6),
            net_percent=round(positive_percent + negative_percent, 6),
            positive_points=round(positive_points, 6) if positive_points is not None else None,
            negative_points=round(negative_points, 6) if negative_points is not None else None,
            net_points=round((positive_points + negative_points), 6) if positive_points is not None and negative_points is not None else None,
            excluded_count=excluded_count,
            missing_outstanding_shares_count=sum(1 for row in rows if row.missing_outstanding_shares),
            missing_free_float_count=sum(1 for row in rows if row.missing_free_float),
        )

    @staticmethod
    def _apply_weight_caps(values: List[float], caps: List[float]) -> List[float]:
        total = sum(values)
        if total <= 0:
            return [0 for _ in values]

        weights = [0.0 for _ in values]
        remaining_indices = set(range(len(values)))
        remaining_weight = 1.0

        while remaining_indices:
            remaining_value = sum(values[index] for index in remaining_indices)
            if remaining_value <= 0:
                break

            capped_this_round = []
            for index in remaining_indices:
                provisional_weight = remaining_weight * (values[index] / remaining_value)
                if provisional_weight > caps[index]:
                    weights[index] = caps[index]
                    capped_this_round.append(index)

            if not capped_this_round:
                for index in remaining_indices:
                    weights[index] = remaining_weight * (values[index] / remaining_value)
                break

            for index in capped_this_round:
                remaining_indices.remove(index)
            remaining_weight = max(0.0, 1.0 - sum(weights))

        return weights

    @staticmethod
    def _round_hose_free_float(value: float) -> float:
        bounded = min(max(value, 0.01), 1.0)
        if bounded <= 0.15:
            return math.ceil(bounded * 100) / 100
        return math.ceil(bounded * 20) / 20

    @staticmethod
    def _resolve_free_float_ratio(overview: Dict[str, Any]) -> tuple[float, bool]:
        for key in ("free_float_ratio", "free_float_percentage", "free_float_percent", "free_float"):
            value = StocksService._safe_float(overview.get(key))
            ratio = StocksService._normalize_ratio(value)
            if ratio is not None and ratio > 0:
                return min(ratio, 1.0), False
        return 1.0, True

    @staticmethod
    def _resolve_outstanding_shares(
        stock: StockInfo,
        overview: Dict[str, Any],
    ) -> tuple[float | None, bool, bool]:
        for key in ("outstanding_shares", "listed_volume", "listed_shares", "total_shares"):
            value = StocksService._safe_float(overview.get(key))
            if value is not None and value > 0:
                return value, False, False

        market_cap = StocksService._safe_float(stock.market_cap)
        price = StocksService._safe_float(stock.price)
        if market_cap is not None and market_cap > 0 and price is not None and price > 0:
            return (market_cap * 1_000_000_000) / price, True, True

        return None, True, False

    @staticmethod
    def _normalize_ratio(value: float | None) -> float | None:
        if value is None or not math.isfinite(value) or value <= 0:
            return None
        if value <= 1:
            return value
        if value <= 100:
            return value / 100
        if value <= 10_000:
            return value / 10_000
        return None

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    @staticmethod
    def _should_apply_single_stock_cap(index_symbol: str) -> bool:
        return index_symbol.upper() in {"VN30", "VN100", "VNMIDCAP", "VNMID", "VNSMALLCAP", "VNSML", "VNALLSHARE", "VNALL"}

    async def get_index_stocks(
        self,
        index_symbol: str,
        limit: int = 100,
        range_start: date | None = None,
        range_end: date | None = None,
    ) -> List[StockInfo]:
        """
        Fetch stocks for a specific index with price and market cap data.
        """
        loop = asyncio.get_event_loop()
        stocks = await loop.run_in_executor(frontend_executor, self._fetch_index_data, index_symbol, limit)

        # Launch background task for enrichment
        asyncio.create_task(self._metadata.enrich_stocks_with_metadata(stocks))

        # Apply current cache to the response immediately
        stocks = await self._metadata.apply_cache_to_stocks(stocks)
        return await self._history.enrich_with_price_extremes(stocks, range_start, range_end)

    async def get_industry_list(self) -> List[Dict[str, str]]:
        """
        Fetch all ICB level 2 industries.
        """
        current_time = time.time()
        if (
            self._industry_list_failure_timestamp
            and (current_time - self._industry_list_failure_timestamp) < self._industry_list_failure_ttl
        ):
            return list(self._industry_list_cache)

        loop = asyncio.get_event_loop()
        try:
            df = await loop.run_in_executor(frontend_executor, self._fetch_industries_sync)
        except Exception as e:
            self._industry_list_failure_timestamp = current_time
            logger.warning(f"Error fetching industry list; using cached fallback if available: {e}")
            return list(self._industry_list_cache)

        if df is not None and not df.empty:
            industries = self._build_industry_list_with_families(df)
            self._industry_list_cache = industries
            self._industry_list_cache_timestamp = current_time
            self._industry_list_failure_timestamp = 0.0
            return list(industries)
        return []

    def _build_industry_list_with_families(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        required_columns = {'level', 'icb_name', 'en_icb_name', 'icb_code'}
        if not required_columns.issubset(df.columns):
            return []

        normalized_df = df[['level', 'icb_name', 'en_icb_name', 'icb_code']].copy()
        normalized_df['icb_name'] = normalized_df['icb_name'].fillna('').astype(str).str.strip()
        normalized_df['en_icb_name'] = normalized_df['en_icb_name'].fillna('').astype(str).str.strip()
        normalized_df['icb_code'] = normalized_df['icb_code'].fillna('').astype(str).str.strip()
        normalized_df['level'] = pd.to_numeric(normalized_df['level'], errors='coerce')

        level1_df = normalized_df[normalized_df['level'] == 1]
        level2_df = normalized_df[normalized_df['level'] == 2]
        if level2_df.empty:
            return []

        def normalize_label(value: str) -> str:
            return value.casefold().strip()

        level1_by_name = {
            normalize_label(row.icb_name): {
                'code': row.icb_code,
                'name': row.icb_name,
                'en_name': row.en_icb_name,
            }
            for row in level1_df.itertuples(index=False)
            if row.icb_name and row.icb_code
        }
        level1_by_code = {
            row.icb_code: {
                'code': row.icb_code,
                'name': row.icb_name,
                'en_name': row.en_icb_name,
            }
            for row in level1_df.itertuples(index=False)
            if row.icb_code
        }

        def infer_level1(level2_code: str, level2_name: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
            direct_match = level1_by_name.get(normalize_label(level2_name))
            if direct_match:
                return direct_match['code'], direct_match['name'], direct_match['en_name']

            if level2_code and level2_code[0].isdigit():
                family_code = f"{level2_code[0]}000"
                family = level1_by_code.get(family_code)
                if family:
                    return family['code'], family['name'], family['en_name']
            return None, None, None

        industries: List[Dict[str, str]] = []
        for row in level2_df.itertuples(index=False):
            family_code, family_name, family_en_name = infer_level1(row.icb_code, row.icb_name)
            industries.append({
                'icb_name': row.icb_name,
                'en_icb_name': row.en_icb_name,
                'icb_code': row.icb_code,
                'icb_family_code': family_code or '',
                'icb_family_name': family_name or '',
                'icb_family_en_name': family_en_name or '',
            })
        return industries

    @staticmethod
    def _build_kbs_industries_frame(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or not {"industry_code", "industry_name"}.issubset(df.columns):
            return pd.DataFrame(columns=["icb_name", "en_icb_name", "icb_code", "level"])

        normalized = df[["industry_code", "industry_name"]].copy()
        normalized["industry_code"] = normalized["industry_code"].fillna("").astype(str).str.strip()
        normalized["industry_name"] = normalized["industry_name"].fillna("").astype(str).str.strip()
        normalized = normalized[(normalized["industry_code"] != "") & (normalized["industry_name"] != "")]
        if normalized.empty:
            return pd.DataFrame(columns=["icb_name", "en_icb_name", "icb_code", "level"])

        normalized = (
            normalized.drop_duplicates(subset=["industry_code", "industry_name"])
            .sort_values(by=["industry_name", "industry_code"])
            .reset_index(drop=True)
        )
        return pd.DataFrame({
            "icb_name": normalized["industry_name"],
            "en_icb_name": normalized["industry_name"],
            "icb_code": "KBS-" + normalized["industry_code"],
            "level": 2,
        })

    @staticmethod
    def _build_symbol_industry_mapping(df: pd.DataFrame) -> Dict[str, str]:
        if df is None or df.empty or "symbol" not in df.columns:
            return {}

        industry_column = None
        for column in ("icb_name2", "industry_name"):
            if column in df.columns:
                industry_column = column
                break
        if industry_column is None:
            return {}

        industry_map: Dict[str, str] = {}
        for _, row in df.iterrows():
            symbol = row.get("symbol", "")
            industry = row.get(industry_column, "")
            if symbol and industry:
                industry_map[str(symbol).upper()] = str(industry)
        return industry_map

    @staticmethod
    def _symbols_for_industry(df: pd.DataFrame, industry_name: str) -> List[str]:
        if df is None or df.empty or "symbol" not in df.columns:
            return []

        cols_to_check = ["icb_name2", "icb_name3", "icb_name4", "industry_name"]
        mask = pd.Series([False] * len(df))
        for col in cols_to_check:
            if col in df.columns:
                mask |= (df[col] == industry_name)
        return df[mask]["symbol"].dropna().astype(str).tolist()

    async def get_industry_stocks(
        self,
        industry_name: str,
        limit: int = 100,
        range_start: date | None = None,
        range_end: date | None = None,
    ) -> List[StockInfo]:
        """
        Fetch stocks for a specific ICB industry.
        """
        loop = asyncio.get_event_loop()
        stocks = await loop.run_in_executor(frontend_executor, self._fetch_industry_data, industry_name, limit)

        # Launch background task for enrichment
        asyncio.create_task(self._metadata.enrich_stocks_with_metadata(stocks))

        # Apply current cache to the response immediately
        stocks = await self._metadata.apply_cache_to_stocks(stocks)
        return await self._history.enrich_with_price_extremes(stocks, range_start, range_end)

    async def get_symbol_stocks(
        self,
        symbols: List[str],
        limit: int = 100,
        range_start: date | None = None,
        range_end: date | None = None,
    ) -> List[StockInfo]:
        """
        Fetch stocks for a list of symbols.
        """
        cleaned_symbols = [symbol.strip().upper() for symbol in symbols if symbol and symbol.strip()]
        if not cleaned_symbols:
            return []
        unique_symbols = list(dict.fromkeys(cleaned_symbols))

        loop = asyncio.get_event_loop()
        stocks = await loop.run_in_executor(frontend_executor, self._fetch_symbols_data, unique_symbols, limit)

        # Launch background task for enrichment
        asyncio.create_task(self._metadata.enrich_stocks_with_metadata(stocks))

        # Apply current cache to the response immediately
        stocks = await self._metadata.apply_cache_to_stocks(stocks)
        return await self._history.enrich_with_price_extremes(stocks, range_start, range_end)

    def _fetch_industries_sync(self) -> pd.DataFrame:
        """Fetch industries synchronously."""
        from vnstock import Listing

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError("Circuit breaker open - cannot fetch industries")

        try:
            result = Listing(source='KBS').symbols_by_industries()
            api_circuit_breaker.record_success()
            return self._build_kbs_industries_frame(result)
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching KBS industries: {e}")
            raise

    def _get_or_fetch_industry_mapping(self) -> Dict[str, str]:
        """
        Get cached industry mapping or fetch fresh data if cache is stale.
        Returns a dict mapping symbol -> ICB level 2 industry name.
        Uses 6-hour TTL similar to fund cache.
        """
        current_time = time.time()
        cache_age = current_time - self._industry_cache_timestamp

        # Return cached data if fresh
        if self._industry_cache and cache_age < self._industry_cache_ttl:
            logger.debug(f"Using cached industry mapping ({len(self._industry_cache)} symbols, age: {cache_age:.0f}s)")
            return self._industry_cache

        if (
            self._industry_mapping_failure_timestamp
            and (current_time - self._industry_mapping_failure_timestamp) < self._industry_mapping_failure_ttl
        ):
            return self._industry_cache if self._industry_cache else {}

        # Fetch fresh industry mapping
        from vnstock import Listing

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            logger.warning("Circuit breaker open - using stale industry cache if available")
            return self._industry_cache if self._industry_cache else {}

        try:
            logger.info("Fetching fresh industry mapping from KBS API")
            listing = Listing(source='KBS')
            df = listing.symbols_by_industries()
            api_circuit_breaker.record_success()

            if df is not None and not df.empty:
                industry_map = self._build_symbol_industry_mapping(df)
                self._industry_cache = industry_map
                self._industry_cache_timestamp = current_time
                self._industry_mapping_failure_timestamp = 0.0
                logger.info(f"KBS industry mapping cached: {len(industry_map)} symbols")
                return industry_map

            self._industry_mapping_failure_timestamp = current_time
            logger.warning("Empty KBS industry mapping returned from API")
            return self._industry_cache if self._industry_cache else {}
        except (SystemExit, Exception) as e:
            self._industry_mapping_failure_timestamp = current_time
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=60.0)
                logger.warning("Rate limited while fetching KBS industry mapping - using stale cache")
            else:
                logger.warning(f"Error fetching KBS industry mapping: {e}")
            return self._industry_cache if self._industry_cache else {}

    def _fetch_index_data(self, index_name: str, limit: int) -> List[StockInfo]:
        """
        Synchronous method to fetch index data (VN100, VN30, etc.) using vnstock.
        Called in thread pool executor to avoid blocking.
        """
        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch index {index_name}")

        try:
            # Map index name to group code
            group_code = get_group_code_for_index(index_name)

            # Get stock symbols for the specified group
            symbols_df = self._fetch_index_symbols_with_fallback(index_name, group_code)

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

    def _fetch_index_symbols_with_fallback(self, index_name: str, group_code: str):
        """Fetch index members from KBS listing membership."""
        from vnstock import Listing

        kbs_group_code = get_group_code_for_index(index_name, source="KBS")
        try:
            symbols_df = Listing(source='KBS').symbols_by_group(kbs_group_code)
            api_circuit_breaker.record_success()
            return symbols_df
        except ValueError as e:
            logger.warning(
                f"Group '{kbs_group_code}' (mapped from '{index_name}') not supported by KBS symbols_by_group: {e}"
            )
            return None
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching KBS symbols for {kbs_group_code}: {e}")
            logger.warning(f"Error fetching KBS symbols for group '{kbs_group_code}': {e}")
            return None

    def _fetch_industry_data(self, industry_name: str, limit: int) -> List[StockInfo]:
        """
        Synchronous method to fetch industry data using vnstock.
        """
        from vnstock import Listing

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch industry {industry_name}")

        try:
            listing = Listing(source='KBS')
            df = listing.symbols_by_industries()
            api_circuit_breaker.record_success()
            symbols = self._symbols_for_industry(df, industry_name)
            return self._fetch_symbols_data(symbols, limit)
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching KBS industry {industry_name}: {e}")
            logger.warning(f"Error fetching KBS industry {industry_name} data: {e}")
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
            # Get industry mapping (cached for 6 hours)
            industry_mapping = self._get_or_fetch_industry_mapping()

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

                            # Get foreign buy/sell values (in billion VND)
                            # API returns values in VND, divide by 1e9 to get Billion VND
                            foreign_buy_value = None
                            if 'match_foreign_buy_value' in row.index:
                                try:
                                    foreign_buy = row['match_foreign_buy_value']
                                    if pd.notna(foreign_buy):
                                        foreign_buy_value = float(foreign_buy) / 1e9
                                except (ValueError, TypeError):
                                    pass

                            foreign_sell_value = None
                            if 'match_foreign_sell_value' in row.index:
                                try:
                                    foreign_sell = row['match_foreign_sell_value']
                                    if pd.notna(foreign_sell):
                                        foreign_sell_value = float(foreign_sell) / 1e9
                                except (ValueError, TypeError):
                                    pass

                            # Get foreign room values (raw shares)
                            current_room = None
                            if 'match_current_room' in row.index:
                                try:
                                    current_room_value = row['match_current_room']
                                    if pd.notna(current_room_value):
                                        current_room = int(float(current_room_value))
                                except (ValueError, TypeError):
                                    pass

                            total_room = None
                            if 'match_total_room' in row.index:
                                try:
                                    total_room_value = row['match_total_room']
                                    if pd.notna(total_room_value):
                                        total_room = int(float(total_room_value))
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

                                # Get industry from cached mapping
                                industry = industry_mapping.get(str(ticker).upper(), '')

                                stocks_data.append(StockInfo(
                                    ticker=str(ticker),
                                    price=price,
                                    company_name=company_name,
                                    exchange=exchange,
                                    market_cap=round(market_cap, 2),
                                    charter_capital=round(charter_capital, 2),
                                    pe_ratio=round(pe_ratio, 2) if pe_ratio is not None else None,
                                    accumulated_value=round(accumulated_value, 2) if accumulated_value is not None else None,
                                    foreign_buy_value=round(foreign_buy_value, 2) if foreign_buy_value is not None else None,
                                    foreign_sell_value=round(foreign_sell_value, 2) if foreign_sell_value is not None else None,
                                    current_room=current_room,
                                    total_room=total_room,
                                    price_change_24h=round(price_change_24h, 2) if price_change_24h is not None else None,
                                    industry=industry
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
