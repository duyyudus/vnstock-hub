from __future__ import annotations

from typing import List, Dict, Any, Callable, Awaitable, Set
import asyncio
import threading
from datetime import datetime, date, timedelta
import pandas as pd

from sqlalchemy import select, and_, func
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.database import async_session
from app.db.models import StockDailyHistory, StockCompany, StockHistorySyncState
from app.services.sync_status import sync_status

from .core import (
    frontend_executor,
    logger,
    bg_logger,
    api_circuit_breaker,
    CircuitOpenError,
    retry_with_backoff,
    _record_rate_limit,
    _is_rate_limit_error,
    get_sync_engine,
)
from .models import StockInfo


# Weekly price history defaults
REQUEST_HISTORY_SYNC_TIMEOUT_SECONDS = 12.0

STANDARD_FOREIGN_HISTORY_ALIASES = {
    "time": "time",
    "date": "time",
    "tradingdate": "time",
    "ngay": "time",
    "frbuyvolume": "fr_buy_volume",
    "foreignbuyvolume": "fr_buy_volume",
    "klmua": "fr_buy_volume",
    "frbuyvalue": "fr_buy_value",
    "foreignbuyvalue": "fr_buy_value",
    "gtmua": "fr_buy_value",
    "frsellvolume": "fr_sell_volume",
    "foreignsellvolume": "fr_sell_volume",
    "klban": "fr_sell_volume",
    "frsellvalue": "fr_sell_value",
    "foreignsellvalue": "fr_sell_value",
    "gtban": "fr_sell_value",
    "frnetvolume": "fr_net_volume",
    "foreignnetvolume": "fr_net_volume",
    "klgdrong": "fr_net_volume",
    "frnetvalue": "fr_net_value",
    "foreignnetvalue": "fr_net_value",
    "gtdgrong": "fr_net_value",
    "frbuyvolumetotal": "fr_buy_volume_total",
    "frbuyvaluetotal": "fr_buy_value_total",
    "frsellvolumetotal": "fr_sell_volume_total",
    "frsellvaluetotal": "fr_sell_value_total",
    "frnetvolumetotal": "fr_net_volume_total",
    "frnetvaluetotal": "fr_net_value_total",
}

STANDARD_PROP_HISTORY_ALIASES = {
    "time": "time",
    "date": "time",
    "tradingdate": "time",
    "ngay": "time",
    "propbuyvolume": "prop_buy_volume",
    "proprietarybuyvolume": "prop_buy_volume",
    "klcpmua": "prop_buy_volume",
    "propbuyvalue": "prop_buy_value",
    "proprietarybuyvalue": "prop_buy_value",
    "gtmua": "prop_buy_value",
    "propsellvolume": "prop_sell_volume",
    "proprietarysellvolume": "prop_sell_volume",
    "klcpban": "prop_sell_volume",
    "propsellvalue": "prop_sell_value",
    "proprietarysellvalue": "prop_sell_value",
    "gtban": "prop_sell_value",
    "totalbuytradevolume": "total_buy_trade_volume",
    "totalbuytradevalue": "total_buy_trade_value",
    "totalselltradevolume": "total_sell_trade_volume",
    "totalselltradevalue": "total_sell_trade_value",
    "totalmatchtradebuyvolume": "total_match_trade_buy_volume",
    "totalmatchtradebuyvalue": "total_match_trade_buy_value",
    "totaldealtradebuyvolume": "total_deal_trade_buy_volume",
    "totaldealtradebuyvalue": "total_deal_trade_buy_value",
    "totalmatchtradesellvolume": "total_match_trade_sell_volume",
    "totalmatchtradesellvalue": "total_match_trade_sell_value",
    "totaldealtradesellvolume": "total_deal_trade_sell_volume",
    "totaldealtradesellvalue": "total_deal_trade_sell_value",
}

STANDARD_TURNOVER_HISTORY_ALIASES = {
    "time": "time",
    "date": "time",
    "tradingdate": "time",
    "matchedvolume": "matched_volume",
    "matchedvalue": "matched_value",
    "dealvolume": "deal_volume",
    "dealvalue": "deal_value",
    "totalvolume": "total_volume",
    "totalvalue": "total_value",
    "totalmatchvolume": "matched_volume",
    "totalmatchvalue": "matched_value",
    "totaldealvolume": "deal_volume",
    "totaldealvalue": "deal_value",
}


class HistoryService:
    """Historical price and volume data operations."""

    def __init__(self) -> None:
        self._weekly_prices_retry_cooldown = timedelta(minutes=10)
        self._symbol_sync_locks: Dict[str, threading.Lock] = {}
        self._symbol_sync_locks_guard = threading.Lock()
        self._request_sync_handler: Callable[[str, date, date, float], Awaitable[Dict[str, Any]]] | None = None
        self._weekly_sync_trigger_handler: Callable[[List[str]], Awaitable[Dict[str, Any]]] | None = None

    def set_on_demand_history_sync_handler(
        self,
        handler: Callable[[str, date, date, float], Awaitable[Dict[str, Any]]]
    ) -> None:
        self._request_sync_handler = handler

    def set_weekly_history_sync_trigger_handler(
        self,
        handler: Callable[[List[str]], Awaitable[Dict[str, Any]]]
    ) -> None:
        self._weekly_sync_trigger_handler = handler

    @staticmethod
    def _round_optional_value(value: float | None) -> float | None:
        if value is None:
            return None
        return round(value, 2)

    @classmethod
    def _vnd_to_billion_vnd(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return cls._round_optional_value(value / 1e9)

    @classmethod
    def _derive_net_value(
        cls,
        buy_value: float | None,
        sell_value: float | None,
    ) -> float | None:
        if buy_value is None or sell_value is None:
            return None
        return cls._vnd_to_billion_vnd(buy_value - sell_value)

    def _get_symbol_sync_lock(self, symbol: str) -> threading.Lock:
        symbol_key = symbol[:3].upper()
        with self._symbol_sync_locks_guard:
            lock = self._symbol_sync_locks.get(symbol_key)
            if lock is None:
                lock = threading.Lock()
                self._symbol_sync_locks[symbol_key] = lock
            return lock

    @staticmethod
    def _normalize_price_extremes_range(
        range_start: date | None,
        range_end: date | None,
    ) -> tuple[date, date] | None:
        if range_start is None and range_end is None:
            return None

        today = date.today()
        bounded_start = min(range_start or date(1900, 1, 1), today)
        bounded_end = min(range_end or today, today)

        if bounded_end < bounded_start:
            bounded_start, bounded_end = bounded_end, bounded_start

        return bounded_start, bounded_end

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _coerce_int(cls, value: Any) -> int | None:
        coerced = cls._coerce_float(value)
        if coerced is None:
            return None
        return int(coerced)

    @staticmethod
    def _normalize_history_column_key(column_name: Any) -> str:
        return "".join(ch for ch in str(column_name).lower() if ch.isalnum())

    def _normalize_history_frame_columns(
        self,
        hist: pd.DataFrame | None,
        aliases: Dict[str, str],
    ) -> pd.DataFrame | None:
        if hist is None or hist.empty:
            return hist

        rename_map: Dict[str, str] = {}
        existing_columns = set(hist.columns)

        for column in hist.columns:
            target = aliases.get(self._normalize_history_column_key(column))
            if target is None or column == target or target in existing_columns:
                continue
            rename_map[column] = target
            existing_columns.add(target)

        if not rename_map:
            return hist

        return hist.rename(columns=rename_map)

    def _normalize_foreign_trade_history_frame(self, hist: pd.DataFrame | None) -> pd.DataFrame | None:
        normalized = self._normalize_history_frame_columns(hist, STANDARD_FOREIGN_HISTORY_ALIASES)
        return self._ensure_foreign_trade_canonical_columns(normalized)

    def _normalize_prop_trade_history_frame(self, hist: pd.DataFrame | None) -> pd.DataFrame | None:
        normalized = self._normalize_history_frame_columns(hist, STANDARD_PROP_HISTORY_ALIASES)
        return self._ensure_prop_trade_canonical_columns(normalized)

    def _normalize_turnover_history_frame(self, hist: pd.DataFrame | None) -> pd.DataFrame | None:
        normalized = self._normalize_history_frame_columns(hist, STANDARD_TURNOVER_HISTORY_ALIASES)
        return self._ensure_turnover_history_canonical_columns(normalized)

    def _fetch_ohlcv_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        source: str = "VCI",
    ) -> pd.DataFrame:
        from vnstock import Quote

        symbol_key = symbol[:3].upper()
        quote = Quote(symbol=symbol_key, source=source)
        return quote.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1D",
        )

    @staticmethod
    def _coalesce_frame_columns(
        hist: pd.DataFrame,
        candidate_columns: tuple[str, ...],
    ) -> pd.Series | None:
        series: pd.Series | None = None
        for column in candidate_columns:
            if column not in hist.columns:
                continue
            candidate = hist[column]
            if series is None:
                series = candidate.copy()
            else:
                series = series.where(series.notna(), candidate)
        return series

    @staticmethod
    def _sum_frame_columns(
        hist: pd.DataFrame,
        component_columns: tuple[str, ...],
    ) -> pd.Series | None:
        series: pd.Series | None = None
        has_value = pd.Series(False, index=hist.index)

        for column in component_columns:
            if column not in hist.columns:
                continue
            candidate = pd.to_numeric(hist[column], errors="coerce")
            has_value = has_value | candidate.notna()
            if series is None:
                series = candidate
            else:
                series = series.add(candidate.fillna(0), fill_value=0)

        if series is None:
            return None
        return series.where(has_value)

    def _ensure_foreign_trade_canonical_columns(self, hist: pd.DataFrame | None) -> pd.DataFrame | None:
        if hist is None or hist.empty:
            return hist

        normalized = hist.copy()
        specs = {
            "fr_buy_volume": (("fr_buy_volume", "fr_buy_volume_total"), ("fr_buy_volume_matched", "fr_buy_volume_deal")),
            "fr_buy_value": (("fr_buy_value", "fr_buy_value_total"), ("fr_buy_value_matched", "fr_buy_value_deal")),
            "fr_sell_volume": (("fr_sell_volume", "fr_sell_volume_total"), ("fr_sell_volume_matched", "fr_sell_volume_deal")),
            "fr_sell_value": (("fr_sell_value", "fr_sell_value_total"), ("fr_sell_value_matched", "fr_sell_value_deal")),
            "fr_net_volume": (("fr_net_volume", "fr_net_volume_total"), ("fr_net_volume_matched", "fr_net_volume_deal")),
            "fr_net_value": (("fr_net_value", "fr_net_value_total"), ("fr_net_value_matched", "fr_net_value_deal")),
        }

        for target, (direct_candidates, component_candidates) in specs.items():
            direct = self._coalesce_frame_columns(normalized, direct_candidates)
            if direct is not None:
                normalized[target] = direct
                continue
            summed = self._sum_frame_columns(normalized, component_candidates)
            if summed is not None:
                normalized[target] = summed

        return normalized

    def _ensure_prop_trade_canonical_columns(self, hist: pd.DataFrame | None) -> pd.DataFrame | None:
        if hist is None or hist.empty:
            return hist

        normalized = hist.copy()
        specs = {
            "prop_buy_volume": (("prop_buy_volume", "total_buy_trade_volume"), ("total_match_trade_buy_volume", "total_deal_trade_buy_volume")),
            "prop_buy_value": (("prop_buy_value", "total_buy_trade_value"), ("total_match_trade_buy_value", "total_deal_trade_buy_value")),
            "prop_sell_volume": (("prop_sell_volume", "total_sell_trade_volume"), ("total_match_trade_sell_volume", "total_deal_trade_sell_volume")),
            "prop_sell_value": (("prop_sell_value", "total_sell_trade_value"), ("total_match_trade_sell_value", "total_deal_trade_sell_value")),
        }

        for target, (direct_candidates, component_candidates) in specs.items():
            direct = self._coalesce_frame_columns(normalized, direct_candidates)
            if direct is not None:
                normalized[target] = direct
                continue
            summed = self._sum_frame_columns(normalized, component_candidates)
            if summed is not None:
                normalized[target] = summed

        return normalized

    def _ensure_turnover_history_canonical_columns(self, hist: pd.DataFrame | None) -> pd.DataFrame | None:
        if hist is None or hist.empty:
            return hist

        normalized = hist.copy()
        specs = {
            "matched_volume": (("matched_volume",), ()),
            "matched_value": (("matched_value",), ()),
            "deal_volume": (("deal_volume",), ()),
            "deal_value": (("deal_value",), ()),
            "total_volume": (("total_volume",), ("matched_volume", "deal_volume")),
            "total_value": (("total_value",), ("matched_value", "deal_value")),
        }

        for target, (direct_candidates, component_candidates) in specs.items():
            direct = self._coalesce_frame_columns(normalized, direct_candidates)
            if direct is not None:
                normalized[target] = direct
                continue
            if component_candidates:
                summed = self._sum_frame_columns(normalized, component_candidates)
                if summed is not None:
                    normalized[target] = summed

        return normalized

    @staticmethod
    def _preserve_existing_on_null(insert_stmt, column_name: str):
        column = getattr(StockDailyHistory.__table__.c, column_name)
        excluded = getattr(insert_stmt.excluded, column_name)
        return func.coalesce(excluded, column)

    def _build_daily_history_payload(
        self,
        symbol: str,
        ohlcv_hist: pd.DataFrame,
        turnover_hist: pd.DataFrame | None = None,
        foreign_hist: pd.DataFrame | None = None,
        prop_hist: pd.DataFrame | None = None,
        created_at: datetime | None = None
    ) -> tuple[List[Dict[str, Any]], date | None, date | None]:
        symbol_key = symbol[:3].upper()
        row_created_at = created_at or datetime.utcnow()
        deduped_by_date: Dict[date, Dict[str, Any]] = {}
        turnover_hist = self._normalize_turnover_history_frame(turnover_hist)
        foreign_hist = self._normalize_foreign_trade_history_frame(foreign_hist)
        prop_hist = self._normalize_prop_trade_history_frame(prop_hist)

        for _, row in ohlcv_hist.iterrows():
            raw_time = row.get('time')
            if pd.isna(raw_time):
                continue
            raw_close = row.get('close')
            if pd.isna(raw_close):
                continue
            try:
                history_date = pd.to_datetime(raw_time).date()
            except Exception:
                continue

            deduped_by_date[history_date] = {
                'symbol': symbol_key,
                'date': history_date,
                'open': self._coerce_float(row.get('open')),
                'high': self._coerce_float(row.get('high')),
                'low': self._coerce_float(row.get('low')),
                'close': float(raw_close),
                'volume': self._coerce_int(row.get('volume')),
                'matched_volume': None,
                'matched_value': None,
                'deal_volume': None,
                'deal_value': None,
                'total_volume': None,
                'total_value': None,
                'foreign_buy_volume': None,
                'foreign_buy_value': None,
                'foreign_sell_volume': None,
                'foreign_sell_value': None,
                'foreign_net_volume': None,
                'foreign_net_value': None,
                'prop_buy_volume': None,
                'prop_buy_value': None,
                'prop_sell_volume': None,
                'prop_sell_value': None,
                'created_at': row_created_at,
            }

        self._merge_daily_metric_history(
            deduped_by_date=deduped_by_date,
            hist=turnover_hist,
            field_mapping={
                'matched_volume': ('matched_volume',),
                'matched_value': ('matched_value',),
                'deal_volume': ('deal_volume',),
                'deal_value': ('deal_value',),
                'total_volume': ('total_volume',),
                'total_value': ('total_value',),
            },
        )
        self._merge_daily_metric_history(
            deduped_by_date=deduped_by_date,
            hist=foreign_hist,
            field_mapping={
                # Provider foreign/proprietary flow endpoints already expose daily totals.
                # We persist those aggregate fields directly; matched/deal stays a separate
                # concern on the price-history endpoint and is not recomputed here.
                'foreign_buy_volume': ('fr_buy_volume', 'foreign_buy_volume'),
                'foreign_buy_value': ('fr_buy_value', 'foreign_buy_value'),
                'foreign_sell_volume': ('fr_sell_volume', 'foreign_sell_volume'),
                'foreign_sell_value': ('fr_sell_value', 'foreign_sell_value'),
                'foreign_net_volume': ('fr_net_volume', 'foreign_net_volume'),
                'foreign_net_value': ('fr_net_value', 'foreign_net_value'),
            },
        )
        self._merge_daily_metric_history(
            deduped_by_date=deduped_by_date,
            hist=prop_hist,
            field_mapping={
                'prop_buy_volume': ('prop_buy_volume',),
                'prop_buy_value': ('prop_buy_value',),
                'prop_sell_volume': ('prop_sell_volume',),
                'prop_sell_value': ('prop_sell_value',),
            },
        )

        if not deduped_by_date:
            return [], None, None

        ordered_dates = sorted(deduped_by_date.keys())
        payload = [deduped_by_date[d] for d in ordered_dates]
        return payload, ordered_dates[0], ordered_dates[-1]

    def _merge_daily_metric_history(
        self,
        deduped_by_date: Dict[date, Dict[str, Any]],
        hist: pd.DataFrame | None,
        field_mapping: Dict[str, tuple[str, ...]],
    ) -> None:
        if hist is None or hist.empty or not deduped_by_date:
            return

        for _, row in hist.iterrows():
            raw_time = row.get('time')
            if pd.isna(raw_time):
                continue

            try:
                history_date = pd.to_datetime(raw_time).date()
            except Exception:
                continue

            target_row = deduped_by_date.get(history_date)
            if target_row is None:
                continue

            for payload_field, candidate_columns in field_mapping.items():
                for column in candidate_columns:
                    if column not in row.index:
                        continue

                    if payload_field.endswith("_volume"):
                        coerced = self._coerce_int(row.get(column))
                    else:
                        coerced = self._coerce_float(row.get(column))

                    if coerced is not None:
                        target_row[payload_field] = coerced
                    break

    async def start_background_workers(self) -> None:
        """No background workers are started for history service."""
        return

    async def stop_background_workers(self) -> None:
        """No-op hook kept for lifecycle symmetry."""
        return

    def enrich_with_price_changes(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """
        Enrich stock data with historical price changes (1w, 1m, 6m, 1y, 2y, 3y).
        """
        return self.enrich_with_price_changes_sync(stocks)

    def enrich_with_price_changes_sync(
        self,
        stocks: List[StockInfo],
        fetch_missing_history: bool = False
    ) -> List[StockInfo]:
        """
        Synchronous fallback for price change enrichment.
        Queries DB cache directly, fetches missing from API.
        """
        # Calculate target dates
        today = datetime.now().date()
        target_dates = {
            '1w': today - timedelta(days=7),
            '1m': today - timedelta(days=30),
            '6m': today - timedelta(days=182),
            '1y': today - timedelta(days=365),
            '2y': today - timedelta(days=730),
            '3y': today - timedelta(days=1095),
        }

        # Use sync connection for DB lookup
        engine = get_sync_engine()

        symbols = [s.ticker[:3] for s in stocks]

        with Session(engine) as session:
            # Get cached prices for all symbols at target dates (with some tolerance)
            cached_prices = self._get_cached_prices_sync(session, symbols, target_dates)

            # Find symbols missing cache data
            symbols_needing_fetch = set()
            for symbol in symbols:
                for period in target_dates.keys():
                    if (symbol, period) not in cached_prices:
                        symbols_needing_fetch.add(symbol)

            # Fetch missing data from API and save to DB
            if symbols_needing_fetch:
                # Limit how many symbols we fetch history for in one request
                # To avoid hitting API limits and long timeouts
                if fetch_missing_history and not sync_status.is_rate_limited and api_circuit_breaker.can_proceed():
                    max_history_fetch = 100
                    symbols_to_fetch = list(symbols_needing_fetch)[:max_history_fetch]
                    logger.info(f"Fetching historical data for {len(symbols_to_fetch)}/{len(symbols_needing_fetch)} symbols")
                    self._fetch_and_cache_history_sync(session, symbols_to_fetch)
                    # Re-query cached prices after fetch
                    cached_prices = self._get_cached_prices_sync(session, symbols, target_dates)
                else:
                    bg_logger.debug("Skipping historical price fetch in request path")

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

            # 6 month change
            if (symbol, '6m') in cached_prices and cached_prices[(symbol, '6m')] > 0:
                six_month_price = cached_prices[(symbol, '6m')]
                stock.price_change_6m = round(((current_price_unit - six_month_price) / six_month_price) * 100, 2)

            # 1 year change
            if (symbol, '1y') in cached_prices and cached_prices[(symbol, '1y')] > 0:
                year_price = cached_prices[(symbol, '1y')]
                stock.price_change_1y = round(((current_price_unit - year_price) / year_price) * 100, 2)

            # 2 year change
            if (symbol, '2y') in cached_prices and cached_prices[(symbol, '2y')] > 0:
                two_year_price = cached_prices[(symbol, '2y')]
                stock.price_change_2y = round(((current_price_unit - two_year_price) / two_year_price) * 100, 2)

            # 3 year change
            if (symbol, '3y') in cached_prices and cached_prices[(symbol, '3y')] > 0:
                three_year_price = cached_prices[(symbol, '3y')]
                stock.price_change_3y = round(((current_price_unit - three_year_price) / three_year_price) * 100, 2)

        return stocks

    async def trigger_missing_price_history_sync(self, stocks: List[StockInfo]) -> bool:
        """
        Deprecated in deterministic sync mode.
        Request-path historical sync is disabled to keep write flow deterministic.
        """
        return False

    def _get_cached_prices_sync(self, session, symbols: List[str], target_dates: Dict[str, date]) -> Dict[tuple, float]:
        """
        Get cached prices for given symbols at target dates.
        Returns dict of (symbol, period) -> close_price
        """
        result: Dict[tuple, float] = {}

        for period, target_date in target_dates.items():
            # Look for prices within 7 days of target (to handle weekends/holidays)
            min_date = target_date - timedelta(days=7)
            max_date = target_date + timedelta(days=1)

            stmt = select(StockDailyHistory).where(
                and_(
                    StockDailyHistory.symbol.in_(symbols),
                    StockDailyHistory.date >= min_date,
                    StockDailyHistory.date <= max_date
                )
            ).order_by(StockDailyHistory.date.desc())

            rows = session.execute(stmt).scalars().all()

            # Group by symbol and take the closest to target date
            symbol_prices: Dict[str, float] = {}
            for row in rows:
                if row.symbol not in symbol_prices:
                    symbol_prices[row.symbol] = row.close

            for symbol, close in symbol_prices.items():
                result[(symbol, period)] = close

        return result

    def _fetch_foreign_trade_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        from vnstock_data import Trading

        span_days = max(1, (end_date - start_date).days + 5)
        trading = Trading(source='VCI', symbol=symbol)
        fetch_method = getattr(trading, "foreign_trade", None)
        if not callable(fetch_method):
            return pd.DataFrame()
        frame = fetch_method(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            resolution='1D',
            limit=max(100, span_days),
        )
        normalized = self._normalize_foreign_trade_history_frame(frame)
        return normalized if normalized is not None else pd.DataFrame()

    def _fetch_prop_trade_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        from vnstock_data import Trading

        span_days = max(1, (end_date - start_date).days + 5)
        trading = Trading(source='VCI', symbol=symbol)
        fetch_method = getattr(trading, "prop_trade", None)
        if not callable(fetch_method):
            return pd.DataFrame()
        frame = fetch_method(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            resolution='1D',
            limit=max(100, span_days),
        )
        normalized = self._normalize_prop_trade_history_frame(frame)
        return normalized if normalized is not None else pd.DataFrame()

    def _fetch_turnover_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        from vnstock_data import Trading

        span_days = max(1, (end_date - start_date).days + 5)
        trading = Trading(source="vci", symbol=symbol, show_log=False)
        frame = trading.price_history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            limit=max(100, span_days),
        )
        normalized = self._normalize_turnover_history_frame(frame)
        return normalized if normalized is not None else pd.DataFrame()

    def _fetch_auxiliary_history_frames(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        log,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        turnover_hist: pd.DataFrame | None = None
        foreign_hist: pd.DataFrame | None = None
        prop_hist: pd.DataFrame | None = None

        try:
            turnover_hist = retry_with_backoff(
                lambda: self._fetch_turnover_history(symbol, start_date, end_date),
                max_retries=2,
            )
            turnover_hist = self._normalize_turnover_history_frame(turnover_hist)
        except Exception as e:
            log.warning(
                "Continuing without turnover enrichment for %s (%s -> %s): %s",
                symbol,
                start_date,
                end_date,
                e,
            )

        try:
            foreign_hist = retry_with_backoff(
                lambda: self._fetch_foreign_trade_history(symbol, start_date, end_date),
                max_retries=2,
            )
            foreign_hist = self._normalize_foreign_trade_history_frame(foreign_hist)
        except Exception as e:
            log.warning(
                "Continuing without foreign flow enrichment for %s (%s -> %s): %s",
                symbol,
                start_date,
                end_date,
                e,
            )

        try:
            prop_hist = retry_with_backoff(
                lambda: self._fetch_prop_trade_history(symbol, start_date, end_date),
                max_retries=2,
            )
            prop_hist = self._normalize_prop_trade_history_frame(prop_hist)
        except Exception as e:
            log.warning(
                "Continuing without proprietary flow enrichment for %s (%s -> %s): %s",
                symbol,
                start_date,
                end_date,
                e,
            )

        return turnover_hist, foreign_hist, prop_hist

    def _upsert_stock_daily_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        session=None,
        raise_on_error: bool = False,
        log=None,
    ) -> int:
        """
        Fetch stock history from API and store in database via atomic upsert.
        Returns number of rows synced (inserted or updated).
        """
        # Use provided session or create a temporary one
        own_session = False
        if session is None:
            engine = get_sync_engine()
            session = Session(engine)
            own_session = True

        symbol_key = symbol[:3].upper()
        symbol_lock = self._get_symbol_sync_lock(symbol_key)
        active_logger = log or bg_logger

        try:
            with symbol_lock:
                # Fetch from API with retry logic
                ohlcv_hist = retry_with_backoff(
                    lambda: self._fetch_ohlcv_history(
                        symbol_key,
                        start_date,
                        end_date,
                        source="VCI",
                    ),
                    max_retries=2,
                )

                if ohlcv_hist is None or ohlcv_hist.empty:
                    return 0

                turnover_hist, foreign_hist, prop_hist = self._fetch_auxiliary_history_frames(
                    symbol=symbol_key,
                    start_date=start_date,
                    end_date=end_date,
                    log=active_logger,
                )

                payload, min_payload_date, max_payload_date = self._build_daily_history_payload(
                    symbol=symbol_key,
                    ohlcv_hist=ohlcv_hist,
                    turnover_hist=turnover_hist,
                    foreign_hist=foreign_hist,
                    prop_hist=prop_hist,
                )
                if not payload:
                    return 0

                insert_stmt = pg_insert(StockDailyHistory.__table__).values(payload)
                upsert_stmt = insert_stmt.on_conflict_do_update(
                    constraint='uq_stock_daily_history_symbol_date',
                    set_={
                        'open': insert_stmt.excluded.open,
                        'high': insert_stmt.excluded.high,
                        'low': insert_stmt.excluded.low,
                        'close': insert_stmt.excluded.close,
                        'volume': insert_stmt.excluded.volume,
                        'matched_volume': self._preserve_existing_on_null(insert_stmt, 'matched_volume'),
                        'matched_value': self._preserve_existing_on_null(insert_stmt, 'matched_value'),
                        'deal_volume': self._preserve_existing_on_null(insert_stmt, 'deal_volume'),
                        'deal_value': self._preserve_existing_on_null(insert_stmt, 'deal_value'),
                        'total_volume': self._preserve_existing_on_null(insert_stmt, 'total_volume'),
                        'total_value': self._preserve_existing_on_null(insert_stmt, 'total_value'),
                        # Preserve existing enrichment when the current sync run
                        # could only fetch OHLCV and auxiliary flow data is absent.
                        'foreign_buy_volume': self._preserve_existing_on_null(insert_stmt, 'foreign_buy_volume'),
                        'foreign_buy_value': self._preserve_existing_on_null(insert_stmt, 'foreign_buy_value'),
                        'foreign_sell_volume': self._preserve_existing_on_null(insert_stmt, 'foreign_sell_volume'),
                        'foreign_sell_value': self._preserve_existing_on_null(insert_stmt, 'foreign_sell_value'),
                        'foreign_net_volume': self._preserve_existing_on_null(insert_stmt, 'foreign_net_volume'),
                        'foreign_net_value': self._preserve_existing_on_null(insert_stmt, 'foreign_net_value'),
                        'prop_buy_volume': self._preserve_existing_on_null(insert_stmt, 'prop_buy_volume'),
                        'prop_buy_value': self._preserve_existing_on_null(insert_stmt, 'prop_buy_value'),
                        'prop_sell_volume': self._preserve_existing_on_null(insert_stmt, 'prop_sell_volume'),
                        'prop_sell_value': self._preserve_existing_on_null(insert_stmt, 'prop_sell_value'),
                    }
                )

                session.execute(upsert_stmt)
                session.commit()
                count = len(payload)
                active_logger.debug(
                    f"Upserted {count} daily history records for {symbol_key} "
                    f"({min_payload_date} -> {max_payload_date})"
                )
                return count
        except Exception as e:
            try:
                session.rollback()
            except Exception:
                pass
            active_logger.error(
                f"Error in _upsert_stock_daily_history for {symbol_key} "
                f"({start_date} -> {end_date}): {e}"
            )
            if raise_on_error:
                raise
            return 0
        finally:
            if own_session:
                session.close()

    def _fetch_and_cache_history_sync(self, session, symbols: List[str]) -> None:
        """
        Fetch historical data for given symbols from vnstock API and cache to DB.
        Optimized version using unified upsert helper.
        """
        today = date.today()
        three_years_ago = today - timedelta(days=1130)

        for symbol in symbols:
            # Check circuit breaker before each symbol to fail fast if rate limited
            if not api_circuit_breaker.can_proceed():
                bg_logger.warning("Circuit breaker open, skipping history fetch for remaining symbols")
                return

            try:
                count = self._upsert_stock_daily_history(
                    symbol=symbol,
                    start_date=three_years_ago,
                    end_date=today,
                    session=session
                )
                if count > 0:
                    bg_logger.debug(f"Cached {count} synced price records for {symbol}")

            except Exception as e:
                try:
                    session.rollback()
                except Exception:
                    pass
                bg_logger.error(f"Error syncing history for {symbol}: {e}")
                continue

    @staticmethod
    def _default_request_sync_metadata() -> Dict[str, Any]:
        return {
            "sync_performed": False,
            "sync_timed_out": False,
            "sync_error": None,
            "updated_through": None,
            "repaired_missing_dates": 0,
        }

    async def _sync_history_for_request(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        if self._request_sync_handler is None:
            return self._default_request_sync_metadata()

        try:
            result = await self._request_sync_handler(
                symbol,
                start_date,
                end_date,
                REQUEST_HISTORY_SYNC_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.warning(
                f"Error running request-path history sync for {symbol}: {e}"
            )
            fallback = self._default_request_sync_metadata()
            fallback["sync_error"] = str(e)[:500]
            return fallback

        merged = self._default_request_sync_metadata()
        if isinstance(result, dict):
            for key in merged.keys():
                if key in result:
                    merged[key] = result[key]
        return merged

    async def get_volume_history(
        self,
        symbol: str,
        days: int = 30,
        auto_sync: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch volume history for a given stock symbol.
        """
        symbol_clean = symbol[:3].upper()
        try:
            safe_days = max(1, int(days))
        except (TypeError, ValueError):
            safe_days = 30

        end_date = date.today()
        start_date = end_date - timedelta(days=safe_days - 1)
        if auto_sync:
            sync_meta = await self._sync_history_for_request(
                symbol=symbol_clean,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            sync_meta = self._default_request_sync_metadata()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            frontend_executor,
            self._fetch_volume_history_sync,
            symbol_clean,
            safe_days
        )
        result.update(sync_meta)
        return result

    async def get_price_history(
        self,
        symbol: str,
        days: int = 30,
        auto_sync: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch price history for a given stock symbol.
        """
        symbol_clean = symbol[:3].upper()
        try:
            safe_days = max(1, int(days))
        except (TypeError, ValueError):
            safe_days = 30

        end_date = date.today()
        start_date = end_date - timedelta(days=safe_days - 1)
        if auto_sync:
            sync_meta = await self._sync_history_for_request(
                symbol=symbol_clean,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            sync_meta = self._default_request_sync_metadata()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            frontend_executor,
            self._fetch_price_history_sync,
            symbol_clean,
            safe_days
        )
        result.update(sync_meta)
        return result

    async def get_price_history_ohlcv(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch full OHLCV history for a given stock symbol from database cache.
        """
        symbol_clean = symbol[:3].upper()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            frontend_executor,
            self._fetch_price_history_ohlcv_sync,
            symbol_clean,
        )

    async def get_stocks_volume_series(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> Dict[str, Any]:
        """
        Fetch volume-value time series for multiple symbols within a date range.
        """
        clean_symbols = list(dict.fromkeys(s[:3].upper() for s in symbols if s))

        if end_date < start_date:
            start_date, end_date = end_date, start_date

        today = date.today()
        bounded_end = min(end_date, today)
        bounded_start = min(start_date, bounded_end)

        if not clean_symbols:
            return {
                "stocks": [],
                "start_date": bounded_start.strftime("%Y-%m-%d"),
                "end_date": bounded_end.strftime("%Y-%m-%d"),
                "is_stale": False,
                "is_syncing": False,
            }

        async with async_session() as session:
            stmt = select(StockDailyHistory).where(
                and_(
                    StockDailyHistory.symbol.in_(clean_symbols),
                    StockDailyHistory.date >= bounded_start,
                    StockDailyHistory.date <= bounded_end,
                )
            ).order_by(StockDailyHistory.symbol.asc(), StockDailyHistory.date.asc())
            result = await session.execute(stmt)
            records = result.scalars().all()

        company_names = await self._get_company_names(clean_symbols)
        series_by_symbol: Dict[str, List[Dict[str, Any]]] = {symbol: [] for symbol in clean_symbols}

        for record in records:
            value = None
            if record.volume is not None and record.close is not None:
                value = round((record.volume * record.close) / 1e6, 2)

            foreign_net_value = self._vnd_to_billion_vnd(record.foreign_net_value)
            if foreign_net_value is None:
                foreign_net_value = self._derive_net_value(
                    record.foreign_buy_value,
                    record.foreign_sell_value,
                )

            prop_net_value = self._derive_net_value(
                record.prop_buy_value,
                record.prop_sell_value,
            )

            series_by_symbol.setdefault(record.symbol, []).append({
                "date": record.date.strftime("%Y-%m-%d"),
                "value": value,
                "foreign_net_value": foreign_net_value,
                "prop_net_value": prop_net_value,
            })

        stale_check_payload = {
            symbol: [{"date": point["date"]} for point in points]
            for symbol, points in series_by_symbol.items()
        }
        stale_symbols = self._get_symbols_with_stale_latest(
            stocks_data=stale_check_payload,
            requested_symbols=clean_symbols,
            end_date=bounded_end,
        )
        is_stale = len(stale_symbols) > 0
        is_syncing_symbols = await self._get_symbols_currently_syncing(clean_symbols)
        is_syncing = len(is_syncing_symbols) > 0

        if stale_symbols:
            triggered = await self._trigger_price_history_sync(
                symbols=stale_symbols,
                start_date=bounded_start,
                end_date=bounded_end,
                force=False,
            )
            if triggered:
                is_syncing = True

        stocks_response = [
            {
                "symbol": symbol,
                "ticker": symbol,
                "company_name": company_names.get(symbol, symbol),
                "data": series_by_symbol.get(symbol, []),
            }
            for symbol in clean_symbols
        ]

        return {
            "stocks": stocks_response,
            "start_date": bounded_start.strftime("%Y-%m-%d"),
            "end_date": bounded_end.strftime("%Y-%m-%d"),
            "is_stale": is_stale,
            "is_syncing": is_syncing,
        }

    async def enrich_with_price_extremes(
        self,
        stocks: List[StockInfo],
        range_start: date | None = None,
        range_end: date | None = None,
    ) -> List[StockInfo]:
        normalized_range = self._normalize_price_extremes_range(range_start, range_end)
        if not stocks or normalized_range is None:
            return stocks

        bounded_start, bounded_end = normalized_range
        symbols = list(dict.fromkeys(stock.ticker[:3].upper() for stock in stocks if stock.ticker))
        if not symbols:
            return stocks

        async with async_session() as session:
            stmt = (
                select(
                    StockDailyHistory.symbol,
                    StockDailyHistory.date,
                    StockDailyHistory.low,
                    StockDailyHistory.high,
                )
                .where(
                    and_(
                        StockDailyHistory.symbol.in_(symbols),
                        StockDailyHistory.date >= bounded_start,
                        StockDailyHistory.date <= bounded_end,
                    )
                )
                .order_by(StockDailyHistory.symbol.asc(), StockDailyHistory.date.desc())
            )
            result = await session.execute(stmt)
            rows = result.all()

        metrics_by_symbol: Dict[str, Dict[str, Any]] = {}
        for symbol, record_date, low, high in rows:
            metrics = metrics_by_symbol.setdefault(
                symbol,
                {
                    "atl_raw": None,
                    "atl_date": None,
                    "ath_raw": None,
                    "ath_date": None,
                },
            )

            if low is not None and (metrics["atl_raw"] is None or low < metrics["atl_raw"]):
                metrics["atl_raw"] = low
                metrics["atl_date"] = record_date

            if high is not None and (metrics["ath_raw"] is None or high > metrics["ath_raw"]):
                metrics["ath_raw"] = high
                metrics["ath_date"] = record_date

        for stock in stocks:
            stock.atl_price = None
            stock.atl_date = None
            stock.atl_diff_pct = None
            stock.ath_price = None
            stock.ath_date = None
            stock.ath_diff_pct = None

            symbol = stock.ticker[:3].upper()
            metrics = metrics_by_symbol.get(symbol)
            if not metrics:
                continue

            current_price_unit = stock.price / 1000 if stock.price is not None else None
            atl_raw = metrics["atl_raw"]
            ath_raw = metrics["ath_raw"]

            if atl_raw is not None:
                stock.atl_price = round(atl_raw * 1000, 2)
                stock.atl_date = metrics["atl_date"].strftime("%Y-%m-%d") if metrics["atl_date"] else None
                if current_price_unit is not None and atl_raw > 0:
                    stock.atl_diff_pct = round(((current_price_unit - atl_raw) / atl_raw) * 100, 2)

            if ath_raw is not None:
                stock.ath_price = round(ath_raw * 1000, 2)
                stock.ath_date = metrics["ath_date"].strftime("%Y-%m-%d") if metrics["ath_date"] else None
                if current_price_unit is not None and ath_raw > 0:
                    stock.ath_diff_pct = round(((current_price_unit - ath_raw) / ath_raw) * 100, 2)

        return stocks

    @staticmethod
    def _classify_recent_trend(return_percent: float | None) -> str | None:
        if return_percent is None:
            return None
        if return_percent > 1:
            return "up"
        if return_percent < -1:
            return "down"
        return "sideways"

    async def enrich_with_recent_trends(self, stocks: List[StockInfo]) -> List[StockInfo]:
        if not stocks:
            return stocks

        symbols = list(dict.fromkeys(stock.ticker[:3].upper() for stock in stocks if stock.ticker))
        if not symbols:
            return stocks

        row_number = func.row_number().over(
            partition_by=StockDailyHistory.symbol,
            order_by=StockDailyHistory.date.desc(),
        ).label("row_number")
        latest_rows = (
            select(
                StockDailyHistory.symbol.label("symbol"),
                StockDailyHistory.date.label("date"),
                StockDailyHistory.close.label("close"),
                row_number,
            )
            .where(StockDailyHistory.symbol.in_(symbols))
            .subquery()
        )

        async with async_session() as session:
            stmt = (
                select(latest_rows.c.symbol, latest_rows.c.date, latest_rows.c.close)
                .where(latest_rows.c.row_number <= 4)
                .order_by(latest_rows.c.symbol.asc(), latest_rows.c.date.asc())
            )
            result = await session.execute(stmt)
            rows = result.all()

        closes_by_symbol: Dict[str, List[float]] = {}
        for symbol, _record_date, close in rows:
            if close is None or close <= 0:
                continue
            closes_by_symbol.setdefault(symbol, []).append(float(close))

        for stock in stocks:
            stock.recent_trend_3d = None
            stock.recent_trend_3d_return = None

            symbol = stock.ticker[:3].upper()
            closes = closes_by_symbol.get(symbol, [])
            current_price = stock.price / 1000 if stock.price is not None else None
            if len(closes) < 4 or current_price is None or current_price <= 0:
                continue
            base = closes[-4]
            if base <= 0:
                continue
            return_percent = round(((current_price / base) - 1) * 100, 2)
            direction = self._classify_recent_trend(return_percent)
            if direction is None:
                continue
            stock.recent_trend_3d = direction
            stock.recent_trend_3d_return = return_percent

        return stocks

    def _fetch_volume_history_sync(self, symbol: str, days: int) -> Dict[str, Any]:
        """Fetch volume history synchronously."""
        symbol_clean = symbol[:3].upper()
        try:
            safe_days = max(1, int(days))
        except (TypeError, ValueError):
            safe_days = 30
        company_name = symbol_clean

        # Use sync connection for DB lookup
        engine = get_sync_engine()

        try:
            with Session(engine) as session:
                # Get company name
                stmt = select(StockCompany).where(StockCompany.symbol == symbol_clean)
                company = session.execute(stmt).scalar_one_or_none()
                if company:
                    company_name = company.company_name

                # Calculate date range
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=safe_days - 1)

                # Query cached data
                stmt = select(StockDailyHistory).where(
                    and_(
                        StockDailyHistory.symbol == symbol_clean,
                        StockDailyHistory.date >= start_date,
                        StockDailyHistory.date <= end_date
                    )
                ).order_by(StockDailyHistory.date.asc())

                cached_records = session.execute(stmt).scalars().all()

                data = []
                for record in cached_records:
                    value = None
                    if record.volume and record.close:
                        # Calculate value in billion VND: (volume * close_price_in_1000_VND) / 1e6
                        value = (record.volume * record.close) / 1e6

                    matched_value = self._vnd_to_billion_vnd(record.matched_value)
                    deal_value = self._vnd_to_billion_vnd(record.deal_value)
                    total_value = self._vnd_to_billion_vnd(record.total_value)
                    if total_value is None and (record.matched_value is not None or record.deal_value is not None):
                        total_value = self._vnd_to_billion_vnd((record.matched_value or 0) + (record.deal_value or 0))

                    total_volume = record.total_volume
                    if total_volume is None and (record.matched_volume is not None or record.deal_volume is not None):
                        total_volume = (record.matched_volume or 0) + (record.deal_volume or 0)

                    foreign_net_value = self._vnd_to_billion_vnd(record.foreign_net_value)
                    if foreign_net_value is None:
                        foreign_net_value = self._derive_net_value(
                            record.foreign_buy_value,
                            record.foreign_sell_value,
                        )

                    prop_buy_value = self._vnd_to_billion_vnd(record.prop_buy_value)
                    prop_sell_value = self._vnd_to_billion_vnd(record.prop_sell_value)

                    data.append({
                        'date': record.date.strftime('%Y-%m-%d'),
                        'volume': record.volume if record.volume else 0,
                        'value': round(value, 2) if value else None,
                        'matched_volume': record.matched_volume,
                        'matched_value': matched_value,
                        'deal_volume': record.deal_volume,
                        'deal_value': deal_value,
                        'total_volume': total_volume,
                        'total_value': total_value,
                        'foreign_net_value': foreign_net_value,
                        'prop_buy_value': prop_buy_value,
                        'prop_sell_value': prop_sell_value,
                        'prop_net_value': self._derive_net_value(
                            record.prop_buy_value,
                            record.prop_sell_value,
                        ),
                    })

                return {
                    'symbol': symbol_clean,
                    'company_name': company_name,
                    'data': data
                }
        except Exception as e:
            logger.warning(f"Error in volume history fetch: {e}")
            return {
                'symbol': symbol_clean,
                'company_name': company_name,
                'data': []
            }
        finally:
            engine.dispose()

    def _fetch_price_history_sync(self, symbol: str, days: int) -> Dict[str, Any]:
        """Fetch price history synchronously."""
        symbol_clean = symbol[:3].upper()
        try:
            safe_days = max(1, int(days))
        except (TypeError, ValueError):
            safe_days = 30
        company_name = symbol_clean

        # Use sync connection for DB lookup
        engine = get_sync_engine()

        try:
            with Session(engine) as session:
                # Get company name
                stmt = select(StockCompany).where(StockCompany.symbol == symbol_clean)
                company = session.execute(stmt).scalar_one_or_none()
                if company:
                    company_name = company.company_name

                # Calculate date range
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=safe_days - 1)

                # Query cached data
                stmt = select(StockDailyHistory).where(
                    and_(
                        StockDailyHistory.symbol == symbol_clean,
                        StockDailyHistory.date >= start_date,
                        StockDailyHistory.date <= end_date
                    )
                ).order_by(StockDailyHistory.date.asc())

                cached_records = session.execute(stmt).scalars().all()

                data = [
                    {
                        'date': record.date.strftime('%Y-%m-%d'),
                        # StockDailyHistory.close is stored in 1,000 VND; convert to VND for UI parity.
                        'close': round(record.close * 1000, 2),
                    }
                    for record in cached_records
                ]

                return {
                    'symbol': symbol_clean,
                    'company_name': company_name,
                    'data': data
                }
        except Exception as e:
            logger.warning(f"Error in price history fetch: {e}")
            return {
                'symbol': symbol_clean,
                'company_name': company_name,
                'data': []
            }
        finally:
            engine.dispose()

    def _fetch_price_history_ohlcv_sync(self, symbol: str) -> Dict[str, Any]:
        """Fetch full OHLCV history synchronously from DB."""
        symbol_clean = symbol[:3].upper()
        company_name = symbol_clean

        engine = get_sync_engine()

        try:
            with Session(engine) as session:
                stmt = select(StockCompany).where(StockCompany.symbol == symbol_clean)
                company = session.execute(stmt).scalar_one_or_none()
                if company:
                    company_name = company.company_name

                stmt = select(StockDailyHistory).where(
                    StockDailyHistory.symbol == symbol_clean
                ).order_by(StockDailyHistory.date.desc())
                cached_records = session.execute(stmt).scalars().all()

                data = [
                    {
                        'date': record.date.strftime('%Y-%m-%d'),
                        'open': record.open,
                        'high': record.high,
                        'low': record.low,
                        'close': record.close,
                        'volume': record.volume,
                    }
                    for record in cached_records
                ]

                return {
                    'symbol': symbol_clean,
                    'company_name': company_name,
                    'data': data,
                }
        except Exception as e:
            logger.warning(f"Error in OHLCV history fetch: {e}")
            return {
                'symbol': symbol_clean,
                'company_name': company_name,
                'data': [],
            }
        finally:
            engine.dispose()

    async def get_stocks_weekly_prices(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        include_benchmarks: bool = True
    ) -> Dict[str, Any]:
        """
        Get weekly price data for multiple stocks.
        Returns cached data immediately and triggers background sync if stale.
        """
        # Clean symbols (use first 3 chars)
        clean_symbols = list(dict.fromkeys(s[:3].upper() for s in symbols if s))

        if end_date < start_date:
            start_date, end_date = end_date, start_date

        today = date.today()
        bounded_end = min(end_date, today)
        bounded_start = min(start_date, bounded_end)

        # Load from database
        stocks_data = await self._load_weekly_prices_from_db(clean_symbols, bounded_start, bounded_end)

        # Check freshness from latest locally available date only.
        stale_latest_symbols = self._get_symbols_with_stale_latest(stocks_data, clean_symbols, bounded_end)
        is_stale = len(stale_latest_symbols) > 0

        # Load benchmarks if requested
        benchmarks = {}
        if include_benchmarks:
            benchmarks = await self._load_benchmark_prices(bounded_start, bounded_end)

        # Get company names
        company_names = await self._get_company_names(clean_symbols)

        # Format response
        stocks_response = []
        for symbol in clean_symbols:
            prices = stocks_data.get(symbol, [])
            stocks_response.append({
                'symbol': symbol,
                'ticker': symbol,
                'company_name': company_names.get(symbol, symbol),
                'prices': prices
            })

        is_syncing_symbols = await self._get_symbols_currently_syncing(clean_symbols)
        is_syncing = len(is_syncing_symbols) > 0

        if stale_latest_symbols:
            triggered = await self._trigger_price_history_sync(
                symbols=stale_latest_symbols,
                start_date=bounded_start,
                end_date=bounded_end,
                force=False
            )
            if triggered:
                is_syncing = True

        return {
            'stocks': stocks_response,
            'benchmarks': benchmarks,
            'start_date': bounded_start.strftime('%Y-%m-%d'),
            'end_date': bounded_end.strftime('%Y-%m-%d'),
            'is_stale': is_stale,
            'is_syncing': is_syncing
        }

    async def _get_company_names(self, symbols: List[str]) -> Dict[str, str]:
        """Get company names for given symbols from database."""
        async with async_session() as session:
            stmt = select(StockCompany).where(StockCompany.symbol.in_(symbols))
            result = await session.execute(stmt)
            companies = result.scalars().all()
            return {c.symbol: c.company_name for c in companies}

    async def _load_weekly_prices_from_db(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load daily prices from database and aggregate to weekly.
        Uses Friday as the weekly reference point.
        """
        async with async_session() as session:
            stmt = select(StockDailyHistory).where(
                and_(
                    StockDailyHistory.symbol.in_(symbols),
                    StockDailyHistory.date >= start_date,
                    StockDailyHistory.date <= end_date
                )
            ).order_by(StockDailyHistory.symbol, StockDailyHistory.date)

            result = await session.execute(stmt)
            records = result.scalars().all()

            # Group by symbol
            symbol_data: Dict[str, List[Dict[str, Any]]] = {}
            for record in records:
                if record.symbol not in symbol_data:
                    symbol_data[record.symbol] = []
                symbol_data[record.symbol].append({
                    'date': record.date,
                    'close': record.close
                })

            # Aggregate to weekly using pandas
            weekly_data: Dict[str, List[Dict[str, Any]]] = {}
            for symbol, daily_prices in symbol_data.items():
                if not daily_prices:
                    weekly_data[symbol] = []
                    continue

                df = pd.DataFrame(daily_prices)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()

                # Resample to weekly (Friday close) - 'W-FRI' means week ending on Friday
                weekly_df = df.resample('W-FRI').last().dropna()

                # Ensure we strictly respect the start_date after resampling
                weekly_df = weekly_df[weekly_df.index >= pd.Timestamp(start_date)]

                weekly_data[symbol] = [
                    {
                        'date': idx.strftime('%Y-%m-%d'),
                        'close': float(row['close'])
                    }
                    for idx, row in weekly_df.iterrows()
                ]

            return weekly_data

    def _check_prices_staleness(
        self,
        stocks_data: Dict[str, List[Dict[str, Any]]],
        requested_symbols: List[str],
        end_date: date
    ) -> bool:
        """
        Check if price data is stale for requested symbols.
        Returns True if any requested stock:
        - Has no data
        - Latest date is >7 days old
        """
        return len(self._get_symbols_with_stale_latest(stocks_data, requested_symbols, end_date)) > 0

    def _get_symbols_with_stale_latest(
        self,
        stocks_data: Dict[str, List[Dict[str, Any]]],
        requested_symbols: List[str],
        end_date: date
    ) -> List[str]:
        if not requested_symbols:
            return []

        stale_symbols: List[str] = []
        stale_threshold = end_date - timedelta(days=7)

        for symbol in requested_symbols:
            prices = stocks_data.get(symbol, [])
            if not prices:
                stale_symbols.append(symbol)
                continue

            latest_date_str = prices[-1]['date']
            try:
                latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d').date()
            except (TypeError, ValueError):
                stale_symbols.append(symbol)
                continue

            if latest_date < stale_threshold:
                stale_symbols.append(symbol)

        return stale_symbols

    async def _load_benchmark_prices(
        self,
        start_date: date,
        end_date: date
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load weekly prices for VNINDEX and VN30 benchmarks.
        Fetches from vnstock API since these are index values, not stock prices.
        """
        benchmarks = {}
        loop = asyncio.get_event_loop()

        for index_symbol in ['VNINDEX', 'VN30']:
            try:
                prices = await loop.run_in_executor(
                    None,
                    self._fetch_index_history_sync,
                    index_symbol,
                    start_date,
                    end_date
                )
                if prices:
                    benchmarks[index_symbol] = prices
            except Exception as e:
                logger.warning(f"Error fetching benchmark {index_symbol}: {e}")

        return benchmarks

    def _fetch_index_history_sync(
        self,
        index_symbol: str,
        start_date: date,
        end_date: date
    ) -> List[Dict[str, Any]]:
        """Fetch historical index values and aggregate to weekly."""
        from vnstock import Vnstock

        # Check circuit breaker before making API call
        if not api_circuit_breaker.can_proceed():
            raise CircuitOpenError(f"Circuit breaker open - cannot fetch index history for {index_symbol}")

        try:
            vs = Vnstock(symbol=index_symbol, source='VCI')
            stock = vs.stock()
            df = stock.quote.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1D'
            )
            api_circuit_breaker.record_success()

            if df is not None and not df.empty:
                # Convert to proper format
                df['date'] = pd.to_datetime(df['time'])
                df = df.set_index('date').sort_index()

                # Resample to weekly (Friday close)
                weekly_df = df[['close']].resample('W-FRI').last().dropna()

                # Ensure we strictly respect the start_date after resampling
                weekly_df = weekly_df[weekly_df.index >= pd.Timestamp(start_date)]

                return [
                    {
                        'date': idx.strftime('%Y-%m-%d'),
                        'close': float(row['close'])
                    }
                    for idx, row in weekly_df.iterrows()
                ]
        except CircuitOpenError:
            raise
        except (SystemExit, Exception) as e:
            if _is_rate_limit_error(e):
                _record_rate_limit(reset_seconds=30.0)
                raise CircuitOpenError(f"Rate limited fetching index history for {index_symbol}: {e}")
            logger.warning(f"Error fetching index history for {index_symbol}: {e}")

        return []

    async def _trigger_price_history_sync(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        force: bool = False
    ) -> bool:
        """
        Trigger delegated price sync for stale symbols.
        Returns True if sync was triggered or currently running.
        """
        if not symbols:
            return False

        normalized_symbols = list(dict.fromkeys(s[:3].upper() for s in symbols if s))
        if not normalized_symbols:
            return False

        now = datetime.utcnow()
        cooldown_state: Dict[str, datetime] = {}
        syncing_symbols = await self._get_symbols_currently_syncing(normalized_symbols)
        if not force:
            cooldown_state = await self._get_weekly_sync_cooldown_state(normalized_symbols)

        symbols_to_sync: List[str] = []
        for symbol in normalized_symbols:
            if symbol in syncing_symbols:
                continue

            if not force:
                last_attempt = cooldown_state.get(symbol)
                if last_attempt and (now - last_attempt) < self._weekly_prices_retry_cooldown:
                    continue

            symbols_to_sync.append(symbol)

        if not symbols_to_sync:
            return len(syncing_symbols) > 0

        if self._weekly_sync_trigger_handler is None:
            logger.warning("Weekly sync trigger handler is not configured; skipping delegated sync")
            return len(syncing_symbols) > 0

        await self._set_weekly_sync_cooldown_state(
            symbols=symbols_to_sync,
            attempted_at=now,
        )

        try:
            result = await self._weekly_sync_trigger_handler(symbols_to_sync)
        except Exception as e:
            logger.warning(f"Error triggering delegated weekly price sync: {e}")
            return len(syncing_symbols) > 0

        if not isinstance(result, dict):
            return True

        started = bool(result.get("started"))
        state = str(result.get("state", "")).strip().lower()
        return started or state == "running"

    async def _get_weekly_sync_cooldown_state(
        self,
        symbols: List[str]
    ) -> Dict[str, datetime]:
        if not symbols:
            return {}

        try:
            async with async_session() as session:
                stmt = select(
                    StockHistorySyncState.symbol,
                    StockHistorySyncState.weekly_sync_last_attempt_at,
                ).where(StockHistorySyncState.symbol.in_(symbols))
                result = await session.execute(stmt)
                rows = result.all()

            state: Dict[str, datetime] = {}
            for symbol, last_attempt_at in rows:
                if last_attempt_at is not None:
                    state[symbol] = last_attempt_at
            return state
        except Exception as e:
            logger.warning(f"Error loading weekly sync cooldown state: {e}")
            return {}

    async def _set_weekly_sync_cooldown_state(
        self,
        symbols: List[str],
        attempted_at: datetime,
    ) -> None:
        if not symbols:
            return

        try:
            async with async_session() as session:
                stmt = select(StockHistorySyncState).where(
                    StockHistorySyncState.symbol.in_(symbols)
                )
                existing_rows = await session.execute(stmt)
                existing = {row.symbol: row for row in existing_rows.scalars().all()}

                for symbol in symbols:
                    state = existing.get(symbol)
                    if state is None:
                        state = StockHistorySyncState(symbol=symbol)
                        session.add(state)
                        existing[symbol] = state

                    state.weekly_sync_last_attempt_at = attempted_at
                    state.updated_at = attempted_at

                await session.commit()
        except Exception as e:
            logger.warning(f"Error updating weekly sync cooldown state: {e}")

    async def _get_symbols_currently_syncing(self, symbols: List[str]) -> Set[str]:
        if not symbols:
            return set()

        # Ignore stale DB "running" flags when price sync runtime is not active.
        if not sync_status.history_sync.is_running:
            return set()

        try:
            async with async_session() as session:
                stmt = select(StockHistorySyncState.symbol).where(
                    and_(
                        StockHistorySyncState.symbol.in_(symbols),
                        StockHistorySyncState.sync_status == "running",
                    )
                )
                result = await session.execute(stmt)
                return {row[0] for row in result.all() if row[0]}
        except Exception as e:
            logger.warning(f"Error loading currently syncing symbols: {e}")
            return set()
