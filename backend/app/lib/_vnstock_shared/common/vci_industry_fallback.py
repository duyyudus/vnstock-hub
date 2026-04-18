"""Shared VCI industry fallback helpers."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
import time
from typing import Any

import pandas as pd

from app.lib._vnstock_shared.core.utils.client import send_request
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib._vnstock_shared.core.utils.user_agent import get_headers


logger = get_logger(__name__)

_SCREENER_CRITERIA_URL = "https://iq.vietcap.com.vn/api/iq-insight-service/v1/screening/criteria"
_SCREENING_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
    "Origin": "https://trading.vietcap.com.vn",
    "Referer": "https://trading.vietcap.com.vn/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Content-Type": "application/json",
}
_CACHE_TTL_SECONDS = 6 * 3600
_SPECIAL_FAMILY_CODE_MAP = {
    "0500": "0001",
    "8300": "8301",
}

_cache_lock = Lock()
_criteria_cache: dict[str, Any] = {"timestamp": 0.0, "value": None}
_fallback_cache: dict[str, Any] = {"timestamp": 0.0, "value": None}


@dataclass(frozen=True)
class VCIIndustryFallbackData:
    industries_icb: pd.DataFrame
    symbols_by_level2: pd.DataFrame


def invalidate_vci_industry_fallback_cache() -> None:
    """Reset process-local fallback caches for tests and diagnostics."""
    with _cache_lock:
        _criteria_cache["timestamp"] = 0.0
        _criteria_cache["value"] = None
        _fallback_cache["timestamp"] = 0.0
        _fallback_cache["value"] = None


def get_vci_screener_criteria(
    *,
    random_agent: bool = False,
    show_log: bool = False,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Fetch and cache VCI screener criteria used for sector dictionaries."""
    current_time = time.time()
    with _cache_lock:
        cached = _criteria_cache["value"]
        cache_age = current_time - float(_criteria_cache["timestamp"])
        if not force_refresh and cached is not None and cache_age < _CACHE_TTL_SECONDS:
            return cached

    headers = get_headers(data_source="VCI", random_agent=random_agent)
    headers.update(_SCREENING_HEADERS)
    payload = send_request(
        url=_SCREENER_CRITERIA_URL,
        headers=headers,
        method="GET",
        show_log=show_log,
    )
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        raise ValueError("Không thể lấy dữ liệu criteria.")

    with _cache_lock:
        _criteria_cache["timestamp"] = current_time
        _criteria_cache["value"] = data
    return data


def build_vci_industry_fallback(
    symbols_by_exchange: pd.DataFrame,
    *,
    criteria_raw: list[dict[str, Any]] | None = None,
    random_agent: bool = False,
    show_log: bool = False,
    force_refresh: bool = False,
) -> VCIIndustryFallbackData:
    """Build and cache level-1/2 VCI industry metadata from working endpoints."""
    current_time = time.time()
    with _cache_lock:
        cached = _fallback_cache["value"]
        cache_age = current_time - float(_fallback_cache["timestamp"])
        if not force_refresh and cached is not None and cache_age < _CACHE_TTL_SECONDS:
            return cached

    if criteria_raw is None:
        criteria_raw = get_vci_screener_criteria(
            random_agent=random_agent,
            show_log=show_log,
            force_refresh=force_refresh,
        )

    level1_by_code = _extract_criteria_options(criteria_raw, "sectorLv1")
    level2_by_code = _extract_criteria_options(criteria_raw, "sector")
    normalized_symbols = _normalize_symbols_by_exchange(symbols_by_exchange)

    industries_rows: list[dict[str, Any]] = []
    seen_level1_codes: set[str] = set()
    symbols_rows: list[dict[str, Any]] = []

    for row in normalized_symbols.itertuples(index=False):
        level2_code = row.icb_code2
        level2_meta = level2_by_code.get(level2_code)
        if level2_meta is None:
            continue

        level1_code = _infer_family_code(level2_code, level1_by_code)
        level1_meta = level1_by_code.get(level1_code, {"vi": "", "en": ""})

        if level1_code and level1_code not in seen_level1_codes:
            industries_rows.append(
                {
                    "icb_name": level1_meta["vi"],
                    "en_icb_name": level1_meta["en"],
                    "icb_code": level1_code,
                    "level": 1,
                }
            )
            seen_level1_codes.add(level1_code)

        industries_rows.append(
            {
                "icb_name": level2_meta["vi"],
                "en_icb_name": level2_meta["en"],
                "icb_code": level2_code,
                "level": 2,
            }
        )
        symbols_rows.append(
            {
                "symbol": row.symbol,
                "organ_name": row.organ_name,
                "en_organ_name": row.en_organ_name,
                "com_type_code": row.com_type_code,
                "icb_code1": level1_code,
                "icb_name1": level1_meta["vi"],
                "en_icb_name1": level1_meta["en"],
                "icb_code2": level2_code,
                "icb_name2": level2_meta["vi"],
                "en_icb_name2": level2_meta["en"],
            }
        )

    industries_df = pd.DataFrame(industries_rows)
    if not industries_df.empty:
        industries_df = (
            industries_df.drop_duplicates(subset=["icb_code", "level"])
            .sort_values(by=["level", "icb_code"])
            .reset_index(drop=True)
        )
    else:
        industries_df = pd.DataFrame(columns=["icb_name", "en_icb_name", "icb_code", "level"])

    symbols_df = pd.DataFrame(symbols_rows)
    if not symbols_df.empty:
        symbols_df = symbols_df.sort_values(by=["symbol"]).reset_index(drop=True)
    else:
        symbols_df = pd.DataFrame(
            columns=[
                "symbol",
                "organ_name",
                "en_organ_name",
                "com_type_code",
                "icb_code1",
                "icb_name1",
                "en_icb_name1",
                "icb_code2",
                "icb_name2",
                "en_icb_name2",
            ]
        )

    industries_df.source = "VCI"
    symbols_df.source = "VCI"
    fallback = VCIIndustryFallbackData(industries_icb=industries_df, symbols_by_level2=symbols_df)

    with _cache_lock:
        _fallback_cache["timestamp"] = current_time
        _fallback_cache["value"] = fallback

    if show_log:
        logger.warning(
            "Falling back to reconstructed VCI industries (%s industries, %s symbol mappings).",
            len(industries_df),
            len(symbols_df),
        )

    return fallback


def _extract_criteria_options(criteria_raw: list[dict[str, Any]], field_name: str) -> dict[str, dict[str, str]]:
    for item in criteria_raw:
        if item.get("name") != field_name:
            continue
        options = item.get("conditionOptions")
        if not isinstance(options, list):
            break
        return {
            str(option.get("value") or "").strip(): {
                "vi": str(option.get("viName") or "").strip(),
                "en": str(option.get("enName") or "").strip(),
            }
            for option in options
            if str(option.get("value") or "").strip()
        }
    raise ValueError(f"Không tìm thấy criteria {field_name}.")


def _infer_family_code(level2_code: str, level1_by_code: dict[str, dict[str, str]]) -> str:
    if level2_code in level1_by_code:
        return level2_code
    special_family_code = _SPECIAL_FAMILY_CODE_MAP.get(level2_code)
    if special_family_code:
        return special_family_code
    if level2_code and level2_code[0].isdigit():
        family_code = f"{level2_code[0]}000"
        if family_code in level1_by_code:
            return family_code
    return ""


def _normalize_symbols_by_exchange(symbols_by_exchange: pd.DataFrame) -> pd.DataFrame:
    if symbols_by_exchange is None or symbols_by_exchange.empty:
        raise ValueError("Không tìm thấy dữ liệu symbols_by_exchange.")

    required_columns = {"symbol", "organ_name", "icb_code2"}
    missing_columns = required_columns.difference(symbols_by_exchange.columns)
    if missing_columns:
        raise ValueError(f"symbols_by_exchange thiếu cột bắt buộc: {sorted(missing_columns)}")

    df = symbols_by_exchange.copy()
    df["symbol"] = df["symbol"].fillna("").astype(str).str.strip().str.upper()
    df["organ_name"] = df["organ_name"].fillna("").astype(str).str.strip()
    df["icb_code2"] = df["icb_code2"].fillna("").astype(str).str.strip()
    if "type" in df.columns:
        df["type"] = df["type"].fillna("").astype(str).str.strip()
        df = df[df["type"].str.upper() == "STOCK"].copy()
    else:
        df["type"] = ""

    df = df[(df["symbol"] != "") & (df["icb_code2"] != "")].copy()
    df["en_organ_name"] = df["organ_name"]
    df["com_type_code"] = df["type"]
    return df[["symbol", "organ_name", "en_organ_name", "com_type_code", "icb_code2"]].drop_duplicates(
        subset=["symbol"]
    )
