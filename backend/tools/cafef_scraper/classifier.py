from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime

from tools.cafef_scraper.types import DocType

_BASKET_TOKENS = ("thong bao", "danh muc", "co cau", "hoan doi")
_SWAP_END_TOKENS = ("ket thuc", "giao dich", "hoan doi")

_DATE_PATTERNS = (
    re.compile(r"ngay[\s:_-]*(\d{1,2})[/-](\d{1,2})[/-](\d{4})", re.IGNORECASE),
    re.compile(r"ngay[\s:_-]*(\d{1,2})[-](\d{1,2})[-](\d{2})", re.IGNORECASE),
    re.compile(r"ngay[\s:_-]*(\d{1,2})(\d{1,2})(\d{4})", re.IGNORECASE),
    re.compile(r"ngay[\s:_-]*(\d{1,2})\s+(\d{1,2})\s+(\d{4})", re.IGNORECASE),
)

_NON_WORD = re.compile(r"[^a-z0-9]+")
_RAW_DATE_PATTERNS = (
    re.compile(r"ngay[\s:_-]*(\d{1,2})[./-](\d{1,2})[./-](\d{4})", re.IGNORECASE),
    re.compile(r"(20\d{2})(\d{2})(\d{2})"),
)


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    text = value.lower().strip()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = _NON_WORD.sub(" ", text)
    return " ".join(text.split())


def slugify(value: str | None, max_len: int = 80) -> str:
    normalized = normalize_text(value)
    if not normalized:
        return "unknown"
    slug = normalized.replace(" ", "-")
    return slug[:max_len].strip("-") or "unknown"


def _contains_tokens(text: str, tokens: tuple[str, ...]) -> bool:
    return all(token in text for token in tokens)


def classify_doc_type(*texts: str | None) -> DocType | None:
    merged = normalize_text(" ".join([t for t in texts if t]))
    if not merged:
        return None
    if _contains_tokens(merged, _BASKET_TOKENS):
        return DocType.BASKET_NOTICE
    if _contains_tokens(merged, _SWAP_END_TOKENS):
        return DocType.SWAP_END
    return None


def likely_target_article_url(url: str) -> bool:
    normalized = normalize_text(url)
    if "e1vfvn30" not in normalized:
        return False
    if "hoan doi" not in normalized:
        return False

    if classify_doc_type(url) is not None:
        return True

    # Broader fallback for legacy title-slug variants.
    is_swap_end = "ket thuc" in normalized and "giao dich" in normalized
    is_basket_notice = (
        ("co cau" in normalized and "danh muc" in normalized)
        or ("co cau" in normalized and "thong bao" in normalized)
    )
    return is_swap_end or is_basket_notice


def extract_event_date(
    *texts: str | None,
    published_at: datetime | None,
) -> tuple[date | None, bool]:
    raw_combined = " ".join([t for t in texts if t])
    normalized = normalize_text(raw_combined)

    parsed = _extract_from_patterns(normalized, _DATE_PATTERNS)
    if parsed is not None:
        return parsed, False

    parsed_raw = _extract_from_raw_patterns(raw_combined)
    if parsed_raw is not None:
        return parsed_raw, False

    if published_at:
        return published_at.date(), True
    return None, False


def _extract_from_patterns(text: str, patterns: tuple[re.Pattern[str], ...]) -> date | None:
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))
        if year < 100:
            year += 2000
        parsed = _safe_date(year=year, month=month, day=day)
        if parsed:
            return parsed
    return None


def _extract_from_raw_patterns(text: str) -> date | None:
    for pattern in _RAW_DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if pattern.pattern.startswith("ngay"):
            day = int(match.group(1))
            month = int(match.group(2))
            year = int(match.group(3))
            parsed = _safe_date(year=year, month=month, day=day)
        else:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            parsed = _safe_date(year=year, month=month, day=day)
        if parsed:
            return parsed
    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None
