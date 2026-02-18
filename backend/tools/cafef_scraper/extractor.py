from __future__ import annotations

import html
import re
from datetime import datetime
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

from tools.cafef_scraper.classifier import normalize_text
from tools.cafef_scraper.types import ArticleDetail, SourceType

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

_TITLE_META_PATTERN = re.compile(
    r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_URL_META_PATTERN = re.compile(
    r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_CANONICAL_LINK_PATTERN = re.compile(
    r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_TITLE_TAG_PATTERN = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_JSON_DATE_PATTERN = re.compile(r'"datePublished"\s*:\s*"([^"]+)"', re.IGNORECASE)
_META_PUBLISHED_PATTERN = re.compile(
    r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_PDATE_PATTERN = re.compile(
    r'<span[^>]+class=["\'][^"\']*pdate[^"\']*["\'][^>]*>(.*?)</span>',
    re.IGNORECASE | re.DOTALL,
)
_GENERIC_DATE_PATTERN = re.compile(
    r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?:\s*[-\s]\s*(\d{1,2}):(\d{2})(?:\s*(AM|PM))?)?",
    re.IGNORECASE,
)
_HREF_PATTERN = re.compile(
    r"<a[^>]+href=(?:\"([^\"]+)\"|'([^']+)')[^>]*>",
    re.IGNORECASE,
)
_ARTICLE_ID_PATTERNS = (
    re.compile(r"-(\d+)\.chn(?:\?|$)", re.IGNORECASE),
    re.compile(r"-(\d+)(?:/|\?|$)", re.IGNORECASE),
)


def extract_article_id(url: str) -> str | None:
    for pattern in _ARTICLE_ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None


def extract_title(html_text: str) -> str | None:
    meta_match = _TITLE_META_PATTERN.search(html_text)
    if meta_match:
        return _clean_title(meta_match.group(1))

    title_match = _TITLE_TAG_PATTERN.search(html_text)
    if title_match:
        return _clean_title(title_match.group(1))
    return None


def _clean_title(raw: str) -> str:
    text = html.unescape(raw).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\|\s*CafeF\.vn$", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_published_at(html_text: str) -> datetime | None:
    for pattern in (_JSON_DATE_PATTERN, _META_PUBLISHED_PATTERN):
        match = pattern.search(html_text)
        if not match:
            continue
        parsed = _parse_iso_datetime(match.group(1))
        if parsed:
            return parsed

    pdate_match = _PDATE_PATTERN.search(html_text)
    if pdate_match:
        raw = html.unescape(pdate_match.group(1))
        parsed = _parse_loose_datetime(raw)
        if parsed:
            return parsed

    return None


def _parse_iso_datetime(value: str) -> datetime | None:
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=VN_TZ)
    return dt.astimezone(VN_TZ)


def _parse_loose_datetime(value: str) -> datetime | None:
    cleaned = html.unescape(value)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    date_match = _GENERIC_DATE_PATTERN.search(cleaned)
    if not date_match:
        return None
    day = int(date_match.group(1))
    month = int(date_match.group(2))
    year = int(date_match.group(3))
    hour = int(date_match.group(4) or "0")
    minute = int(date_match.group(5) or "0")
    marker = (date_match.group(6) or "").upper()
    if marker == "PM" and hour < 12:
        hour += 12
    if marker == "AM" and hour == 12:
        hour = 0
    try:
        return datetime(year, month, day, hour, minute, tzinfo=VN_TZ)
    except ValueError:
        return None


def extract_pdf_urls(html_text: str, base_url: str) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()
    for match in _HREF_PATTERN.finditer(html_text):
        href = match.group(1) or match.group(2) or ""
        href = html.unescape(href.strip())
        if not href:
            continue
        if "/download/" not in href and ".pdf" not in href.lower():
            continue
        absolute = normalize_url(urljoin(base_url, href))
        if absolute in seen:
            continue
        if not _is_allowed_pdf_host(absolute):
            continue
        seen.add(absolute)
        results.append(absolute)
    return results


def extract_canonical_url(html_text: str, fallback_url: str) -> str:
    for pattern in (_URL_META_PATTERN, _CANONICAL_LINK_PATTERN):
        match = pattern.search(html_text)
        if not match:
            continue
        candidate = normalize_url(html.unescape(match.group(1).strip()))
        if candidate and ".chn" in candidate:
            return candidate
    return normalize_url(fallback_url)


def _is_allowed_pdf_host(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return "cafef.vn" in host or "mediacdn.vn" in host


def normalize_url(url: str) -> str:
    normalized = url.strip()
    if normalized.startswith("//"):
        normalized = "https:" + normalized
    if normalized.startswith("http://"):
        normalized = "https://" + normalized[len("http://") :]
    return normalized


def parse_article_detail(html_text: str, url: str, source_type: SourceType) -> ArticleDetail:
    title = extract_title(html_text)
    article_url = extract_canonical_url(html_text, fallback_url=url)
    return ArticleDetail(
        url=article_url,
        source_type=source_type,
        article_id=extract_article_id(article_url),
        title=title,
        normalized_title=normalize_text(title),
        published_at=extract_published_at(html_text),
        pdf_urls=extract_pdf_urls(html_text, base_url=article_url),
    )
