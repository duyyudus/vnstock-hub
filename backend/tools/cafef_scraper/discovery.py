from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable
from datetime import date
from urllib.parse import urljoin

EVENT_FEED_URL = "https://cafef.vn/du-lieu/Ajax/Events_RelatedNews_New.aspx"
SITEMAP_INDEX_URL = "https://cafef.vn/sitemap.xml"
BASE_URL = "https://cafef.vn"

_HREF_CHN_PATTERN = re.compile(
    r"<a[^>]+href=(?:\"([^\"]+\.chn[^\"]*)\"|'([^']+\.chn[^']*)')[^>]*>",
    re.IGNORECASE,
)
_SITEMAP_YEAR_PATTERN = re.compile(r"sitemaps-(\d{4})-", re.IGNORECASE)

TextFetcher = Callable[[str, str], Awaitable[str | None]]
EventPageCallback = Callable[[int, list[str]], None]
SitemapCallback = Callable[[int, int, list[str]], None]


def parse_event_feed_urls(html_fragment: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in _HREF_CHN_PATTERN.finditer(html_fragment):
        href = html.unescape(match.group(1) or match.group(2) or "").strip()
        if not href:
            continue
        absolute = _normalize_article_url(urljoin(BASE_URL, href))
        if absolute in seen:
            continue
        seen.add(absolute)
        urls.append(absolute)
    return urls


def parse_sitemap_locs(xml_text: str) -> list[str]:
    if not xml_text.strip():
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    locs: list[str] = []
    for elem in root.iter():
        if elem.tag.endswith("loc") and elem.text:
            loc = elem.text.strip()
            if loc:
                locs.append(loc)
    return locs


def _normalize_article_url(url: str) -> str:
    normalized = url.strip()
    if normalized.startswith("http://"):
        normalized = "https://" + normalized[len("http://") :]
    return normalized


def build_symbol_probe_url(article_id: int, symbol: str = "E1VFVN30") -> str:
    symbol_slug = symbol.lower()
    return f"{BASE_URL}/du-lieu/{symbol_slug}-{article_id}/{symbol_slug}.chn"


def _is_symbol_article(url: str, symbol: str) -> bool:
    lower = url.lower()
    if symbol.lower() not in lower:
        return False
    return "cafef.vn" in lower and ".chn" in lower


def _sitemap_in_year_range(url: str, start_date: date, end_date: date) -> bool:
    match = _SITEMAP_YEAR_PATTERN.search(url)
    if not match:
        return True
    year = int(match.group(1))
    return start_date.year <= year <= end_date.year


class CafefDiscovery:
    def __init__(
        self,
        fetch_text: TextFetcher,
        symbol: str = "E1VFVN30",
        logger: logging.Logger | None = None,
    ):
        self._fetch_text = fetch_text
        self._symbol = symbol
        self._logger = logger

    async def fetch_event_feed_urls(
        self,
        max_pages: int = 300,
        page_size: int = 30,
        start_page: int = 1,
        on_page: EventPageCallback | None = None,
    ) -> int:
        all_urls: list[str] = []
        seen_urls: set[str] = set()
        repeated_signature_count = 0
        previous_signature: tuple[str, str, int] | None = None

        for page_index in range(start_page, start_page + max_pages):
            params = (
                f"?symbol={self._symbol}&floorID=0&configID=0&PageIndex={page_index}"
                f"&PageSize={page_size}&Type=2"
            )
            url = EVENT_FEED_URL + params
            body = await self._fetch_text(url, "event_feed")
            if not body:
                break

            page_urls = [u for u in parse_event_feed_urls(body) if _is_symbol_article(u, self._symbol)]
            if not page_urls:
                break

            signature = (page_urls[0], page_urls[-1], len(page_urls))
            if signature == previous_signature:
                repeated_signature_count += 1
                if repeated_signature_count >= 2:
                    break
            else:
                repeated_signature_count = 0
            previous_signature = signature

            new_count = 0
            page_new_urls: list[str] = []
            for url in page_urls:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                all_urls.append(url)
                page_new_urls.append(url)
                new_count += 1

            if on_page:
                on_page(page_index, page_new_urls)

            if self._logger and ((page_index - start_page + 1) % 10 == 0 or page_index == start_page):
                self._logger.info(
                    "Discovery[event_feed] page=%s collected=%s new_in_page=%s",
                    page_index,
                    len(all_urls),
                    new_count,
                )

            if new_count == 0:
                repeated_signature_count += 1
                if repeated_signature_count >= 2:
                    break

        return len(all_urls)

    async def fetch_sitemap_article_urls(
        self,
        start_date: date,
        end_date: date,
        max_sitemaps: int | None = None,
        start_index: int = 0,
        on_sitemap: SitemapCallback | None = None,
    ) -> int:
        index_body = await self._fetch_text(SITEMAP_INDEX_URL, "sitemap_index")
        if not index_body:
            return 0

        sitemap_urls = [
            url
            for url in parse_sitemap_locs(index_body)
            if _sitemap_in_year_range(url, start_date, end_date)
        ]
        if max_sitemaps is not None:
            sitemap_urls = sitemap_urls[:max_sitemaps]
        total_count = len(sitemap_urls)
        if start_index > 0:
            sitemap_urls = sitemap_urls[start_index:]

        article_urls: list[str] = []
        seen_urls: set[str] = set()

        for offset, sitemap_url in enumerate(sitemap_urls, start=1):
            index = start_index + offset
            sitemap_body = await self._fetch_text(sitemap_url, "sitemap_page")
            if not sitemap_body:
                continue
            new_count = 0
            page_new_urls: list[str] = []
            for loc in parse_sitemap_locs(sitemap_body):
                normalized = _normalize_article_url(loc)
                if normalized in seen_urls:
                    continue
                if not _is_symbol_article(normalized, self._symbol):
                    continue
                seen_urls.add(normalized)
                article_urls.append(normalized)
                page_new_urls.append(normalized)
                new_count += 1

            if on_sitemap:
                on_sitemap(index, total_count, page_new_urls)

            if self._logger and (index % 25 == 0 or index == 1):
                self._logger.info(
                    "Discovery[sitemap] sitemap=%s/%s collected=%s new_in_sitemap=%s",
                    index,
                    total_count,
                    len(article_urls),
                    new_count,
                )

        return len(article_urls)
